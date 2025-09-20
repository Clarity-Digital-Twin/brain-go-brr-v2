"""Performance tests for inference latency and real-time processing capability."""

import gc
import time

import numpy as np
import pytest
import torch

from src.brain_brr.config.schemas import (
    DecoderConfig,
    EncoderConfig,
    MambaConfig,
    ModelConfig,
    ResCNNConfig,
)
from src.brain_brr.models import SeizureDetector


@pytest.mark.serial
class TestInferenceLatency:
    """Test inference latency for real-time processing requirements."""

    @pytest.fixture(scope="class")
    def production_model(self):
        """Create production-sized model for benchmarking."""
        config = ModelConfig(
            encoder=EncoderConfig(channels=[64, 128, 256, 512], stages=4),
            rescnn=ResCNNConfig(n_blocks=3, kernel_sizes=[3, 5, 7]),
            mamba=MambaConfig(n_layers=6, d_model=512, d_state=16, conv_kernel=5),
            decoder=DecoderConfig(stages=4, kernel_size=4),
        )
        model = SeizureDetector.from_config(config)
        model.eval()

        # Move to CUDA if available for performance testing
        if torch.cuda.is_available():
            model = model.cuda()

        return model

    @pytest.mark.performance
    def test_single_window_latency(self, production_model, benchmark_timer):
        """Test latency for processing a single 60s window."""
        # 60s window at 256Hz
        window = torch.randn(1, 19, 15360)

        # Move to same device as model
        device = next(production_model.parameters()).device
        window = window.to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = production_model(window)

        # Benchmark
        with benchmark_timer, torch.no_grad():
            for _ in range(100):
                _ = production_model(window)

        p95_latency_ms = benchmark_timer.p95 * 1000 / 100  # Convert to ms per inference
        median_latency_ms = benchmark_timer.median * 1000 / 100

        # Requirements: <100ms for real-time processing
        assert p95_latency_ms < 100, f"P95 latency {p95_latency_ms:.1f}ms exceeds 100ms target"
        assert median_latency_ms < 50, (
            f"Median latency {median_latency_ms:.1f}ms exceeds 50ms target"
        )

    @pytest.mark.performance
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_batch_inference_latency(self, production_model, batch_size, benchmark_timer):
        """Test latency scaling with batch size."""
        window = torch.randn(batch_size, 19, 15360)

        # Move to same device as model
        device = next(production_model.parameters()).device
        window = window.to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = production_model(window)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(20):
                start = time.perf_counter()
                _ = production_model(window)
                times.append(time.perf_counter() - start)

        avg_time_per_sample = np.mean(times) * 1000 / batch_size  # ms per sample

        # Should have sub-linear scaling
        assert avg_time_per_sample < 100, (
            f"Batch {batch_size}: {avg_time_per_sample:.1f}ms per sample"
        )

    @pytest.mark.performance
    def test_streaming_latency(self, production_model):
        """Test latency for streaming inference with overlapping windows."""
        # Simulate streaming with 10s stride
        total_duration = 600  # 10 minutes
        window_size = 60  # 60s windows
        stride = 10  # 10s stride
        sample_rate = 256

        n_windows = (total_duration - window_size) // stride + 1
        window_samples = window_size * sample_rate

        # Get device
        device = next(production_model.parameters()).device

        latencies = []

        with torch.no_grad():
            for i in range(n_windows):
                window = torch.randn(1, 19, window_samples).to(device)

                start = time.perf_counter()
                _ = production_model(window)
                latency = time.perf_counter() - start
                latencies.append(latency)

                # Real-time constraint: must process faster than stride
                if i > 10:  # After warmup
                    assert latency < stride, f"Window {i}: {latency:.2f}s > {stride}s stride"

        # Check overall statistics
        p95 = np.percentile(latencies[10:], 95)  # Exclude warmup
        assert p95 < stride * 0.5, f"P95 latency {p95:.2f}s too high for {stride}s stride"

    @pytest.mark.performance
    @pytest.mark.gpu
    def test_gpu_vs_cpu_speedup(self, production_model):
        """Test GPU provides significant speedup over CPU."""
        window = torch.randn(1, 19, 15360)

        # CPU timing
        model_cpu = production_model.cpu()
        window_cpu = window.cpu()

        cpu_times = []
        with torch.no_grad():
            for _ in range(5):
                start = time.perf_counter()
                _ = model_cpu(window_cpu)
                cpu_times.append(time.perf_counter() - start)

        cpu_time = np.median(cpu_times)

        # GPU timing (if available)
        if torch.cuda.is_available():
            model_gpu = production_model.cuda()
            window_gpu = window.cuda()

            # Warmup
            for _ in range(10):
                _ = model_gpu(window_gpu)
            torch.cuda.synchronize()

            gpu_times = []
            with torch.no_grad():
                for _ in range(20):
                    start = time.perf_counter()
                    _ = model_gpu(window_gpu)
                    torch.cuda.synchronize()
                    gpu_times.append(time.perf_counter() - start)

            gpu_time = np.median(gpu_times)

            speedup = cpu_time / gpu_time
            assert speedup > 5, f"GPU speedup only {speedup:.1f}x (expected >5x)"

    @pytest.mark.performance
    def test_model_compilation_speedup(self, production_model):
        """Test torch.compile optimization provides speedup."""
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile not available")

        window = torch.randn(1, 19, 15360)

        # Baseline timing
        baseline_times = []
        with torch.no_grad():
            for _ in range(20):
                start = time.perf_counter()
                _ = production_model(window)
                baseline_times.append(time.perf_counter() - start)

        baseline_time = np.median(baseline_times)

        # Compiled model timing
        compiled_model = torch.compile(production_model, mode="reduce-overhead")

        # Warmup compilation
        with torch.no_grad():
            for _ in range(5):
                _ = compiled_model(window)

        compiled_times = []
        with torch.no_grad():
            for _ in range(20):
                start = time.perf_counter()
                _ = compiled_model(window)
                compiled_times.append(time.perf_counter() - start)

        compiled_time = np.median(compiled_times)

        speedup = baseline_time / compiled_time
        assert speedup > 1.2, f"Compilation speedup only {speedup:.1f}x (expected >1.2x)"


@pytest.mark.serial
class TestThroughput:
    """Test throughput for batch processing scenarios."""

    @pytest.mark.performance
    def test_hourly_throughput(self, minimal_model):
        """Test processing throughput for 1 hour of data."""
        # 1 hour = 3600s, with 60s windows and 10s stride = 354 windows
        n_windows = 354
        window_size = 15360  # 60s at 256Hz

        # Get device
        device = next(minimal_model.parameters()).device

        start_time = time.perf_counter()

        with torch.no_grad():
            for i in range(n_windows):
                window = torch.randn(1, 19, window_size).to(device)
                _ = minimal_model(window)

                # Garbage collect periodically
                if i % 50 == 0:
                    gc.collect()

        total_time = time.perf_counter() - start_time

        # Should process 1 hour of data in less than 5 minutes
        assert total_time < 300, f"Processing 1 hour took {total_time:.1f}s (>5 min)"

        throughput = 3600 / total_time  # Hours of data per hour of compute
        assert throughput > 10, f"Throughput {throughput:.1f}x realtime (expected >10x)"

    @pytest.mark.performance
    def test_daily_batch_throughput(self, minimal_model):
        """Test throughput for processing 24 hours of data."""
        # Simulate batch processing of daily data
        hours_per_day = 24
        windows_per_hour = 354  # With 10s stride
        batch_size = 8

        total_windows = hours_per_day * windows_per_hour
        n_batches = total_windows // batch_size

        start_time = time.perf_counter()

        with torch.no_grad():
            for i in range(min(n_batches, 100)):  # Test subset for speed
                batch = torch.randn(batch_size, 19, 15360)
                _ = minimal_model(batch)

                if i % 10 == 0:
                    gc.collect()

        subset_time = time.perf_counter() - start_time

        # Extrapolate to full day
        estimated_full_time = subset_time * (n_batches / min(n_batches, 100))

        # Should process 24 hours in less than 1 hour
        assert estimated_full_time < 3600, (
            f"Processing 24 hours estimated at {estimated_full_time / 3600:.1f} hours"
        )


@pytest.mark.serial
class TestLatencyUnderLoad:
    """Test latency stability under various load conditions."""

    @pytest.mark.performance
    def test_latency_stability(self, minimal_model):
        """Test latency remains stable over extended operation."""
        window = torch.randn(1, 19, 15360)
        latencies = []

        with torch.no_grad():
            for i in range(500):
                start = time.perf_counter()
                _ = minimal_model(window)
                latencies.append(time.perf_counter() - start)

                if i % 100 == 0:
                    gc.collect()

        # Calculate statistics
        latencies = latencies[100:]  # Remove warmup
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        cv = std_latency / mean_latency  # Coefficient of variation

        # Latency should be stable (low variance)
        assert cv < 0.2, f"Latency variance too high: CV={cv:.2f} (expected <0.2)"

        # No degradation over time
        early = np.mean(latencies[:100])
        late = np.mean(latencies[-100:])
        degradation = (late - early) / early

        assert abs(degradation) < 0.1, f"Latency degraded by {degradation * 100:.1f}% over time"

    @pytest.mark.performance
    def test_concurrent_inference(self, minimal_model):
        """Test latency with concurrent inference requests."""
        import queue
        import threading

        window = torch.randn(1, 19, 15360)
        n_threads = 4
        n_requests_per_thread = 25
        latency_queue = queue.Queue()

        def inference_worker():
            local_latencies = []
            with torch.no_grad():
                for _ in range(n_requests_per_thread):
                    start = time.perf_counter()
                    _ = minimal_model(window.clone())
                    local_latencies.append(time.perf_counter() - start)
            latency_queue.put(local_latencies)

        # Run concurrent inference
        threads = []
        start_time = time.perf_counter()

        for _ in range(n_threads):
            t = threading.Thread(target=inference_worker)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        total_time = time.perf_counter() - start_time

        # Collect all latencies
        all_latencies = []
        while not latency_queue.empty():
            all_latencies.extend(latency_queue.get())

        # Check performance under concurrent load
        p95_latency = np.percentile(all_latencies, 95)
        median_latency = np.median(all_latencies)

        # Should maintain reasonable latency even under concurrent load
        assert p95_latency < 0.5, f"P95 latency {p95_latency:.2f}s under concurrent load"
        assert median_latency < 0.2, f"Median latency {median_latency:.2f}s under concurrent load"

        # Check throughput improvement
        sequential_time = median_latency * n_threads * n_requests_per_thread
        speedup = sequential_time / total_time
        assert speedup > n_threads * 0.5, (
            f"Concurrent speedup only {speedup:.1f}x with {n_threads} threads"
        )
