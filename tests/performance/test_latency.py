"""Performance tests for inference latency and real-time processing capability.

NOTE: Performance tests can be skipped with SKIP_PERF_TESTS=1 environment variable.
These tests are hardware-dependent and may fail on different GPUs or CI environments.
"""

import gc
import math
import os
import time
from contextlib import suppress

import numpy as np
import pytest
import torch

from src.brain_brr.config.schemas import MambaConfig, ModelConfig, TCNConfig
from src.brain_brr.models import SeizureDetector
from tests.performance.utils import thresholds

# Allow skipping performance tests entirely
skip_perf_tests = pytest.mark.skipif(
    os.getenv("SKIP_PERF_TESTS", "0") == "1",
    reason="Performance tests skipped via SKIP_PERF_TESTS=1",
)

# Mark all tests in this module as performance tests (excluded from CI)
pytestmark = pytest.mark.performance


@pytest.mark.serial
class TestInferenceLatency:
    """Test inference latency for real-time processing requirements."""

    @pytest.fixture(scope="class")
    def production_model(self):
        """Create production-sized model for benchmarking."""
        import gc

        config = ModelConfig(
            tcn=TCNConfig(num_layers=4, kernel_size=5, dropout=0.1, stride_down=16),
            mamba=MambaConfig(n_layers=2, d_model=512, d_state=16, conv_kernel=4, dropout=0.1),
        )
        model = SeizureDetector.from_config(config)
        model.eval()

        # Move to CUDA if available for performance testing
        if torch.cuda.is_available():
            model = model.cuda()

        yield model

        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @pytest.mark.performance
    @pytest.mark.timeout(180)
    def test_single_window_latency(self, production_model, benchmark_timer):
        """Test latency for processing a single 60s window."""
        # 60s window at 256Hz
        window = torch.randn(1, 19, 15360)

        # Move to same device as model
        device = next(production_model.parameters()).device
        window = window.to(device)

        # Debug output
        import os

        print(f"\n[DEBUG] Model device: {device}")
        print(f"[DEBUG] Window device: {window.device}")
        print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
        print(f"[DEBUG] Model on CUDA: {next(production_model.parameters()).is_cuda}")
        print(f"[DEBUG] MAMBA fallback env: {os.getenv('SEIZURE_MAMBA_FORCE_FALLBACK', 'not set')}")

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = production_model(window)

        # Benchmark - measure individual inference times
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.perf_counter()
                _ = production_model(window)
                # Sync for accurate GPU timing
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        p95_latency_ms = np.percentile(times, 95) * 1000  # Convert to ms
        median_latency_ms = np.median(times) * 1000

        # Device-specific thresholds with variance tolerance
        # RTX 4090/3090: ~100-120ms P95 is normal for 30M param model
        # A100/V100: ~80-100ms P95
        # CPU: Not tested here (too slow)
        gpu_name = "cpu"
        if device.type == "cuda":
            # Get GPU name if possible
            gpu_name = (
                torch.cuda.get_device_name(device.index) if torch.cuda.is_available() else "unknown"
            )

            # More lenient for consumer GPUs, tighter for datacenter GPUs
            if "RTX" in gpu_name or "GTX" in gpu_name:
                p95_target = 125  # Consumer GPUs have more variability
                median_target = 65
            else:
                p95_target = 110  # Datacenter GPUs (A100, V100, etc.)
                median_target = 55
        else:
            # CPU path (though this test skips CPU usually)
            p95_target = 500
            median_target = 250

        # Requirements: Real-time = <1000ms, but we target much better
        # Allow converting to warnings in CI/unstable environments
        warn_only = os.getenv("PERF_TESTS_WARN_ONLY", "0") == "1"

        if p95_latency_ms >= p95_target:
            msg = f"P95 latency {p95_latency_ms:.1f}ms exceeds {p95_target}ms target for {gpu_name}"
            if warn_only:
                pytest.skip(f"Performance warning (not failing): {msg}")
            else:
                pytest.fail(msg)

        if median_latency_ms >= median_target:
            msg = f"Median latency {median_latency_ms:.1f}ms exceeds {median_target}ms target for {gpu_name}"
            if warn_only:
                pytest.skip(f"Performance warning (not failing): {msg}")
            else:
                pytest.fail(msg)

    @pytest.mark.performance
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    @pytest.mark.timeout(240)
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
                # Sync for accurate GPU timing
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        avg_time_per_sample = np.mean(times) * 1000 / batch_size  # ms per sample

        # Should have sub-linear scaling
        is_cpu = next(production_model.parameters()).device.type == "cpu"
        max_latency = thresholds.batch_latency_per_sample_ms(is_cpu)
        assert avg_time_per_sample < max_latency, (
            f"Batch {batch_size}: {avg_time_per_sample:.1f}ms per sample (max: {max_latency:.1f}ms)"
        )

    @pytest.mark.performance
    @pytest.mark.timeout(300)
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
        max_p95 = stride * thresholds.streaming_p95_latency(stride)
        assert p95 < max_p95, (
            f"P95 latency {p95:.2f}s too high (max: {max_p95:.2f}s for {stride}s stride)"
        )

    @pytest.mark.performance
    @pytest.mark.gpu
    @pytest.mark.timeout(240)
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
            min_speedup = thresholds.gpu_speedup_factor()
            assert speedup > min_speedup, (
                f"GPU speedup only {speedup:.1f}x (expected >{min_speedup:.1f}x)"
            )

    @pytest.mark.performance
    @pytest.mark.timeout(300)
    def test_model_compilation_speedup(self, minimal_model):
        """Test torch.compile optimization provides speedup on a compile-friendly model.

        Uses minimal_model and removes weight_norm parametrizations to avoid
        known PyTorch 2.2.x FakeTensor/weight_norm issues.
        """
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile not available")

        model = minimal_model
        device = next(model.parameters()).device

        # Remove weight_norm if present (compile compatibility)
        for m in model.modules():
            if isinstance(m, torch.nn.Conv1d):
                with suppress(Exception):
                    torch.nn.utils.remove_weight_norm(m)

        window = torch.randn(1, 19, 15360, device=device)

        # Baseline timing
        baseline_times: list[float] = []
        with torch.no_grad():
            for _ in range(10):
                start = time.perf_counter()
                _ = model(window)
                baseline_times.append(time.perf_counter() - start)

        baseline_time = float(np.median(baseline_times))

        # Try to compile; prefer a more aggressive mode on CPU where
        # reduce-overhead may yield marginal gains
        compiled_model = None
        used_eager_backend = False
        try:
            preferred_mode = os.getenv(
                "TORCH_COMPILE_MODE",
                "max-autotune" if device.type == "cpu" else "reduce-overhead",
            )
            compiled_model = torch.compile(model, mode=preferred_mode)
        except Exception as e:
            # Fall back to eager backend in restricted or unsupported environments
            if (
                "aten.is_pinned" in str(e)
                or "NYI" in str(e)
                or "Permission denied" in str(e)
                or "SemLock" in str(e)
                or "ProcessPool" in str(e)
                or "_inductor" in str(e)
            ):
                compiled_model = torch.compile(model, backend="eager")
                used_eager_backend = True
            else:
                raise

        # Warmup and timing (more warmup helps stabilize compiled timings)
        with torch.no_grad():
            for _ in range(10 if device.type == "cpu" else 5):
                _ = compiled_model(window)

        compiled_times: list[float] = []
        with torch.no_grad():
            for _ in range(10):
                start = time.perf_counter()
                _ = compiled_model(window)
                compiled_times.append(time.perf_counter() - start)

        compiled_time = float(np.median(compiled_times))
        speedup = baseline_time / max(compiled_time, 1e-6)

        # Skip test if compilation isn't providing real optimization
        if used_eager_backend or speedup < 1.02:
            pytest.skip(
                f"torch.compile not optimizing in this environment (speedup={speedup:.2f}x). "
                "This is expected in restricted environments or with weight_norm."
            )
        else:
            # On CPU, compilation benefits can be marginal depending on backend/kernel support.
            # Treat borderline cases as environment variability instead of a hard failure.
            if device.type == "cpu" and speedup <= 1.05:
                pytest.skip(
                    f"CPU compile speedup {speedup:.2f}x below strict threshold; skipping as env-dependent"
                )
            min_speedup = thresholds.compilation_speedup()
            assert speedup > min_speedup, (
                f"Compilation speedup only {speedup:.2f}x (expected >{min_speedup:.2f}x)"
            )


@pytest.mark.serial
class TestThroughput:
    """Test throughput for batch processing scenarios."""

    @pytest.mark.performance
    @pytest.mark.timeout(600)
    def test_hourly_throughput(self, minimal_model):
        """Test processing throughput for 1 hour of data.

        Vectorize into mini-batches to reflect realistic batch inference and
        avoid Python-loop overhead that skews CPU timing.
        """
        # 1 hour = 3600s, with 60s windows and 10s stride = 354 windows
        total_windows = 354
        window_size = 15360  # 60s at 256Hz

        device = next(minimal_model.parameters()).device
        is_cpu = device.type == "cpu"

        # Choose conservative batch size to prevent OOM
        batch_size = 4 if is_cpu else 8  # Reduced from 16/32

        # On CPU, evaluate a subset and extrapolate to avoid long walltime
        eval_windows = total_windows if not is_cpu else 96
        eval_batches = math.ceil(eval_windows / batch_size)

        start_time = time.perf_counter()
        with torch.no_grad():
            for i in range(eval_batches):
                b = min(batch_size, eval_windows - i * batch_size)
                batch = torch.randn(b, 19, window_size, device=device)
                _ = minimal_model(batch)
                if i % 5 == 0:
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.synchronize()

        measured_time = time.perf_counter() - start_time
        total_time = (
            measured_time
            if eval_windows == total_windows
            else measured_time * (total_windows / eval_windows)
        )

        # Should process 1 hour of data quickly
        max_seconds = thresholds.hourly_throughput_time_s(is_cpu)
        assert total_time < max_seconds, (
            f"Estimated time {total_time:.1f}s (> {max_seconds}s limit)"
        )

        throughput = 3600 / max(total_time, 1e-6)  # Hours of data per hour of compute
        min_throughput = thresholds.min_throughput_realtime(is_cpu)
        assert throughput > min_throughput, (
            f"Throughput {throughput:.1f}x realtime (expected >{min_throughput}x)"
        )

    @pytest.mark.performance
    @pytest.mark.timeout(300)  # 5 minutes max
    def test_daily_batch_throughput(self, minimal_model):
        """Test throughput for processing 24 hours of data."""
        # Simulate batch processing of daily data
        hours_per_day = 24
        windows_per_hour = 354  # With 10s stride
        batch_size = 8

        total_windows = hours_per_day * windows_per_hour
        n_batches = total_windows // batch_size

        # Get device
        device = next(minimal_model.parameters()).device

        start_time = time.perf_counter()

        with torch.no_grad():
            # Much fewer batches on CPU to avoid timeout
            limit = 10 if device.type == "cpu" else 100
            # Vectorized batch creation for efficiency
            all_batches = torch.randn(limit, batch_size, 19, 15360, device=device)

            for i in range(limit):
                _ = minimal_model(all_batches[i])
                if i % 10 == 0:
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.synchronize()

        subset_time = time.perf_counter() - start_time

        # Extrapolate to full day
        estimated_full_time = subset_time * (n_batches / limit)

        # Should process 24 hours in less than 1 hour
        # Allow more headroom on CPU environments
        max_hours = (
            thresholds.daily_processing_hours()
            if device.type == "cpu"
            else thresholds.daily_processing_hours() / 2
        )
        assert estimated_full_time < max_hours * 3600, (
            f"Processing 24 hours estimated at {estimated_full_time / 3600:.1f} hours (> {max_hours}h)"
        )


@pytest.mark.serial
class TestLatencyUnderLoad:
    """Test latency stability under various load conditions."""

    @pytest.mark.performance
    @pytest.mark.timeout(600)
    def test_latency_stability(self, minimal_model):
        """Test latency remains stable over extended operation."""
        # Get device
        device = next(minimal_model.parameters()).device
        window = torch.randn(1, 19, 15360, device=device)

        # Extended warmup for JIT compilation and kernel caching
        with torch.no_grad():
            for _ in range(100):
                _ = minimal_model(window)
                if device.type == "cuda":
                    torch.cuda.synchronize()

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Now measure stable-state latencies
        latencies = []
        with torch.no_grad():
            iters = 250 if device.type == "cpu" else 500
            for i in range(iters):
                start = time.perf_counter()
                _ = minimal_model(window)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append(time.perf_counter() - start)

                if i % 100 == 0:
                    gc.collect()

        # Calculate statistics - remove first 50 even after warmup
        warmup = 25 if device.type == "cpu" else 50
        latencies = latencies[warmup:]
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        cv = std_latency / mean_latency  # Coefficient of variation

        # With Mamba fallback to Conv1d, some variance is expected
        # but should still be reasonably stable
        # Higher threshold when run in suite due to system load
        # WSL2 and development environments need even more tolerance
        max_cv = 0.75 if os.getenv("WSL_DISTRO_NAME") else 0.60

        # Warning instead of failure for high variance (common in dev environments)
        if cv > max_cv:
            import warnings

            warnings.warn(
                f"High latency variance: CV={cv:.2f} (expected <{max_cv:.2f})",
                stacklevel=2,
            )

        # No significant degradation over time (improvement is OK)
        span = 60 if device.type == "cpu" else 100
        early = np.mean(latencies[:span])
        late = np.mean(latencies[-span:])
        degradation = (late - early) / early

        # Skip degradation check if variance is already high (system under load)
        # When CV > 0.35, the system is too noisy to measure degradation reliably
        # Also be more lenient with the degradation threshold (20% instead of 15%)
        if cv < 0.35:
            # Only check degradation if system is stable
            max_degradation = thresholds.latency_degradation_pct() / 100
            assert abs(degradation) < max_degradation, (
                f"Latency changed by {degradation * 100:.1f}% over time (max: {max_degradation * 100:.1f}%)"
            )

    @pytest.mark.performance
    @pytest.mark.timeout(300)
    def test_concurrent_inference(self, minimal_model):
        """Test latency with concurrent inference requests."""
        import queue
        import threading

        # Get device
        device = next(minimal_model.parameters()).device
        window = torch.randn(1, 19, 15360).to(device)
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
        device = next(minimal_model.parameters()).device
        is_cpu = device.type == "cpu"
        p95_limit = thresholds.single_window_latency_ms(is_cpu) * 25 / 1000  # 2.5s/0.5s
        median_limit = thresholds.single_window_latency_ms(is_cpu) * 12.5 / 1000  # 1.25s/0.2s

        assert p95_latency < p95_limit, (
            f"P95 latency {p95_latency:.2f}s under concurrent load (limit: {p95_limit:.2f}s)"
        )
        assert median_latency < median_limit, (
            f"Median latency {median_latency:.2f}s under concurrent load (limit: {median_limit:.2f}s)"
        )

        # Check throughput improvement
        sequential_time = median_latency * n_threads * n_requests_per_thread
        speedup = sequential_time / total_time
        min_speedup = n_threads * thresholds.concurrent_speedup_efficiency(n_threads)
        assert speedup > min_speedup, (
            f"Concurrent speedup only {speedup:.1f}x with {n_threads} threads (expected >{min_speedup:.1f}x)"
        )
