"""Memory profiling tests for Brain-Go-Brr v2."""

import gc
import os
import tracemalloc

import psutil
import pytest
import torch

from src.brain_brr.config.schemas import (
    DecoderConfig,
    EncoderConfig,
    MambaConfig,
    ModelConfig,
    ResidualCNNConfig,
)
from src.brain_brr.models import SeizureDetector


class TestMemoryUsage:
    """Profile memory consumption across different scenarios."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up memory before and after each test."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_memory_usage(self) -> tuple[float, float]:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / 1024 / 1024

        gpu_mb = 0
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024

        return ram_mb, gpu_mb

    @pytest.mark.performance
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
    def test_inference_memory_scaling(self, minimal_model, batch_size):
        """Test memory scales linearly with batch size."""
        window_size = 15360  # 60s at 256Hz

        # Baseline memory
        gc.collect()
        baseline_ram, baseline_gpu = self.get_memory_usage()

        # Create batch
        batch = torch.randn(batch_size, 19, window_size)

        # Run inference
        with torch.no_grad():
            output = minimal_model(batch)

        # Measure peak memory
        peak_ram, peak_gpu = self.get_memory_usage()

        ram_used = peak_ram - baseline_ram
        gpu_used = peak_gpu - baseline_gpu

        # Memory per sample
        ram_per_sample = ram_used / batch_size
        gpu_per_sample = gpu_used / batch_size if gpu_used > 0 else 0

        # Should use <500MB RAM per sample
        assert ram_per_sample < 500, (
            f"RAM usage {ram_per_sample:.1f}MB per sample exceeds 500MB limit"
        )

        # GPU memory should be reasonable if available
        if gpu_used > 0:
            assert gpu_per_sample < 200, (
                f"GPU usage {gpu_per_sample:.1f}MB per sample exceeds 200MB limit"
            )

        # Clean up
        del batch, output

    @pytest.mark.performance
    def test_model_memory_footprint(self):
        """Test model memory footprint for different configurations."""
        configs = [
            (
                "minimal",
                ModelConfig(
                    encoder=EncoderConfig(channels=[32, 64, 128, 256], stages=4),
                    rescnn=ResidualCNNConfig(n_blocks=1, kernel_sizes=[3]),
                    mamba=MambaConfig(n_layers=1, d_model=256, d_state=8, d_conv=5),
                    decoder=DecoderConfig(stages=4, kernel_size=4),
                ),
            ),
            (
                "standard",
                ModelConfig(
                    encoder=EncoderConfig(channels=[64, 128, 256, 512], stages=4),
                    rescnn=ResidualCNNConfig(n_blocks=3, kernel_sizes=[3, 5, 7]),
                    mamba=MambaConfig(n_layers=6, d_model=512, d_state=16, d_conv=5),
                    decoder=DecoderConfig(stages=4, kernel_size=4),
                ),
            ),
        ]

        for name, config in configs:
            gc.collect()
            baseline_ram, _ = self.get_memory_usage()

            model = SeizureDetector(config)

            model_ram, _ = self.get_memory_usage()
            footprint = model_ram - baseline_ram

            # Calculate parameter count
            n_params = sum(p.numel() for p in model.parameters())
            expected_mb = n_params * 4 / 1024 / 1024  # 4 bytes per float32

            # Model footprint should be close to parameter size
            assert footprint < expected_mb * 3, (
                f"{name} model uses {footprint:.1f}MB (expected ~{expected_mb:.1f}MB)"
            )

            del model
            gc.collect()

    @pytest.mark.performance
    def test_memory_leak_detection(self, minimal_model):
        """Test for memory leaks during repeated inference."""
        window = torch.randn(1, 19, 15360)

        # Initial warmup
        with torch.no_grad():
            for _ in range(10):
                _ = minimal_model(window)

        gc.collect()
        initial_ram, initial_gpu = self.get_memory_usage()

        # Run many iterations
        with torch.no_grad():
            for i in range(100):
                output = minimal_model(window)
                del output  # Explicit cleanup

                if i % 20 == 0:
                    gc.collect()

        gc.collect()
        final_ram, final_gpu = self.get_memory_usage()

        # Memory should not grow significantly
        ram_growth = final_ram - initial_ram
        gpu_growth = final_gpu - initial_gpu

        assert ram_growth < 50, f"RAM grew by {ram_growth:.1f}MB (possible leak)"

        if torch.cuda.is_available():
            assert gpu_growth < 10, f"GPU memory grew by {gpu_growth:.1f}MB (possible leak)"

    @pytest.mark.performance
    def test_gradient_memory(self, minimal_model):
        """Test memory usage during training (with gradients)."""
        batch_size = 4
        window = torch.randn(batch_size, 19, 15360, requires_grad=False)
        labels = torch.randn(batch_size, 15360)

        # Forward pass without gradients
        gc.collect()
        baseline_ram, baseline_gpu = self.get_memory_usage()

        with torch.no_grad():
            output_no_grad = minimal_model(window)

        no_grad_ram, no_grad_gpu = self.get_memory_usage()

        del output_no_grad
        gc.collect()

        # Forward pass with gradients
        minimal_model.train()
        output = minimal_model(window)
        loss = torch.nn.functional.mse_loss(output, labels)

        before_backward_ram, before_backward_gpu = self.get_memory_usage()

        # Backward pass
        loss.backward()

        after_backward_ram, after_backward_gpu = self.get_memory_usage()

        # Calculate memory overhead
        forward_overhead = (before_backward_ram - no_grad_ram) / no_grad_ram
        backward_overhead = (after_backward_ram - before_backward_ram) / no_grad_ram

        # Gradient memory should be reasonable
        assert forward_overhead < 2.0, (
            f"Forward pass with gradients uses {forward_overhead * 100:.0f}% more memory"
        )
        assert backward_overhead < 3.0, (
            f"Backward pass uses {backward_overhead * 100:.0f}% additional memory"
        )

    @pytest.mark.performance
    def test_cache_memory_usage(self, tmp_path):
        """Test memory usage of caching mechanisms."""

        # Create mock dataset with caching
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Simulate cached data
        n_windows = 1000
        window_size = 15360

        gc.collect()
        baseline_ram, _ = self.get_memory_usage()

        # Create cached tensors
        cached_data = []
        for i in range(n_windows):
            tensor = torch.randn(19, window_size)
            cached_data.append(tensor)

            # Check memory periodically
            if i % 100 == 0:
                current_ram, _ = self.get_memory_usage()
                used = current_ram - baseline_ram
                per_window = used / (i + 1)

                # Each window should use ~1.2MB (19*15360*4 bytes)
                expected_per_window = 19 * window_size * 4 / 1024 / 1024
                assert per_window < expected_per_window * 2, (
                    f"Cache using {per_window:.1f}MB per window (expected ~{expected_per_window:.1f}MB)"
                )

    @pytest.mark.performance
    def test_streaming_memory_stability(self, minimal_model):
        """Test memory stability during streaming inference."""
        window_size = 15360
        stride_samples = 2560  # 10s stride

        # Simulate streaming buffer
        buffer_size = window_size + stride_samples * 10
        buffer = torch.zeros(1, 19, buffer_size)

        gc.collect()
        initial_ram, initial_gpu = self.get_memory_usage()

        memory_readings = []

        with torch.no_grad():
            for i in range(100):
                # Simulate new data arrival
                new_data = torch.randn(1, 19, stride_samples)
                buffer = torch.cat([buffer[:, :, stride_samples:], new_data], dim=2)

                # Extract window
                window = buffer[:, :, -window_size:]

                # Run inference
                output = minimal_model(window)

                # Track memory
                if i % 10 == 0:
                    current_ram, current_gpu = self.get_memory_usage()
                    memory_readings.append(current_ram - initial_ram)

                del output, new_data, window
                gc.collect()

        # Memory should be stable (not growing)
        memory_growth = max(memory_readings) - min(memory_readings)
        assert memory_growth < 20, f"Memory varied by {memory_growth:.1f}MB during streaming"

    @pytest.mark.performance
    def test_peak_memory_tracking(self, minimal_model):
        """Track peak memory usage during inference."""
        if not torch.cuda.is_available():
            pytest.skip("Peak memory tracking requires CUDA")

        batch_size = 8
        window = torch.randn(batch_size, 19, 15360).cuda()
        minimal_model = minimal_model.cuda()

        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            output = minimal_model(window)

        # Get peak memory
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Peak should be reasonable
        assert peak_memory_mb < 2000, f"Peak GPU memory {peak_memory_mb:.1f}MB exceeds 2GB limit"

        # Calculate efficiency
        output_size_mb = output.numel() * 4 / 1024 / 1024
        efficiency = output_size_mb / peak_memory_mb

        assert efficiency > 0.01, f"Memory efficiency {efficiency:.3f} is too low"

    @pytest.mark.performance
    def test_memory_profiling_detailed(self, minimal_model):
        """Detailed memory profiling with tracemalloc."""
        tracemalloc.start()

        # Take initial snapshot
        snapshot1 = tracemalloc.take_snapshot()

        # Run inference
        batch = torch.randn(4, 19, 15360)
        with torch.no_grad():
            output = minimal_model(batch)

        # Take second snapshot
        snapshot2 = tracemalloc.take_snapshot()

        # Calculate differences
        top_stats = snapshot2.compare_to(snapshot1, "lineno")

        # Find total allocated
        total_mb = sum(stat.size_diff for stat in top_stats) / 1024 / 1024

        # Check top allocations
        top_10_mb = sum(stat.size_diff for stat in top_stats[:10]) / 1024 / 1024

        assert total_mb < 500, f"Total allocation {total_mb:.1f}MB exceeds 500MB"
        assert top_10_mb / total_mb > 0.8, (
            "Memory allocation too fragmented (top 10 lines < 80% of total)"
        )

        tracemalloc.stop()

    @pytest.mark.performance
    @pytest.mark.slow
    def test_long_running_memory_stability(self, minimal_model):
        """Test memory stability over extended operation."""
        window = torch.randn(1, 19, 15360)
        memory_checkpoints = []

        with torch.no_grad():
            for i in range(1000):
                output = minimal_model(window)
                del output

                if i % 100 == 0:
                    gc.collect()
                    ram, gpu = self.get_memory_usage()
                    memory_checkpoints.append(ram)

        # Check for memory growth trend
        first_half = sum(memory_checkpoints[:5]) / 5
        second_half = sum(memory_checkpoints[5:]) / 5

        growth_rate = (second_half - first_half) / first_half

        assert abs(growth_rate) < 0.05, f"Memory grew by {growth_rate * 100:.1f}% over long run"
