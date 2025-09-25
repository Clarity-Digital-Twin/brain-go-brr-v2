"""Performance test utilities with configurable tolerances."""

from src.brain_brr.utils.env import env


class PerfThresholds:
    """Centralized performance thresholds with environment-based tolerance."""

    def __init__(self):
        self.tolerance = 1.0 if env.perf_strict_mode() else env.perf_tolerance_factor()

    # Latency thresholds (milliseconds)
    def single_window_latency_ms(self, is_cpu: bool = False) -> float:
        """Max latency for single window inference."""
        base = 100 if is_cpu else 50
        return base * self.tolerance

    def batch_latency_per_sample_ms(self, is_cpu: bool = False) -> float:
        """Max latency per sample in batch inference."""
        base = 100 if is_cpu else 25
        return base * self.tolerance

    def streaming_p95_latency(self, stride_s: float) -> float:
        """P95 latency for streaming (fraction of stride)."""
        base = 0.5  # 50% of stride
        return min(base * self.tolerance, 0.9)  # Never exceed 90% of stride

    def hourly_throughput_time_s(self, is_cpu: bool = False) -> float:
        """Max time to process 1 hour of data."""
        base = 390 if is_cpu else 180
        return base * self.tolerance

    def daily_processing_hours(self) -> float:
        """Max hours to process 24h of data."""
        base = 1.5
        return base * self.tolerance

    def min_throughput_realtime(self, is_cpu: bool = False) -> float:
        """Minimum throughput as multiple of realtime."""
        base = 6.0 if is_cpu else 15.0
        return base / self.tolerance  # Lower is better for min threshold

    def gpu_speedup_factor(self) -> float:
        """Minimum GPU speedup over CPU."""
        base = 5.0
        return base / self.tolerance

    def compilation_speedup(self) -> float:
        """Minimum speedup from torch.compile."""
        base = 1.05
        return base / self.tolerance

    # Memory thresholds (MB)
    def ram_per_sample_mb(self) -> float:
        """Max RAM per sample."""
        base = 500
        return base * self.tolerance

    def gpu_per_sample_mb(self) -> float:
        """Max GPU memory per sample."""
        base = 200
        return base * self.tolerance

    def model_footprint_mb(self) -> float:
        """Max model memory footprint."""
        base = 300  # ~125MB checkpoint * 2.4x overhead
        return base * self.tolerance

    def memory_growth_mb(self) -> float:
        """Max memory growth during inference."""
        base = 50
        return base * self.tolerance

    def gpu_memory_growth_mb(self) -> float:
        """Max GPU memory growth during inference."""
        base = 10
        return base * self.tolerance

    def forward_memory_overhead(self) -> float:
        """Max memory overhead ratio for forward pass."""
        base = 1.5
        return base * self.tolerance

    def backward_memory_overhead(self) -> float:
        """Max memory overhead ratio for backward pass."""
        base = 3.0
        return base * self.tolerance

    def streaming_memory_variation_pct(self) -> float:
        """Max memory variation during streaming (%)."""
        base = 20
        return base * self.tolerance

    def gpu_peak_reserved_mb(self) -> float:
        """Max GPU reserved memory."""
        base = 2000
        return base * self.tolerance

    def memory_efficiency_ratio(self) -> float:
        """Min memory efficiency (active/reserved)."""
        base = 0.30
        return base / self.tolerance

    def long_running_growth_rate(self) -> float:
        """Max memory growth rate over long runs."""
        base = 0.05  # 5%
        return base * self.tolerance

    def latency_degradation_pct(self) -> float:
        """Max latency degradation over time (%)."""
        base = 20
        return base * self.tolerance

    def concurrent_speedup_efficiency(self, n_threads: int) -> float:
        """Min efficiency for concurrent inference."""
        base = 0.5  # 50% of ideal speedup
        return base / self.tolerance


# Global instance
thresholds = PerfThresholds()