"""Tests for ExperimentInfra caching behavior.

Verifies that the cached_stage decorator persists and loads results
when outputs are serializable, and that repeated calls hit the cache.
"""

from pathlib import Path

from src.experiment.infra import ExperimentInfra


def test_cached_stage_hits_cache(tmp_path: Path):
    infra = ExperimentInfra(cache_dir=tmp_path, experiment_name="cache_test")

    calls = {"count": 0}

    @infra.cached_stage("dummy_stage")
    def compute():
        calls["count"] += 1
        return {"value": 123}

    # First call should compute
    res1 = compute(_cache_context={"param": 1})
    assert res1 == {"value": 123}
    assert calls["count"] == 1

    # Second call with same context should hit cache (no new compute)
    res2 = compute(_cache_context={"param": 1})
    assert res2 == {"value": 123}
    assert calls["count"] == 1

    # Different context should compute again
    res3 = compute(_cache_context={"param": 2})
    assert res3 == {"value": 123}
    assert calls["count"] == 2
