"""Tests for Distributed JIT Cache

Tests cache functionality, persistence, synchronization, and thread safety.
"""

import os
import pickle
import tempfile
import threading
import time
from pathlib import Path

import pytest

from python_optimizer.distributed.jit_cache import (
    DistributedJITCache,
    clear_distributed_jit_cache,
    get_distributed_jit_cache,
)


# Module-level functions for pickling (Windows/multiprocessing compatibility)
def _test_multiply_by_two(x):
    return x * 2


def _test_multiply_by_three(x):
    return x * 3


def _test_add_one(x):
    return x + 1


class TestDistributedJITCacheBasics:
    """Basic functionality tests for distributed JIT cache."""

    def test_cache_initialization(self):
        """Test cache initializes with correct defaults."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedJITCache(cache_dir=cache_dir)

        assert cache.cache_dir == cache_dir
        assert cache.cache_dir.exists()
        assert len(cache._memory_cache) == 0
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0

    def test_cache_key_generation(self):
        """Test cache key generation is deterministic."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedJITCache(cache_dir=cache_dir)

        def test_func():
            return 42

        options1 = {"opt": "value"}
        options2 = {"opt": "value"}
        options3 = {"opt": "different"}

        key1 = cache._generate_key(test_func, options1)
        key2 = cache._generate_key(test_func, options2)
        key3 = cache._generate_key(test_func, options3)

        # Same function + options = same key
        assert key1 == key2
        # Different options = different key
        assert key1 != key3

    def test_cache_put_and_get(self):
        """Test basic put and get operations."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedJITCache(cache_dir=cache_dir)

        def test_func():
            return 42

        # Use module-level function (picklable)
        compiled_func = _test_multiply_by_two

        options = {"fastmath": True}
        key = cache._generate_key(test_func, options)

        # Put in cache
        cache.put(key, compiled_func)

        # Get from cache
        result = cache.get(key)
        assert result is not None
        assert result(5) == 10

        # Check stats
        assert cache.stats["hits"] == 1
        assert cache.stats["compilations"] == 1

    def test_cache_miss(self):
        """Test cache miss behavior."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedJITCache(cache_dir=cache_dir)

        result = cache.get("nonexistent_key")
        assert result is None
        assert cache.stats["misses"] == 1
        assert cache.stats["hits"] == 0


class TestDistributedJITCachePersistence:
    """Test cache persistence across instances."""

    def test_disk_persistence(self):
        """Test cache persists to disk and can be reloaded."""
        cache_dir = Path(tempfile.mkdtemp())

        # First cache instance
        cache1 = DistributedJITCache(cache_dir=cache_dir)

        def test_func():
            return 42

        compiled_func = lambda x: x * 3
        options = {"parallel": True}
        key = cache1._generate_key(test_func, options)

        cache1.put(key, compiled_func)

        # Create second cache instance (simulates restart)
        cache2 = DistributedJITCache(cache_dir=cache_dir)

        # Should load from disk
        result = cache2.get(key)
        assert result is not None
        assert result(4) == 12

        # Should be marked as network transfer
        assert cache2.stats["network_transfers"] == 1

    def test_clear_cache(self):
        """Test clearing cache removes all entries."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedJITCache(cache_dir=cache_dir)

        # Add some entries
        for i in range(5):
            key = f"key_{i}"
            cache.put(key, lambda x: x * i)

        assert len(cache._memory_cache) == 5
        assert len(list(cache.cache_dir.glob("*.pkl"))) == 5

        # Clear cache
        cache.clear()

        assert len(cache._memory_cache) == 0
        assert len(list(cache.cache_dir.glob("*.pkl"))) == 0
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0


class TestDistributedJITCacheGetOrCompile:
    """Test get_or_compile functionality."""

    def test_get_or_compile_cache_hit(self):
        """Test get_or_compile returns cached version."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedJITCache(cache_dir=cache_dir)

        def source_func(x):
            return x + 1

        compile_count = [0]

        def mock_compiler(func):
            compile_count[0] += 1
            return lambda x: func(x)

        options = {"nopython": True}

        # First call - should compile
        result1 = cache.get_or_compile(source_func, mock_compiler, options)
        assert compile_count[0] == 1
        assert result1(5) == 6

        # Second call - should use cache
        result2 = cache.get_or_compile(source_func, mock_compiler, options)
        assert compile_count[0] == 1  # No additional compilation
        assert result2(5) == 6

        # Stats should show 1 miss, 1 hit
        assert cache.stats["misses"] == 1
        assert cache.stats["hits"] == 1
        assert cache.stats["compilations"] == 1

    def test_get_or_compile_different_options(self):
        """Test different options trigger new compilation."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedJITCache(cache_dir=cache_dir)

        def source_func(x):
            return x * 2

        compile_count = [0]

        def mock_compiler(func):
            compile_count[0] += 1
            return lambda x: func(x)

        # First call with options1
        result1 = cache.get_or_compile(source_func, mock_compiler, {"opt": "a"})
        assert compile_count[0] == 1

        # Second call with different options
        result2 = cache.get_or_compile(source_func, mock_compiler, {"opt": "b"})
        assert compile_count[0] == 2  # New compilation

        # Both should work
        assert result1(3) == 6
        assert result2(3) == 6


class TestDistributedJITCacheConcurrency:
    """Test thread-safety of cache operations."""

    def test_concurrent_reads(self):
        """Test multiple threads can read simultaneously."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedJITCache(cache_dir=cache_dir)

        # Pre-populate cache
        for i in range(10):
            key = f"key_{i}"
            cache.put(key, lambda x, i=i: x * i)

        results = []
        errors = []

        def read_worker(worker_id):
            try:
                for i in range(10):
                    key = f"key_{i}"
                    result = cache.get(key)
                    if result is None:
                        errors.append(f"Worker {worker_id}: key {key} not found")
                    else:
                        results.append((worker_id, i, result(2)))
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        # Start multiple reader threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=read_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 50  # 5 workers * 10 reads

    def test_concurrent_writes(self):
        """Test multiple threads can write without corruption."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedJITCache(cache_dir=cache_dir)

        errors = []

        def write_worker(worker_id):
            try:
                for i in range(10):
                    key = f"worker_{worker_id}_key_{i}"
                    cache.put(key, lambda x, v=worker_id * 10 + i: x + v)
                    time.sleep(0.001)  # Small delay to increase contention
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        # Start multiple writer threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=write_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(cache._memory_cache) == 50  # 5 workers * 10 writes

        # Verify each entry is correct
        for worker_id in range(5):
            for i in range(10):
                key = f"worker_{worker_id}_key_{i}"
                result = cache.get(key)
                assert result is not None
                expected = worker_id * 10 + i
                assert result(0) == expected


class TestDistributedJITCacheStats:
    """Test cache statistics tracking."""

    def test_stats_tracking(self):
        """Test statistics are tracked correctly."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedJITCache(cache_dir=cache_dir)

        # Initial stats
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["compilations"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["cache_size"] == 0

        # Add entries
        for i in range(5):
            key = f"key_{i}"
            cache.put(key, lambda x: x)

        stats = cache.get_stats()
        assert stats["compilations"] == 5
        assert stats["cache_size"] == 5

        # Trigger hits
        for i in range(5):
            key = f"key_{i}"
            cache.get(key)

        stats = cache.get_stats()
        assert stats["hits"] == 5
        assert stats["hit_rate"] == 1.0  # 5 hits, 0 misses = 100%

        # Trigger misses
        for i in range(3):
            cache.get(f"nonexistent_{i}")

        stats = cache.get_stats()
        assert stats["misses"] == 3
        assert stats["hit_rate"] == 5 / 8  # 5 hits, 3 misses


class TestGlobalCache:
    """Test global cache instance."""

    def test_get_global_cache(self):
        """Test get_distributed_jit_cache returns singleton."""
        cache1 = get_distributed_jit_cache()
        cache2 = get_distributed_jit_cache()

        assert cache1 is cache2

    def test_clear_global_cache(self):
        """Test clearing global cache."""
        cache = get_distributed_jit_cache()

        # Add entry
        cache.put("test_key", lambda x: x)
        assert len(cache._memory_cache) == 1

        # Clear
        clear_distributed_jit_cache()
        assert len(cache._memory_cache) == 0


@pytest.mark.benchmark
class TestDistributedJITCachePerformance:
    """Performance benchmarks for cache operations."""

    def test_cache_put_performance(self, benchmark):
        """Benchmark cache put operation."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedJITCache(cache_dir=cache_dir)

        def put_operation():
            key = "test_key"
            cache.put(key, lambda x: x * 2)

        benchmark(put_operation)

    def test_cache_get_performance(self, benchmark):
        """Benchmark cache get operation."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedJITCache(cache_dir=cache_dir)

        # Pre-populate
        cache.put("test_key", lambda x: x * 2)

        def get_operation():
            return cache.get("test_key")

        benchmark(get_operation)

    def test_cache_overhead(self):
        """Test cache overhead is minimal."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedJITCache(cache_dir=cache_dir)

        # Measure direct function call
        direct_func = lambda x: x * 2
        start = time.perf_counter()
        for _ in range(1000):
            direct_func(5)
        direct_time = time.perf_counter() - start

        # Measure cached function call
        cache.put("test_key", direct_func)
        start = time.perf_counter()
        for _ in range(1000):
            cached_func = cache.get("test_key")
            cached_func(5)
        cached_time = time.perf_counter() - start

        # Overhead should be minimal (allow 5x for safety)
        overhead = cached_time / direct_time
        assert overhead < 5.0, f"Cache overhead too high: {overhead}x"
