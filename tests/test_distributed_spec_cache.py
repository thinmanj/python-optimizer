"""Tests for Distributed Specialization Cache

Tests cache functionality, synchronization, metadata tracking, and thread safety.
"""

import tempfile
import threading
import time
from pathlib import Path

import pytest

from python_optimizer.distributed.spec_cache import (
    DistributedSpecializationCache,
    clear_distributed_spec_cache,
    get_distributed_spec_cache,
)


class TestDistributedSpecCacheBasics:
    """Basic functionality tests for distributed specialization cache."""

    def test_cache_initialization(self):
        """Test cache initializes with correct defaults."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedSpecializationCache(cache_dir=cache_dir)

        assert cache.cache_dir == cache_dir
        assert cache.cache_dir.exists()
        assert len(cache._memory_cache) == 0
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0

    def test_cache_key_generation(self):
        """Test cache key generation is deterministic."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedSpecializationCache(cache_dir=cache_dir)

        key1 = cache._generate_key("my_func", "int_int")
        key2 = cache._generate_key("my_func", "int_int")
        key3 = cache._generate_key("my_func", "float_float")
        key4 = cache._generate_key("other_func", "int_int")

        # Same func + signature = same key
        assert key1 == key2
        # Different signature = different key
        assert key1 != key3
        # Different function = different key
        assert key1 != key4

    def test_cache_put_and_get(self):
        """Test basic put and get operations."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedSpecializationCache(cache_dir=cache_dir)

        # Mock specialized version
        specialized_func = lambda x, y: x + y

        metadata = {
            "perf_gain": 10.5,
            "usage_count": 1,
        }

        # Put in cache
        cache.put("test_func", "int_int", specialized_func, metadata)

        # Get from cache
        result = cache.get("test_func", "int_int")
        assert result is not None
        spec_func, meta = result
        assert spec_func(3, 4) == 7
        assert meta["perf_gain"] == 10.5
        assert meta["usage_count"] == 2  # Incremented on get

    def test_cache_miss(self):
        """Test cache miss behavior."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedSpecializationCache(cache_dir=cache_dir)

        result = cache.get("nonexistent_func", "some_signature")
        assert result is None
        assert cache.stats["misses"] == 1


class TestDistributedSpecCachePersistence:
    """Test cache persistence across instances."""

    def test_disk_persistence(self):
        """Test cache persists to disk and can be reloaded."""
        cache_dir = Path(tempfile.mkdtemp())

        # First cache instance
        cache1 = DistributedSpecializationCache(cache_dir=cache_dir)

        specialized_func = lambda x: x ** 2
        metadata = {
            "func_name": "square",
            "type_signature": "int",
            "perf_gain": 5.0,
        }

        cache1.put("square", "int", specialized_func, metadata)

        # Create second cache instance (simulates restart)
        cache2 = DistributedSpecializationCache(cache_dir=cache_dir)

        # Should load from disk
        result = cache2.get("square", "int")
        assert result is not None
        spec_func, meta = result
        assert spec_func(5) == 25
        assert meta["perf_gain"] == 5.0

    def test_metadata_update(self):
        """Test metadata can be updated."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedSpecializationCache(cache_dir=cache_dir)

        # Put initial entry
        cache.put("test_func", "sig1", lambda x: x, {"usage_count": 1})

        # Update metadata
        cache.update_metadata("test_func", "sig1", {"usage_count": 10})

        # Verify update
        result = cache.get("test_func", "sig1")
        _, meta = result
        assert meta["usage_count"] == 11  # 10 + 1 from get

    def test_clear_cache_all(self):
        """Test clearing entire cache."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedSpecializationCache(cache_dir=cache_dir)

        # Add entries
        for i in range(5):
            cache.put(f"func{i}", "sig", lambda x: x, {})

        assert len(cache._memory_cache) == 5

        # Clear all
        cache.clear()

        assert len(cache._memory_cache) == 0
        assert len(list(cache.cache_dir.glob("*.pkl"))) == 0

    def test_clear_cache_specific_function(self):
        """Test clearing cache for specific function."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedSpecializationCache(cache_dir=cache_dir)

        # Add entries for multiple functions
        cache.put("func1", "sig1", lambda x: x, {})
        cache.put("func1", "sig2", lambda x: x, {})
        cache.put("func2", "sig1", lambda x: x, {})

        assert len(cache._memory_cache) == 3

        # Clear only func1
        cache.clear("func1")

        assert len(cache._memory_cache) == 1
        assert cache.get("func2", "sig1") is not None
        assert cache.get("func1", "sig1") is None


class TestDistributedSpecCacheSync:
    """Test cache synchronization."""

    def test_sync_loads_new_entries(self):
        """Test sync loads entries added by other instances."""
        cache_dir = Path(tempfile.mkdtemp())

        # Cache 1 adds entry
        cache1 = DistributedSpecializationCache(cache_dir=cache_dir)
        metadata1 = {
            "func_name": "test_func",
            "type_signature": "sig1",
            "perf_gain": 3.0,
        }
        cache1.put("test_func", "sig1", lambda x: x * 2, metadata1)

        # Cache 2 doesn't have it yet
        cache2 = DistributedSpecializationCache(cache_dir=cache_dir)
        assert len(cache2._memory_cache) == 0

        # Sync pulls from disk
        cache2.sync()

        # Now cache2 should have it
        result = cache2.get("test_func", "sig1")
        assert result is not None
        spec_func, meta = result
        assert spec_func(5) == 10

    def test_sync_stats(self):
        """Test sync operation increments stats."""
        cache_dir = Path(tempfile.mkdtemp())

        cache1 = DistributedSpecializationCache(cache_dir=cache_dir)
        cache1.put("func1", "sig1", lambda x: x, {"func_name": "func1", "type_signature": "sig1"})

        cache2 = DistributedSpecializationCache(cache_dir=cache_dir)
        initial_syncs = cache2.stats["sync_operations"]

        cache2.sync()

        assert cache2.stats["sync_operations"] == initial_syncs + 1


class TestDistributedSpecCacheConcurrency:
    """Test thread-safety of cache operations."""

    def test_concurrent_reads(self):
        """Test multiple threads can read simultaneously."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedSpecializationCache(cache_dir=cache_dir)

        # Pre-populate cache
        for i in range(10):
            cache.put(f"func{i}", "sig", lambda x, i=i: x * i, {})

        results = []
        errors = []

        def read_worker(worker_id):
            try:
                for i in range(10):
                    result = cache.get(f"func{i}", "sig")
                    if result is None:
                        errors.append(f"Worker {worker_id}: func{i} not found")
                    else:
                        spec_func, _ = result
                        results.append((worker_id, i, spec_func(2)))
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
        cache = DistributedSpecializationCache(cache_dir=cache_dir)

        errors = []

        def write_worker(worker_id):
            try:
                for i in range(10):
                    func_name = f"worker{worker_id}_func{i}"
                    cache.put(
                        func_name,
                        "sig",
                        lambda x, v=worker_id * 10 + i: x + v,
                        {"worker": worker_id, "index": i},
                    )
                    time.sleep(0.001)
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

        # Verify each entry
        for worker_id in range(5):
            for i in range(10):
                func_name = f"worker{worker_id}_func{i}"
                result = cache.get(func_name, "sig")
                assert result is not None
                spec_func, meta = result
                expected = worker_id * 10 + i
                assert spec_func(0) == expected


class TestDistributedSpecCacheStats:
    """Test cache statistics tracking."""

    def test_stats_tracking(self):
        """Test statistics are tracked correctly."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedSpecializationCache(cache_dir=cache_dir)

        # Initial stats
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["specializations_created"] == 0
        assert stats["hit_rate"] == 0.0

        # Add entries
        for i in range(5):
            cache.put(f"func{i}", "sig", lambda x: x, {})

        stats = cache.get_stats()
        assert stats["specializations_created"] == 5
        assert stats["cache_size"] == 5

        # Trigger hits
        for i in range(5):
            cache.get(f"func{i}", "sig")

        stats = cache.get_stats()
        assert stats["hits"] == 5
        assert stats["hit_rate"] == 1.0

        # Trigger misses
        for i in range(3):
            cache.get(f"missing{i}", "sig")

        stats = cache.get_stats()
        assert stats["misses"] == 3
        assert stats["hit_rate"] == 5 / 8

    def test_metadata_entries_count(self):
        """Test metadata entries are counted."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedSpecializationCache(cache_dir=cache_dir)

        for i in range(5):
            cache.put(f"func{i}", "sig", lambda x: x, {"data": i})

        stats = cache.get_stats()
        assert stats["total_metadata_entries"] == 5


class TestGlobalCache:
    """Test global cache instance."""

    def test_get_global_cache(self):
        """Test get_distributed_spec_cache returns singleton."""
        cache1 = get_distributed_spec_cache()
        cache2 = get_distributed_spec_cache()

        assert cache1 is cache2

    def test_clear_global_cache(self):
        """Test clearing global cache."""
        cache = get_distributed_spec_cache()

        # Add entry
        cache.put("test_func", "sig", lambda x: x, {})
        assert len(cache._memory_cache) == 1

        # Clear
        clear_distributed_spec_cache()
        assert len(cache._memory_cache) == 0

    def test_clear_global_cache_specific_function(self):
        """Test clearing specific function from global cache."""
        cache = get_distributed_spec_cache()

        # Add entries
        cache.put("func1", "sig", lambda x: x, {})
        cache.put("func2", "sig", lambda x: x, {})

        # Clear only func1
        clear_distributed_spec_cache("func1")

        assert cache.get("func1", "sig") is None
        assert cache.get("func2", "sig") is not None


@pytest.mark.benchmark
class TestDistributedSpecCachePerformance:
    """Performance benchmarks for cache operations."""

    def test_cache_put_performance(self, benchmark):
        """Benchmark cache put operation."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedSpecializationCache(cache_dir=cache_dir)

        def put_operation():
            cache.put("test_func", "sig", lambda x: x * 2, {"perf": 1.0})

        benchmark(put_operation)

    def test_cache_get_performance(self, benchmark):
        """Benchmark cache get operation."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedSpecializationCache(cache_dir=cache_dir)

        # Pre-populate
        cache.put("test_func", "sig", lambda x: x * 2, {})

        def get_operation():
            return cache.get("test_func", "sig")

        benchmark(get_operation)

    def test_cache_overhead(self):
        """Test cache overhead is minimal."""
        cache_dir = Path(tempfile.mkdtemp())
        cache = DistributedSpecializationCache(cache_dir=cache_dir)

        # Measure direct function call
        direct_func = lambda x: x * 2
        start = time.perf_counter()
        for _ in range(1000):
            direct_func(5)
        direct_time = time.perf_counter() - start

        # Measure cached function call
        cache.put("test_func", "sig", direct_func, {})
        start = time.perf_counter()
        for _ in range(1000):
            result = cache.get("test_func", "sig")
            if result:
                spec_func, _ = result
                spec_func(5)
        cached_time = time.perf_counter() - start

        # Overhead should be minimal (allow 5x for safety)
        overhead = cached_time / direct_time
        assert overhead < 5.0, f"Cache overhead too high: {overhead}x"
