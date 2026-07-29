"""Tests for Advanced Profiler

Tests event recording, export formats, thread safety, and overhead.
"""

import json
import tempfile
import threading
import time
from pathlib import Path

import pytest

from python_optimizer.profiling.advanced_profiler import (
    AdvancedProfiler,
    EventType,
    ProfilingEvent,
    get_profiler,
    profile_function,
)


class TestProfilingEventBasics:
    """Test profiling event data structure."""

    def test_event_creation(self):
        """Test creating a profiling event."""
        event = ProfilingEvent(
            name="test_event",
            category="test",
            event_type=EventType.DURATION,
            timestamp=1.0,
            duration=0.5,
            tid=12345,
            pid=1,
            args={"key": "value"},
        )

        assert event.name == "test_event"
        assert event.category == "test"
        assert event.event_type == EventType.DURATION
        assert event.timestamp == 1.0
        assert event.duration == 0.5
        assert event.tid == 12345
        assert event.args["key"] == "value"

    def test_event_to_chrome_trace(self):
        """Test converting event to Chrome trace format."""
        event = ProfilingEvent(
            name="test_func",
            category="function",
            event_type=EventType.DURATION,
            timestamp=1.5,
            duration=0.25,
            tid=100,
            pid=1,
            args={"arg1": "val1"},
        )

        chrome_event = event.to_chrome_trace()

        assert chrome_event["name"] == "test_func"
        assert chrome_event["cat"] == "function"
        assert chrome_event["ph"] == "X"  # Duration event
        assert chrome_event["ts"] == 1_500_000  # microseconds
        assert chrome_event["dur"] == 250_000  # microseconds
        assert chrome_event["tid"] == 100
        assert chrome_event["args"]["arg1"] == "val1"

    def test_instant_event_to_chrome_trace(self):
        """Test instant event has no duration."""
        event = ProfilingEvent(
            name="marker",
            category="instant",
            event_type=EventType.INSTANT,
            timestamp=2.0,
            tid=200,
            pid=1,
        )

        chrome_event = event.to_chrome_trace()

        assert chrome_event["ph"] == "i"  # Instant event
        assert "dur" not in chrome_event  # No duration


class TestAdvancedProfilerBasics:
    """Basic functionality tests for advanced profiler."""

    def test_profiler_initialization(self):
        """Test profiler initializes correctly."""
        profiler = AdvancedProfiler(enabled=True)

        assert profiler.enabled is True
        assert len(profiler.events) == 0
        assert profiler.stats["total_events"] == 0

    def test_profiler_disabled(self):
        """Test disabled profiler doesn't record events."""
        profiler = AdvancedProfiler(enabled=False)

        profiler.record_event("test", "cat", EventType.INSTANT)

        assert len(profiler.events) == 0
        assert profiler.stats["total_events"] == 0

    def test_record_instant_event(self):
        """Test recording an instant event."""
        profiler = AdvancedProfiler()

        profiler.record_instant("marker1", category="test", extra="data")

        assert len(profiler.events) == 1
        event = profiler.events[0]
        assert event.name == "marker1"
        assert event.category == "test"
        assert event.event_type == EventType.INSTANT
        assert event.args["extra"] == "data"

    def test_record_function_call(self):
        """Test recording a function call with duration."""
        profiler = AdvancedProfiler()

        profiler.record_function_call("my_function", 0.123, category="compute")

        assert len(profiler.events) == 1
        event = profiler.events[0]
        assert event.name == "my_function"
        assert event.category == "compute"
        assert event.event_type == EventType.DURATION
        assert event.duration == 0.123

    def test_record_counter(self):
        """Test recording a counter value."""
        profiler = AdvancedProfiler()

        profiler.record_counter("memory_usage", 1024.5, category="metrics")

        assert len(profiler.events) == 1
        event = profiler.events[0]
        assert event.name == "memory_usage"
        assert event.category == "metrics"
        assert event.event_type == EventType.COUNTER
        assert event.args["value"] == 1024.5


class TestAdvancedProfilerSpans:
    """Test profiling spans with context managers."""

    def test_span_context_manager(self):
        """Test using span as context manager."""
        profiler = AdvancedProfiler()

        with profiler.start_span("operation", "cat"):
            time.sleep(0.01)  # Small delay

        assert len(profiler.events) == 1
        event = profiler.events[0]
        assert event.name == "operation"
        assert event.category == "cat"
        assert event.duration is not None
        assert event.duration >= 0.01  # At least the sleep time

    def test_nested_spans(self):
        """Test nested profiling spans."""
        profiler = AdvancedProfiler()

        with profiler.start_span("outer", "cat1"):
            time.sleep(0.01)
            with profiler.start_span("inner", "cat2"):
                time.sleep(0.01)

        assert len(profiler.events) == 2
        outer_event = profiler.events[0]
        inner_event = profiler.events[1]

        assert outer_event.name == "outer"
        assert inner_event.name == "inner"
        assert outer_event.duration > inner_event.duration  # Outer includes inner


class TestAdvancedProfilerExports:
    """Test various export formats."""

    def test_chrome_trace_export(self):
        """Test exporting to Chrome tracing format."""
        profiler = AdvancedProfiler()

        # Record some events
        profiler.record_function_call("func1", 0.1, category="compute")
        profiler.record_instant("marker", category="event")
        profiler.record_counter("metric", 42.0)

        # Export to file
        output_file = Path(tempfile.mktemp(suffix=".json"))
        profiler.export_chrome_trace(output_file)

        assert output_file.exists()

        # Verify format
        with open(output_file) as f:
            data = json.load(f)

        assert "traceEvents" in data
        assert "displayTimeUnit" in data
        assert data["displayTimeUnit"] == "ms"

        # Should have 4 events (3 + 1 metadata)
        assert len(data["traceEvents"]) == 4

        # Check metadata event
        metadata_event = data["traceEvents"][0]
        assert metadata_event["ph"] == "M"
        assert metadata_event["name"] == "process_name"

        # Cleanup
        output_file.unlink()

    def test_flamegraph_export(self):
        """Test exporting to flamegraph format."""
        profiler = AdvancedProfiler()

        # Record nested events
        profiler.record_function_call("main", 0.5)
        profiler.record_function_call("foo", 0.3)
        profiler.record_function_call("bar", 0.2)

        # Export
        output_file = Path(tempfile.mktemp(suffix=".txt"))
        profiler.export_flamegraph_data(output_file)

        assert output_file.exists()

        # Verify format
        with open(output_file) as f:
            lines = f.readlines()

        # Should have stack entries
        assert len(lines) > 0

        # Each line should be: "stack1;stack2;stack3 count"
        for line in lines:
            parts = line.strip().split()
            assert len(parts) == 2  # stack and count
            assert parts[1].isdigit()  # count is numeric

        # Cleanup
        output_file.unlink()

    def test_timeline_export(self):
        """Test exporting to timeline format."""
        profiler = AdvancedProfiler()

        # Record events
        profiler.record_function_call("func1", 0.1, category="cat1")
        profiler.record_function_call("func2", 0.2, category="cat2")

        # Export
        output_file = Path(tempfile.mktemp(suffix=".json"))
        profiler.export_timeline_data(output_file)

        assert output_file.exists()

        # Verify format
        with open(output_file) as f:
            data = json.load(f)

        assert "start_time" in data
        assert "duration" in data
        assert "threads" in data

        # Should have thread entries
        assert len(data["threads"]) > 0

        # Each thread should have events
        for tid, thread_data in data["threads"].items():
            assert "events" in thread_data
            assert "name" in thread_data

        # Cleanup
        output_file.unlink()


class TestAdvancedProfilerStats:
    """Test statistics tracking."""

    def test_stats_tracking(self):
        """Test profiler tracks statistics correctly."""
        profiler = AdvancedProfiler()

        # Initial stats
        stats = profiler.get_stats()
        assert stats["total_events"] == 0
        assert len(stats["categories"]) == 0

        # Record events in different categories
        profiler.record_instant("event1", category="cat1")
        profiler.record_instant("event2", category="cat2")
        profiler.record_instant("event3", category="cat1")

        stats = profiler.get_stats()
        assert stats["total_events"] == 3
        assert len(stats["categories"]) == 2
        assert "cat1" in stats["categories"]
        assert "cat2" in stats["categories"]

    def test_category_breakdown(self):
        """Test category breakdown in stats."""
        profiler = AdvancedProfiler()

        profiler.record_function_call("f1", 0.1, category="compute")
        profiler.record_function_call("f2", 0.2, category="compute")
        profiler.record_function_call("f3", 0.3, category="io")

        stats = profiler.get_stats()

        assert stats["category_counts"]["compute"] == 2
        assert stats["category_counts"]["io"] == 1
        assert stats["category_times"]["compute"] == 0.3
        assert stats["category_times"]["io"] == 0.3

    def test_clear_profiler(self):
        """Test clearing profiler data."""
        profiler = AdvancedProfiler()

        # Add events
        profiler.record_instant("event1")
        profiler.record_instant("event2")

        assert len(profiler.events) == 2

        # Clear
        profiler.clear()

        assert len(profiler.events) == 0
        assert profiler.stats["total_events"] == 0


class TestAdvancedProfilerConcurrency:
    """Test thread-safety of profiler."""

    def test_concurrent_event_recording(self):
        """Test multiple threads can record events safely."""
        profiler = AdvancedProfiler()
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):
                    profiler.record_instant(
                        f"event_{worker_id}_{i}",
                        category="test",
                        worker=worker_id,
                    )
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        # Start threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(profiler.events) == 50  # 5 workers * 10 events

    def test_concurrent_spans(self):
        """Test concurrent span recording."""
        profiler = AdvancedProfiler()
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):
                    with profiler.start_span(f"span_{worker_id}_{i}", "test"):
                        time.sleep(0.001)
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        # Start threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait
        for thread in threads:
            thread.join()

        # Verify
        assert len(errors) == 0
        assert len(profiler.events) == 30


class TestProfilerDecorator:
    """Test function decorator for profiling."""

    def test_profile_function_decorator(self):
        """Test @profile_function decorator."""
        profiler = get_profiler()
        profiler.clear()

        @profile_function(category="test")
        def test_func(x):
            return x * 2

        result = test_func(5)

        assert result == 10
        assert len(profiler.events) == 1
        event = profiler.events[0]
        assert event.name == "test_func"
        assert event.category == "test"


class TestGlobalProfiler:
    """Test global profiler instance."""

    def test_get_global_profiler(self):
        """Test get_profiler returns singleton."""
        profiler1 = get_profiler()
        profiler2 = get_profiler()

        assert profiler1 is profiler2

    def test_global_profiler_isolation(self):
        """Test global profiler is isolated between tests."""
        profiler = get_profiler()
        profiler.clear()

        profiler.record_instant("test")
        assert len(profiler.events) == 1


@pytest.mark.benchmark
class TestAdvancedProfilerPerformance:
    """Performance tests for profiler."""

    def test_event_recording_overhead(self):
        """Test event recording overhead is minimal."""
        profiler = AdvancedProfiler()

        # Measure overhead
        start = time.perf_counter()
        for i in range(1000):
            profiler.record_instant(f"event_{i}", category="test")
        elapsed = time.perf_counter() - start

        # Should be fast (< 10ms for 1000 events)
        assert elapsed < 0.01, f"Overhead too high: {elapsed}s"

    def test_span_overhead(self):
        """Test span context manager overhead."""
        profiler = AdvancedProfiler()

        def direct_function():
            return sum(range(100))

        def profiled_function():
            with profiler.start_span("test", "cat"):
                return sum(range(100))

        # Measure direct
        start = time.perf_counter()
        for _ in range(1000):
            direct_function()
        direct_time = time.perf_counter() - start

        # Measure profiled
        start = time.perf_counter()
        for _ in range(1000):
            profiled_function()
        profiled_time = time.perf_counter() - start

        # Overhead should be < 10x
        overhead = profiled_time / direct_time
        assert overhead < 10.0, f"Overhead too high: {overhead}x"

    def test_disabled_profiler_overhead(self):
        """Test disabled profiler has zero overhead."""
        profiler = AdvancedProfiler(enabled=False)

        start = time.perf_counter()
        for i in range(10000):
            profiler.record_instant(f"event_{i}")
        elapsed = time.perf_counter() - start

        # Should be near instant (< 1ms)
        assert elapsed < 0.001, f"Disabled profiler has overhead: {elapsed}s"


class TestAdvancedProfilerEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_profiler_export(self):
        """Test exporting with no events."""
        profiler = AdvancedProfiler()

        output_file = Path(tempfile.mktemp(suffix=".json"))
        profiler.export_chrome_trace(output_file)

        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        # Should have metadata event only
        assert len(data["traceEvents"]) == 1

        output_file.unlink()

    def test_profiler_print_summary(self, capsys):
        """Test print_summary output."""
        profiler = AdvancedProfiler()

        profiler.record_function_call("f1", 0.1, category="cat1")
        profiler.record_function_call("f2", 0.2, category="cat2")

        profiler.print_summary()

        captured = capsys.readouterr()
        assert "Advanced Profiler Summary" in captured.out
        assert "Total Events: 2" in captured.out
        assert "cat1" in captured.out
        assert "cat2" in captured.out
