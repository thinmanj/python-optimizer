"""GPU Optimization Examples

Demonstrates GPU acceleration capabilities of Python Optimizer.
Shows automatic CPU/GPU dispatching, memory management, and performance comparisons.

Requirements:
    - CuPy (pip install cupy-cuda12x) for CUDA support
    - NVIDIA GPU with CUDA support

If GPU is not available, examples will automatically fall back to CPU.
"""

import time
import numpy as np
from python_optimizer import optimize, is_gpu_available, get_gpu_info


def example_1_basic_gpu():
    """Example 1: Basic GPU acceleration with @optimize(gpu=True)"""
    print("=" * 70)
    print("Example 1: Basic GPU Acceleration")
    print("=" * 70)

    # Check GPU availability
    if is_gpu_available():
        print("✓ GPU is available!")
        info = get_gpu_info()
        for device in info["devices"]:
            print(f"  - {device['name']}: {device['memory_gb']} GB")
    else:
        print("⚠ GPU not available - will use CPU fallback")

    print()

    # Define a function with GPU optimization
    @optimize(gpu=True, jit=False)
    def compute_with_gpu(data):
        """Compute with automatic GPU acceleration."""
        return data**2 + data * 3 - 5

    # Test with different sizes
    sizes = [1000, 10_000, 100_000, 1_000_000]

    for size in sizes:
        data = np.random.randn(size).astype(np.float32)

        start = time.perf_counter()
        result = compute_with_gpu(data)
        elapsed = time.perf_counter() - start

        print(f"Size {size:>10,}: {elapsed*1000:>8.3f} ms (result shape: {result.shape})")

    print()


def example_2_gpu_threshold():
    """Example 2: GPU threshold tuning"""
    print("=" * 70)
    print("Example 2: GPU Threshold Tuning")
    print("=" * 70)

    # Small threshold - uses GPU for smaller data
    @optimize(gpu=True, gpu_min_size=1000, jit=False)
    def compute_low_threshold(data):
        return np.sum(data**2)

    # Large threshold - uses GPU only for large data
    @optimize(gpu=True, gpu_min_size=100_000, jit=False)
    def compute_high_threshold(data):
        return np.sum(data**2)

    # Test both
    small_data = np.random.randn(5000).astype(np.float32)
    large_data = np.random.randn(500_000).astype(np.float32)

    print("\nSmall data (5K elements):")
    result1 = compute_low_threshold(small_data)
    print(f"  Low threshold (1K):  Uses GPU - Result: {result1:.2f}")

    result2 = compute_high_threshold(small_data)
    print(f"  High threshold (100K): Uses CPU - Result: {result2:.2f}")

    print("\nLarge data (500K elements):")
    result3 = compute_low_threshold(large_data)
    print(f"  Low threshold (1K):  Uses GPU - Result: {result3:.2f}")

    result4 = compute_high_threshold(large_data)
    print(f"  High threshold (100K): Uses GPU - Result: {result4:.2f}")

    print()


def example_3_matrix_operations():
    """Example 3: GPU-accelerated matrix operations"""
    print("=" * 70)
    print("Example 3: GPU Matrix Operations")
    print("=" * 70)

    @optimize(gpu=True, gpu_min_size=1000, jit=False)
    def matrix_computation(a, b):
        """Complex matrix computation."""
        result = np.matmul(a, b)
        result = result**2
        result = result + np.mean(result)
        return result

    # Test different matrix sizes
    sizes = [100, 500, 1000]

    for size in sizes:
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)

        start = time.perf_counter()
        result = matrix_computation(a, b)
        elapsed = time.perf_counter() - start

        print(f"Matrix {size}x{size}: {elapsed*1000:>8.3f} ms")

    print()


def example_4_combined_optimizations():
    """Example 4: Combining GPU with JIT and specialization"""
    print("=" * 70)
    print("Example 4: Combined Optimizations (GPU + JIT + Specialization)")
    print("=" * 70)

    @optimize(gpu=True, jit=True, specialize=True, gpu_min_size=10_000)
    def combined_optimization(data, multiplier):
        """Function with all optimizations enabled."""
        return data**2 * multiplier + data

    # Test with different types
    print("\nTesting with different input types:")

    # List input (CPU - small size)
    small_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    result1 = combined_optimization(small_list, 2.0)
    print(f"Small list: {result1}")

    # Numpy array (GPU - large size)
    large_array = np.random.randn(50_000).astype(np.float32)
    start = time.perf_counter()
    result2 = combined_optimization(large_array, 3.0)
    elapsed = time.perf_counter() - start
    print(f"Large array (50K): {elapsed*1000:.3f} ms, mean={np.mean(result2):.3f}")

    print()


def example_5_performance_comparison():
    """Example 5: CPU vs GPU performance comparison"""
    print("=" * 70)
    print("Example 5: CPU vs GPU Performance Comparison")
    print("=" * 70)

    # CPU version
    @optimize(gpu=False, jit=False)
    def compute_cpu(data):
        result = data**2
        result = result + data * 3
        result = np.sqrt(np.abs(result))
        return result

    # GPU version
    @optimize(gpu=True, jit=False, gpu_min_size=1000)
    def compute_gpu(data):
        result = data**2
        result = result + data * 3
        result = np.sqrt(np.abs(result))
        return result

    # Test sizes
    sizes = [10_000, 100_000, 1_000_000]

    print("\nPerformance comparison:")
    print(f"{'Size':<15} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
    print("-" * 50)

    for size in sizes:
        data = np.random.randn(size).astype(np.float32)

        # CPU timing
        start = time.perf_counter()
        result_cpu = compute_cpu(data.copy())
        cpu_time = time.perf_counter() - start

        # GPU timing
        start = time.perf_counter()
        result_gpu = compute_gpu(data.copy())
        gpu_time = time.perf_counter() - start

        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0

        print(
            f"{size:>13,}  {cpu_time*1000:>10.3f}  "
            f"{gpu_time*1000:>10.3f}  {speedup:>8.2f}x"
        )

    print()


def example_6_memory_management():
    """Example 6: GPU memory management"""
    print("=" * 70)
    print("Example 6: GPU Memory Management")
    print("=" * 70)

    if not is_gpu_available():
        print("GPU not available - skipping memory management example")
        return

    from python_optimizer import get_gpu_memory_info, clear_gpu_cache

    # Check initial memory
    mem_info = get_gpu_memory_info()
    if mem_info:
        print(f"\nInitial GPU Memory:")
        print(f"  Total: {mem_info.total_gb:.2f} GB")
        print(f"  Free:  {mem_info.free_gb:.2f} GB")
        print(f"  Used:  {mem_info.used_gb:.2f} GB")
        print(f"  Utilization: {mem_info.utilization_percent:.1f}%")

    # Allocate large array
    @optimize(gpu=True, gpu_min_size=1000, jit=False)
    def allocate_large(size):
        return np.ones(size) * 2.0

    print("\nAllocating 100M floats...")
    result = allocate_large(100_000_000)

    # Check memory after allocation
    mem_info = get_gpu_memory_info()
    if mem_info:
        print(f"\nAfter Allocation:")
        print(f"  Free:  {mem_info.free_gb:.2f} GB")
        print(f"  Used:  {mem_info.used_gb:.2f} GB")
        print(f"  Utilization: {mem_info.utilization_percent:.1f}%")

    # Clear cache
    print("\nClearing GPU cache...")
    clear_gpu_cache()

    mem_info = get_gpu_memory_info()
    if mem_info:
        print(f"\nAfter Cache Clear:")
        print(f"  Free:  {mem_info.free_gb:.2f} GB")
        print(f"  Used:  {mem_info.used_gb:.2f} GB")

    print()


def main():
    """Run all GPU examples."""
    print("\n" + "=" * 70)
    print("Python Optimizer - GPU Acceleration Examples")
    print("=" * 70)
    print()

    examples = [
        example_1_basic_gpu,
        example_2_gpu_threshold,
        example_3_matrix_operations,
        example_4_combined_optimizations,
        example_5_performance_comparison,
        example_6_memory_management,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            print()

    print("=" * 70)
    print("Examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
