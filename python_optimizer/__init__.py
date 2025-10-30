"""
Python Optimizer - High-performance Python optimization toolkit

Provides JIT compilation, variable specialization, and runtime optimizations
to accelerate Python code execution without changing language syntax.
"""

__version__ = "0.1.0"
__author__ = "Julio Ona"
__email__ = "thinmanj@gmail.com"

# Import main optimization decorator
from .core.decorator import (
    clear_specialization_cache,
    configure_specialization,
    get_specialization_stats,
    optimize,
)

# Import genetic algorithm components
from .genetic import FitnessEvaluator, GeneticOptimizer, Individual, ParameterRange

# Import JIT functions
from .jit import (
    JITBacktestFitnessEvaluator,
    calculate_max_drawdown_jit,
    calculate_profit_factor_jit,
    calculate_returns_jit,
    calculate_sharpe_ratio_jit,
    calculate_win_rate_jit,
    generate_ma_signals_jit,
    generate_rsi_signals_jit,
    simulate_strategy_jit,
)

# Import profiling components
from .profiling import (
    PerformanceProfiler,
    ProfilerConfig,
    clear_performance_stats,
    get_performance_stats,
)

# Import GPU components (optional - graceful degradation if not available)
try:
    from .gpu import (
        GPUDevice,
        GPUDispatcher,
        GPUKernelLibrary,
        GPUMemoryManager,
        clear_gpu_cache,
        get_gpu_device,
        get_gpu_info,
        get_gpu_memory_info,
        is_gpu_available,
        set_gpu_device,
    )

    _GPU_AVAILABLE = True
except ImportError:
    _GPU_AVAILABLE = False

    # Provide stub functions for graceful degradation
    def is_gpu_available():
        return False

    def get_gpu_info():
        return {"available": False, "message": "GPU support not installed"}


__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    "__email__",
    # Core optimization
    "optimize",
    "get_specialization_stats",
    "clear_specialization_cache",
    "configure_specialization",
    # JIT functions
    "calculate_returns_jit",
    "calculate_sharpe_ratio_jit",
    "calculate_max_drawdown_jit",
    "calculate_profit_factor_jit",
    "calculate_win_rate_jit",
    "simulate_strategy_jit",
    "generate_ma_signals_jit",
    "generate_rsi_signals_jit",
    "JITBacktestFitnessEvaluator",
    # Genetic algorithm
    "Individual",
    "ParameterRange",
    "GeneticOptimizer",
    "FitnessEvaluator",
    # Profiling
    "PerformanceProfiler",
    "ProfilerConfig",
    "get_performance_stats",
    "clear_performance_stats",
    # GPU (if available)
    "is_gpu_available",
    "get_gpu_info",
]

# Add full GPU API to __all__ if available
if _GPU_AVAILABLE:
    __all__.extend(
        [
            "GPUDevice",
            "GPUDispatcher",
            "GPUKernelLibrary",
            "GPUMemoryManager",
            "clear_gpu_cache",
            "get_gpu_device",
            "get_gpu_memory_info",
            "set_gpu_device",
        ]
    )
