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
    optimize,
    get_specialization_stats,
    clear_specialization_cache,
    configure_specialization
)

# Import JIT functions
from .jit import (
    calculate_returns_jit,
    calculate_sharpe_ratio_jit,
    calculate_max_drawdown_jit,
    calculate_profit_factor_jit,
    calculate_win_rate_jit,
    simulate_strategy_jit,
    generate_ma_signals_jit,
    generate_rsi_signals_jit,
    JITBacktestFitnessEvaluator
)

# Import genetic algorithm components
from .genetic import (
    Individual,
    ParameterRange,
    GeneticOptimizer,
    FitnessEvaluator
)

# Import profiling components
from .profiling import (
    PerformanceProfiler,
    ProfilerConfig,
    get_performance_stats,
    clear_performance_stats
)

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
    "clear_performance_stats"
]
