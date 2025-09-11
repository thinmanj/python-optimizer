"""
Main optimization decorator for Python functions.

The @optimize decorator provides an easy-to-use interface for applying
various optimization techniques to Python functions.
"""

import functools
import logging
from typing import Any, Callable, Optional, Union
from .engine import OptimizationEngine
from ..specialization.engine import get_global_engine as get_specialization_engine

logger = logging.getLogger(__name__)

# Global optimization engine instance
_optimization_engine = OptimizationEngine()


def optimize(
    func: Optional[Callable] = None,
    *,
    jit: bool = True,
    specialize: bool = False,
    profile: bool = False,
    aggressiveness: int = 1,
    cache: bool = True,
    parallel: bool = False,
    nogil: bool = False,
    fastmath: bool = True,
) -> Union[Callable, Callable[[Callable], Callable]]:
    """
    Optimize a Python function using JIT compilation and other techniques.
    
    Args:
        func: The function to optimize (can be None when used with parameters)
        jit: Enable JIT compilation using Numba
        specialize: Enable variable specialization
        profile: Enable performance profiling  
        aggressiveness: Optimization level (0-3, higher = more aggressive)
        cache: Enable JIT compilation caching
        parallel: Enable parallel execution where possible
        nogil: Release GIL during execution (requires compatible code)
        fastmath: Enable fast math optimizations (may reduce accuracy)
        
    Returns:
        Optimized function or decorator
        
    Examples:
        @optimize
        def simple_function(x, y):
            return x * y + x ** 2
        
        @optimize(jit=True, aggressiveness=3)
        def complex_computation(data):
            result = 0
            for i in range(len(data)):
                result += data[i] ** 2
            return result
    """
    
    def decorator(fn: Callable) -> Callable:
        # Create optimization configuration
        config = {
            'jit': jit,
            'specialize': specialize,
            'profile': profile,
            'aggressiveness': aggressiveness,
            'cache': cache,
            'parallel': parallel,
            'nogil': nogil,
            'fastmath': fastmath,
        }
        
        # Start with original function
        optimized_fn = fn
        
        # Apply JIT optimization first if enabled
        if jit or profile:  # Profile requires JIT engine
            optimized_fn = _optimization_engine.optimize_function(optimized_fn, config)
        
        # Apply variable specialization if enabled
        if specialize:
            spec_engine = get_specialization_engine()
            optimized_fn = spec_engine.optimize_function(optimized_fn)
        
        # Preserve function metadata
        optimized_fn = functools.update_wrapper(optimized_fn, fn)
        
        # Add optimization metadata
        optimized_fn._optimization_config = config
        optimized_fn._original_function = fn
        optimized_fn._has_jit = jit or profile
        optimized_fn._has_specialization = specialize
        
        logger.debug(f"Optimized function {fn.__name__} with config: {config}")
        
        return optimized_fn
    
    # Handle both @optimize and @optimize(...) usage patterns
    if func is None:
        # Called with parameters: @optimize(...)
        return decorator
    else:
        # Called without parameters: @optimize
        return decorator(func)


def get_optimization_stats() -> dict:
    """Get optimization statistics from the global engine."""
    return _optimization_engine.get_stats()


def clear_optimization_cache() -> None:
    """Clear the optimization cache."""
    _optimization_engine.clear_cache()


def set_optimization_level(level: int) -> None:
    """Set the global optimization level (0-3)."""
    _optimization_engine.set_optimization_level(level)


def enable_profiling() -> None:
    """Enable global profiling."""
    _optimization_engine.enable_profiling()


def disable_profiling() -> None:
    """Disable global profiling."""
    _optimization_engine.disable_profiling()


# Specialization functions
def get_specialization_stats(func_name: Optional[str] = None) -> dict:
    """Get variable specialization statistics."""
    from ..specialization.engine import get_specialization_stats
    return get_specialization_stats(func_name)


def clear_specialization_cache(func_name: Optional[str] = None) -> None:
    """Clear specialization cache."""
    from ..specialization.engine import clear_specialization_cache
    clear_specialization_cache(func_name)


def configure_specialization(**kwargs) -> None:
    """Configure specialization parameters."""
    from ..specialization.engine import configure_specialization, SpecializationConfig
    config = SpecializationConfig(**kwargs)
    configure_specialization(config)
