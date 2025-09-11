"""
Variable Specialization Module

This module implements intelligent variable type specialization for Python functions.
It analyzes code patterns, detects variable types, and generates optimized specialized
versions of functions for different type combinations.

Key Components:
- TypeAnalyzer: Analyzes code and detects variable types and patterns
- SpecializationEngine: Creates specialized versions of functions
- RuntimeDispatcher: Selects optimal specialized version at runtime
- SpecializationCache: Manages cached specialized functions
"""

from .analyzer import TypeAnalyzer, VariableUsage, TypePattern
from .engine import SpecializationEngine, SpecializationConfig
from .dispatcher import RuntimeDispatcher, DispatchResult
from .cache import SpecializationCache, CacheEntry
from .generators import (
    NumericSpecializer,
    ArraySpecializer,
    StringSpecializer,
    ContainerSpecializer,
    SpecializationGenerator
)

__all__ = [
    # Core components
    "TypeAnalyzer",
    "SpecializationEngine", 
    "RuntimeDispatcher",
    "SpecializationCache",
    
    # Data structures
    "VariableUsage",
    "TypePattern",
    "SpecializationConfig",
    "DispatchResult", 
    "CacheEntry",
    
    # Specialized generators
    "NumericSpecializer",
    "ArraySpecializer", 
    "StringSpecializer",
    "ContainerSpecializer",
    "SpecializationGenerator",
]
