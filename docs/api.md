# API Documentation

## Core Components

### @optimize Decorator

The main entry point for Python optimization.

```python
from python_optimizer import optimize

@optimize(jit=True, specialize=False, profile=True, aggressiveness=2)
def my_function(x, y):
    return x * y + expensive_computation()
```

#### Parameters

- **jit** (bool, default=True): Enable JIT compilation using Numba
- **specialize** (bool, default=False): Enable variable type specialization
- **profile** (bool, default=True): Enable performance profiling
- **aggressiveness** (int, 0-3, default=2): Optimization level
  - 0: Conservative optimizations only
  - 1: Standard optimizations
  - 2: Aggressive optimizations (recommended)
  - 3: Experimental optimizations (may break compatibility)

#### Returns

Decorated function with applied optimizations.

#### Example Usage

```python
# Basic JIT optimization
@optimize(jit=True)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# High-performance numerical computation
@optimize(jit=True, fastmath=True, nogil=True)
def monte_carlo_pi(n_samples):
    count = 0
    for i in range(n_samples):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1:
            count += 1
    return 4.0 * count / n_samples
```

---

## JIT Functions

Pre-built JIT-optimized functions for common use cases.

### Financial Computing

#### calculate_returns_jit(prices)

Calculate returns from price series.

```python
from python_optimizer.jit import calculate_returns_jit
import numpy as np

prices = np.array([100, 102, 98, 101, 105])
returns = calculate_returns_jit(prices)
# Returns: [0.02, -0.0392, 0.0306, 0.0396]
```

#### calculate_sharpe_ratio_jit(returns, risk_free_rate=0.0)

Calculate Sharpe ratio from returns.

```python
sharpe = calculate_sharpe_ratio_jit(returns, risk_free_rate=0.02)
```

#### calculate_max_drawdown_jit(equity_curve)

Calculate maximum drawdown from equity curve.

```python
equity = np.array([100, 110, 105, 120, 115, 130])
max_dd = calculate_max_drawdown_jit(equity)
```

### Signal Generation

#### generate_ma_signals_jit(prices, short_window, long_window)

Generate moving average crossover signals.

```python
signals = generate_ma_signals_jit(prices, short_window=10, long_window=30)
```

#### generate_rsi_signals_jit(prices, window, oversold, overbought)

Generate RSI-based trading signals.

```python
rsi_signals = generate_rsi_signals_jit(prices, window=14, oversold=30, overbought=70)
```

---

## Genetic Algorithm Optimization

### GeneticOptimizer Class

Optimize function parameters using genetic algorithms.

```python
from python_optimizer.genetic import GeneticOptimizer, ParameterRange

# Define parameter search space
param_ranges = [
    ParameterRange('learning_rate', 0.001, 0.1, 'float'),
    ParameterRange('hidden_units', 10, 100, 'int'),
    ParameterRange('dropout', 0.1, 0.5, 'float'),
]

# Create optimizer
optimizer = GeneticOptimizer(
    param_ranges=param_ranges,
    population_size=50,
    mutation_rate=0.1,
    crossover_rate=0.8,
    elitism_ratio=0.2
)

# Define fitness function
def fitness_function(individual):
    params = individual.genes
    # Train model with params and return validation score
    return model_score

# Run optimization
best_individual = optimizer.optimize(
    fitness_function=fitness_function,
    generations=100,
    early_stopping=True,
    patience=10
)

print(f"Best parameters: {best_individual.genes}")
print(f"Best fitness: {best_individual.fitness}")
```

### Individual Class

Represents a single solution in the genetic algorithm.

```python
from python_optimizer.genetic import Individual

individual = Individual(genes={'lr': 0.01, 'units': 64})
print(individual.fitness)  # Fitness score (set by evaluator)
```

### ParameterRange Class

Defines search space for optimization parameters.

```python
# Float parameter
float_param = ParameterRange('learning_rate', min_val=0.001, max_val=0.1, param_type='float')

# Integer parameter  
int_param = ParameterRange('epochs', min_val=10, max_val=100, param_type='int')

# Choice parameter
choice_param = ParameterRange('optimizer', choices=['adam', 'sgd', 'rmsprop'], param_type='choice')
```

---

## Profiling and Performance

### PerformanceProfiler Class

Profile and analyze optimization performance.

```python
from python_optimizer.profiling import PerformanceProfiler

profiler = PerformanceProfiler()

@profiler.profile
def my_function():
    # Function to profile
    pass

# Get performance statistics
stats = profiler.get_stats()
print(f"Total calls: {stats.call_count}")
print(f"Average time: {stats.avg_time:.4f}s")
```

### Performance Statistics

```python
from python_optimizer import get_performance_stats, clear_performance_stats

# Get current performance data
stats = get_performance_stats()
for func_name, func_stats in stats.items():
    print(f"{func_name}: {func_stats.total_time:.4f}s")

# Clear statistics
clear_performance_stats()
```

---

## Configuration

### Environment Variables

```bash
# JIT Settings
export PYTHON_OPTIMIZER_JIT_CACHE=1
export PYTHON_OPTIMIZER_JIT_CACHE_DIR=~/.python_optimizer/cache

# Profiling
export PYTHON_OPTIMIZER_PROFILE=1
export PYTHON_OPTIMIZER_PROFILE_DIR=./profiles

# Parallel Processing
export PYTHON_OPTIMIZER_PARALLEL=1
export PYTHON_OPTIMIZER_MAX_WORKERS=4
```

### Configuration File

Create `python_optimizer.toml`:

```toml
[jit]
cache_dir = "~/.python_optimizer/cache"
compile_timeout = 30

[profiling]
enabled = true
output_dir = "./profiles"
threshold = 0.001

[specialization]
max_variants = 5
threshold = 100

[parallel]
max_workers = 4
backend = "threading"  # or "multiprocessing"
```

---

## Error Handling

### OptimizationError

Raised when optimization fails.

```python
from python_optimizer.exceptions import OptimizationError

try:
    @optimize(jit=True)
    def problematic_function():
        # Code that can't be JIT compiled
        pass
except OptimizationError as e:
    print(f"Optimization failed: {e}")
    # Fallback to non-optimized version
```

### JITCompilationError

Raised when JIT compilation fails.

```python
from python_optimizer.exceptions import JITCompilationError

try:
    result = jit_function(data)
except JITCompilationError as e:
    print(f"JIT compilation failed: {e}")
    # Use fallback implementation
```

---

## Best Practices

### When to Use JIT

✅ **Good for JIT:**
- Numerical computations with loops
- Mathematical operations on arrays
- Recursive algorithms
- Financial calculations

❌ **Avoid JIT for:**
- String manipulation
- File I/O operations
- Object-oriented code with inheritance
- Functions with dynamic typing

### Performance Tips

1. **Warm-up JIT functions:**
```python
@optimize(jit=True)
def my_function(data):
    return compute_result(data)

# Warm up with small data
my_function(small_data)
# Now use with real data
result = my_function(large_data)
```

2. **Use appropriate data types:**
```python
# Prefer NumPy arrays over lists
data = np.array([1, 2, 3, 4, 5], dtype=np.float64)
```

3. **Enable fastmath for numerical code:**
```python
@optimize(jit=True, fastmath=True)
def numerical_computation(x):
    return np.sqrt(x * x + 1)
```

4. **Profile before optimizing:**
```python
from python_optimizer.profiling import PerformanceProfiler

profiler = PerformanceProfiler()
# Profile your code to identify bottlenecks
```
