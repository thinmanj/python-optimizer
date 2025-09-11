# Python Optimizer 🚀

A high-performance Python optimization toolkit that provides JIT compilation, variable specialization, and runtime optimizations to accelerate Python code execution without changing language syntax.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Numba](https://img.shields.io/badge/numba-JIT%20compilation-orange.svg)](http://numba.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Goal

Accelerate Python program execution by **10-100x** through:
- **JIT compilation** of hot code paths using Numba
- **Variable specialization** for type-specific optimizations  
- **Runtime profiling** and adaptive optimization
- **Zero syntax changes** - works with existing Python code

## ⚡ Performance Results

Real-world performance improvements achieved:

| Function Type | Original Time | Optimized Time | Speedup |
|---------------|---------------|----------------|---------|
| Numerical Computation | 2.06ms | 0.04ms | **51x** |
| Financial Metrics | 100ms | 2ms | **50x** |
| Trading Simulation | 500ms | 5ms | **100x** |
| Genetic Algorithm | 30s | 0.14s | **214x** |

**Throughput:** Up to **36,456 evaluations/second**

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/thinmanj/python-optimizer.git
cd python-optimizer

# Install with pip
pip install -e .

# Or install from PyPI (coming soon)
pip install python-optimizer
```

## 🚀 Quick Start

### 1. Basic JIT Optimization

```python
from python_optimizer import optimize

@optimize(jit=True)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# First call compiles, subsequent calls are blazing fast
result = fibonacci(35)  # ~100x faster after compilation
```

### 2. Financial Computing Example

```python
import numpy as np
from python_optimizer.jit import calculate_sharpe_ratio_jit

# JIT-compiled financial metrics
returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
sharpe = calculate_sharpe_ratio_jit(returns)   # ~50x faster
```

### 3. Trading Strategy Optimization

```python
from python_optimizer.jit import JITBacktestFitnessEvaluator
from python_optimizer.genetic import Individual

# Ultra-fast backtesting with JIT compilation
evaluator = JITBacktestFitnessEvaluator(initial_cash=10000)
individual = Individual(genes={'ma_short': 10, 'ma_long': 30})

# Evaluate strategy performance
metrics = evaluator.evaluate(individual, market_data)
# Achieves 36,000+ evaluations per second
```

## 📦 Features

### JIT Compilation Engine
- **Numba-powered** JIT compilation for numerical code
- **Automatic type inference** and optimization
- **GIL-free execution** for parallel processing
- **Caching system** for compiled functions

### Variable Specialization
- **Type-specific optimizations** for common patterns
- **Adaptive specialization** based on runtime behavior
- **Memory-efficient** specialized code paths

### Performance Profiling
- **Runtime profiling** with minimal overhead
- **Hot path detection** and prioritization
- **Performance analytics** and reporting

### Financial Computing
- **JIT-optimized** financial metrics (Sharpe ratio, drawdown, etc.)
- **Fast backtesting** engine for trading strategies
- **Genetic algorithm** optimization for parameter tuning

## 📖 Documentation

### Core Decorator

The `@optimize` decorator is the main entry point:

```python
from python_optimizer import optimize

@optimize(
    jit=True,              # Enable JIT compilation
    specialize=False,      # Enable variable specialization
    profile=True,          # Enable performance profiling
    aggressiveness=2       # Optimization level (0-3)
)
def your_function(x, y):
    # Your code here
    return x * y + compute_heavy_operation()
```

### JIT Functions

Pre-built JIT-optimized functions:

```python
from python_optimizer.jit import (
    calculate_returns_jit,
    calculate_sharpe_ratio_jit,
    calculate_max_drawdown_jit,
    simulate_strategy_jit
)
```

### Genetic Algorithm Optimization

```python
from python_optimizer.genetic import GeneticOptimizer, ParameterRange

# Define optimization parameters
param_ranges = [
    ParameterRange('learning_rate', 0.001, 0.1, 'float'),
    ParameterRange('hidden_layers', 1, 5, 'int'),
]

# Run optimization
optimizer = GeneticOptimizer(param_ranges, population_size=100)
best_params = optimizer.optimize(fitness_function, generations=50)
```

## 🧪 Examples

Check out the `examples/` directory for:

- **Financial modeling** with JIT optimization
- **Machine learning** parameter optimization
- **Numerical computing** acceleration
- **Trading strategy** backtesting

## 📊 Benchmarks

Run benchmarks to see performance on your system:

```bash
python benchmarks/jit_performance_test.py
python benchmarks/genetic_algorithm_benchmark.py
python benchmarks/financial_metrics_benchmark.py
```

## 🔧 Configuration

### Environment Variables

```bash
export PYTHON_OPTIMIZER_JIT_CACHE=1     # Enable JIT cache
export PYTHON_OPTIMIZER_PROFILE=1       # Enable profiling
export PYTHON_OPTIMIZER_PARALLEL=1      # Enable parallel execution
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

[specialization]
max_variants = 5
threshold = 100
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/thinmanj/python-optimizer.git
cd python-optimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
black python_optimizer/
isort python_optimizer/
flake8 python_optimizer/
```

## 📈 Roadmap

- [x] **JIT Compilation Engine** - Numba-based optimization
- [x] **Financial Computing Module** - Trading strategy optimization
- [x] **Genetic Algorithm** - Parameter optimization
- [ ] **GPU Acceleration** - CUDA support for parallel execution
- [ ] **ML Model Optimization** - PyTorch/TensorFlow integration  
- [ ] **Distributed Computing** - Multi-node optimization
- [ ] **Web Interface** - Browser-based optimization dashboard

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Numba team** for excellent JIT compilation framework
- **NumPy community** for foundational numerical computing
- **Trading algorithm researchers** for inspiration and validation

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/thinmanj/python-optimizer/issues)
- **Discussions:** [GitHub Discussions](https://github.com/thinmanj/python-optimizer/discussions)
- **Email:** thinmanj@users.noreply.github.com

---

⭐ **Star this repository** if Python Optimizer helps accelerate your code!
