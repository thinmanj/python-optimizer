"""ML Model Optimization Module

Provides optimization capabilities for machine learning models with focus on PyTorch.

Features:
- Automatic model compilation and optimization
- Training loop acceleration with mixed precision
- Inference optimization with dynamic batching
- GPU acceleration integration
- Memory optimization

Supported Frameworks:
- PyTorch (primary support)
- TensorFlow (planned)

Usage:
    from python_optimizer.ml import optimize_model, optimize_training
    
    # Optimize PyTorch model
    optimized_model = optimize_model(model, device='cuda')
    
    # Optimize training loop
    @optimize_training(mixed_precision=True, gpu=True)
    def train_step(model, data, target):
        # Training code
        pass
"""

# Try to import PyTorch
try:
    import torch

    PYTORCH_AVAILABLE = True
    PYTORCH_VERSION = torch.__version__
except ImportError:
    PYTORCH_AVAILABLE = False
    PYTORCH_VERSION = None
    torch = None

# Core ML optimization functions
from python_optimizer.ml.pytorch_optimizer import (
    optimize_model,
    optimize_training,
    optimize_inference,
    PyTorchModelOptimizer,
    TrainingOptimizer,
    InferenceOptimizer,
)

__all__ = [
    # PyTorch optimization
    "optimize_model",
    "optimize_training",
    "optimize_inference",
    "PyTorchModelOptimizer",
    "TrainingOptimizer",
    "InferenceOptimizer",
    # Framework availability
    "PYTORCH_AVAILABLE",
    "PYTORCH_VERSION",
]


def check_framework_availability():
    """Check which ML frameworks are available.
    
    Returns:
        dict: Framework availability status
    """
    return {
        "pytorch": {
            "available": PYTORCH_AVAILABLE,
            "version": PYTORCH_VERSION,
        },
        "tensorflow": {
            "available": False,
            "version": None,
        },
    }
