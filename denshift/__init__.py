"""Denshift: black-box, source-free test-time adaptation for 3D point clouds.

The package is split so that the estimator never depends on a deep-learning
framework.  ``geometry``, ``streaming``, ``calibrate`` and ``changepoint`` are
pure NumPy/SciPy; ``backends`` provides the optional PyTorch glue.
"""

from .adapter import (
    Adapter,
    DSC,
    DSCNoBucket,
    DSCNoDrift,
    EntropyBias,
    GlobalBias,
    SourceOnly,
    build_adapter,
)
from .blackbox import BlackBoxModel, CallableBox, ContractViolation
from .calibrate import DensityBiasModel
from .changepoint import DescriptorShiftDetector, ShockDetector
from .geometry import density_descriptor, stratified_subsample, subsample
from .metrics import (
    accuracy,
    corruption_error,
    expected_calibration_error,
    mean_corruption_error,
    per_class_accuracy,
    resilience_rate,
)
from .streaming import RunningStandardizer, StreamingQuantile

__version__ = "0.1.0"

__all__ = [
    "Adapter",
    "BlackBoxModel",
    "CallableBox",
    "DescriptorShiftDetector",
    "ContractViolation",
    "DSC",
    "DSCNoBucket",
    "DSCNoDrift",
    "DensityBiasModel",
    "EntropyBias",
    "GlobalBias",
    "RunningStandardizer",
    "ShockDetector",
    "SourceOnly",
    "StreamingQuantile",
    "__version__",
    "accuracy",
    "build_adapter",
    "corruption_error",
    "density_descriptor",
    "expected_calibration_error",
    "mean_corruption_error",
    "per_class_accuracy",
    "resilience_rate",
    "stratified_subsample",
    "subsample",
]
