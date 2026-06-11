"""
fMCG: Fetal Magnetocardiography Signal Processing Package

Main entry point:
    Pipeline: Main processing pipeline for fMCG data

Configuration classes:
    - PipelineConfig: Top-level configuration
    - DataConfig: Data loading configuration
    - PreprocessingConfig: Preprocessing configuration
    - SolverConfig: Solver configuration
    - PostProcessingConfig: Post-processing configuration

For lower-level functionality, import from submodules:
    - fmcg.analysis: Heartbeat detection and analysis
    - fmcg.signal: Filtering, whitening, artifact removal
    - fmcg.fitting: Inverse solver and field models
    - fmcg.utils: Utilities and plotting
    - fmcg.ics: Independent Component Subtraction
"""

from .pipeline import Pipeline
from .pipeline.preprocessing import load_and_preprocess, preprocess, PreprocessedData
from .config import (
    PipelineConfig,
    DataConfig,
    PreprocessingConfig,
    SolverConfig,
    PostProcessingConfig,
    LearningRateConfig,
    LossConfig,
    InitializerConfig,
    AveragingConfig,
)

# Expose commonly used submodules at package level for notebook/legacy imports
from . import utils as utils
from .utils import data as data
from .utils.plotting import plot_utils as plot_utils
from .fitting.inverse_solver import InverseSolver
from .signal.filtering import filter_sensor_dict
from .signal.artifact_removal import remove_outlier_dict
from . import fitting
from . import signal
from . import analysis
from . import ica
from .utils.utils import get_config_for_date, reduced2full
from .signal.whitening import unwhiten

__all__ = [
    "Pipeline",
    "load_and_preprocess",
    "preprocess",
    "PreprocessedData",
    "PipelineConfig",
    "DataConfig",
    "PreprocessingConfig",
    "SolverConfig",
    "PostProcessingConfig",
    "LearningRateConfig",
    "LossConfig",
    "InitializerConfig",
    "AveragingConfig",
    "utils",
    "data",
    "plot_utils",
    "InverseSolver",
    "filter_sensor_dict",
    "remove_outlier_dict",
]
__version__ = "0.1.0"