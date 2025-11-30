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

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "DataConfig",
    "PreprocessingConfig",
    "SolverConfig",
    "PostProcessingConfig",
    "LearningRateConfig",
    "LossConfig",
    "InitializerConfig",
    "AveragingConfig",
]
__version__ = "0.1.0"