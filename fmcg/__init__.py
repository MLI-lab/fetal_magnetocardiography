"""
fMCG: Fetal Magnetocardiography Signal Processing Package

Main entry point:
    Pipeline: Main processing pipeline for fMCG data

For lower-level functionality, import from submodules:
    - fmcg.analysis: Heartbeat detection and analysis
    - fmcg.signal: Filtering, whitening, artifact removal
    - fmcg.fitting: Inverse solver and field models
    - fmcg.utils: Utilities and plotting
    - fmcg.ics: Independent Component Subtraction
"""

from .pipeline import Pipeline

__all__ = ["Pipeline"]
__version__ = "0.1.0"