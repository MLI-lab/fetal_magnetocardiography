"""
Shared pytest fixtures for fMCG tests.

This module provides common fixtures and test utilities used across
integration and unit tests.
"""

import pytest
import numpy as np
from fmcg.config import (
    PipelineConfig,
    DataConfig,
    PreprocessingConfig,
    SolverConfig,
    PostProcessingConfig,
)


@pytest.fixture
def sample_data_config():
    """Create a sample DataConfig for testing."""
    return DataConfig(
        ds_path="/test/data/path",
        patient="P001",
        series="S01",
        sig_group_names="R001",
        noise_group_names="R002",
        sampfrom=0,
        sampto=1000,
    )


@pytest.fixture
def sample_preprocessing_config():
    """Create a sample PreprocessingConfig for testing."""
    return PreprocessingConfig(
        whitening="PCA",
        rescale_whitening=True,
        bandpass_low=1.0,
        bandpass_high=40.0,
        bandpass_order=4,
        signal_threshold=2.0,
        noise_threshold=0.1,
        downsample_factor=2,
        process_segments=False,  # Disable for faster tests
        min_segment_length=10.0,
    )


@pytest.fixture
def sample_solver_config():
    """Create a sample SolverConfig for testing."""
    return SolverConfig(
        parallel_processing=False,  # Disable for simpler testing
        max_workers=1,
        num_dipoles=2,
        optimizer="lamb",
        m_scaling=[1e0, 1e0],
        r_basis="sigmoid_time_constant",
    )


@pytest.fixture
def sample_post_processing_config():
    """Create a sample PostProcessingConfig for testing."""
    return PostProcessingConfig(
        method="subsequent",
        decomposition="ica",
        nlms=False,
        mu=0.001,
        consider_mean=False,
        wavelet_denoise=False,
        ica_components=3,
        n_trials=1,  # Reduce for faster tests
    )


@pytest.fixture
def sample_pipeline_config(
    sample_data_config,
    sample_preprocessing_config,
    sample_solver_config,
    sample_post_processing_config,
):
    """Create a complete PipelineConfig for testing."""
    return PipelineConfig(
        data=sample_data_config,
        preprocessing=sample_preprocessing_config,
        solver=sample_solver_config,
        post_processing=sample_post_processing_config,
        skip_existing=False,
        catch_errors=False,  # Let tests see real errors
        device="cpu",
        save_reports=False,
        save_plots=False,
        save_data=False,
        output_dir="/tmp/test_output",
    )


@pytest.fixture
def sample_config_dict():
    """Create a sample configuration dictionary (legacy format)."""
    return {
        "data": {
            "ds_path": "/test/data/path",
            "patient": "P001",
            "series": "S01",
            "sig_group_names": "R001",
            "noise_group_names": "R002",
            "sampfrom": 0,
            "sampto": 1000,
        },
        "preprocessing": {
            "whitening": "PCA",
            "rescale_whitening": True,
            "bandpass_low": 1.0,
            "bandpass_high": 40.0,
        },
        "solver": {
            "num_dipoles": 2,
            "optimizer": "lamb",
        },
        "post_processing": {
            "method": "subsequent",
            "decomposition": "ica",
        },
        "device": "cpu",
        "save_plots": False,
    }


@pytest.fixture
def synthetic_fmcg_signal():
    """
    Generate a small synthetic fMCG signal for testing.

    Returns:
        dict: Contains 'data', 'time', 'fs', 'true_dipoles' keys
    """
    fs = 500  # Hz (after downsampling from 1000)
    duration = 5  # seconds
    n_samples = int(fs * duration)
    n_sensors = 16
    n_axes = 3

    # Create simple synthetic dipole moments (sinusoidal)
    time = np.linspace(0, duration, n_samples)

    # Fetal heartbeat: ~140 bpm = 2.33 Hz
    fetal_freq = 2.33
    fetal_dipole = 1e-6 * np.sin(2 * np.pi * fetal_freq * time)

    # Maternal heartbeat: ~80 bpm = 1.33 Hz
    maternal_freq = 1.33
    maternal_dipole = 5e-6 * np.sin(2 * np.pi * maternal_freq * time)

    # Create synthetic sensor data (simplified forward model)
    # In reality this would use proper field calculations
    data = np.random.randn(n_samples, n_sensors * n_axes) * 1e-8  # Small noise

    # Add dipole contributions (simplified)
    for i in range(n_sensors * n_axes):
        weight_fetal = np.random.randn() * 0.1
        weight_maternal = np.random.randn() * 0.2
        data[:, i] += weight_fetal * fetal_dipole + weight_maternal * maternal_dipole

    return {
        "data": data,
        "time": time,
        "fs": fs,
        "true_dipoles": {
            "fetal": {"moment": fetal_dipole, "freq": fetal_freq, "hr_bpm": fetal_freq * 60},
            "maternal": {"moment": maternal_dipole, "freq": maternal_freq, "hr_bpm": maternal_freq * 60},
        },
    }
