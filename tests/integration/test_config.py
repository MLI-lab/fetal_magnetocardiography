"""
Integration tests for configuration dataclasses.

Tests the Phase 4 dataclass-based configuration system including
creation, conversion, and compatibility with legacy dict format.
"""

import pytest
import numpy as np
from fmcg.config import (
    PipelineConfig,
    DataConfig,
    PreprocessingConfig,
    SolverConfig,
    PostProcessingConfig,
    LearningRateConfig,
)


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_dataconfig_creation_with_required_fields(self):
        """Test creating DataConfig with only required fields."""
        config = DataConfig(
            ds_path="/test/path",
            patient="P001",
            series="S01",
            sig_group_names="R001",
            noise_group_names="R002",
        )
        assert config.ds_path == "/test/path"
        assert config.patient == "P001"
        assert config.sampfrom == 0  # default
        assert config.sampto is None  # default

    def test_dataconfig_with_optional_fields(self):
        """Test creating DataConfig with optional fields."""
        config = DataConfig(
            ds_path="/test/path",
            patient="P001",
            series="S01",
            sig_group_names="R001",
            noise_group_names="R002",
            noise_patient="P002",
            sampfrom=100,
            sampto=2000,
        )
        assert config.noise_patient == "P002"
        assert config.sampfrom == 100
        assert config.sampto == 2000


class TestPipelineConfigConversion:
    """Tests for PipelineConfig dict conversion."""

    def test_pipelineconfig_from_dict(self, sample_config_dict):
        """Test creating PipelineConfig from dictionary."""
        config = PipelineConfig.from_dict(sample_config_dict)

        assert isinstance(config, PipelineConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.preprocessing, PreprocessingConfig)
        assert isinstance(config.solver, SolverConfig)
        assert config.device == "cpu"

    def test_pipelineconfig_to_dict(self, sample_pipeline_config):
        """Test converting PipelineConfig to dictionary."""
        config_dict = sample_pipeline_config.to_dict()

        assert isinstance(config_dict, dict)
        assert "data" in config_dict
        assert "preprocessing" in config_dict
        assert "solver" in config_dict
        assert config_dict["device"] == "cpu"

    def test_roundtrip_conversion(self, sample_config_dict):
        """Test dict -> PipelineConfig -> dict preserves data."""
        config = PipelineConfig.from_dict(sample_config_dict)
        config_dict_back = config.to_dict()

        # Check key fields are preserved
        assert config_dict_back["data"]["patient"] == sample_config_dict["data"]["patient"]
        assert config_dict_back["device"] == sample_config_dict["device"]
        assert config_dict_back["preprocessing"]["whitening"] == sample_config_dict["preprocessing"]["whitening"]


class TestNestedDataclassConversion:
    """Tests for nested dataclass instantiation."""

    def test_solver_config_nested_conversion(self):
        """Test SolverConfig converts nested dicts to dataclasses."""
        config_dict = {
            "num_dipoles": 2,
            "optimizer": "lamb",
            "lr": {
                "niter": 500,
                "warmup": 50,
                "lr_groups": [1e-1, 1e-1, 1e-2],
            },
        }
        config = SolverConfig(**config_dict)

        assert isinstance(config.lr, LearningRateConfig)
        assert config.lr.niter == 500
        assert config.lr.warmup == 50

    def test_pipelineconfig_all_nested_conversions(self, sample_config_dict):
        """Test PipelineConfig converts all nested structures."""
        config = PipelineConfig.from_dict(sample_config_dict)

        # Check all nested configs are dataclasses
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.preprocessing, PreprocessingConfig)
        assert isinstance(config.solver, SolverConfig)
        assert isinstance(config.post_processing, PostProcessingConfig)

        # Check deeply nested
        assert isinstance(config.solver.lr, LearningRateConfig)
        assert isinstance(config.post_processing.averaging, dict) or hasattr(config.post_processing.averaging, '__dataclass_fields__')


class TestNumpyArrayHandling:
    """Tests for numpy array handling in configs."""

    def test_solver_config_with_numpy_arrays(self):
        """Test SolverConfig handles numpy arrays correctly."""
        r_bnds = np.array([
            [(-0.3, 0.3), (-0.3, -0.005), (-0.3, 0.3)],
            [(-0.3, 0.3), (-0.3, -0.005), (0.1, 0.6)]
        ])
        r_init = np.array([[[0, -0.02, 0.0], [0, -0.05, 0.2]]])

        config = SolverConfig(
            num_dipoles=2,
            r_bnds=r_bnds,
            r_init=r_init,
        )

        assert isinstance(config.r_bnds, np.ndarray)
        assert isinstance(config.r_init, np.ndarray)
        assert config.r_bnds.shape == (2, 3, 2)
        assert config.r_init.shape == (1, 2, 3)

    def test_solver_config_default_numpy_arrays(self):
        """Test SolverConfig creates default numpy arrays in __post_init__."""
        config = SolverConfig(num_dipoles=2)

        # Should have default arrays from __post_init__
        assert config.r_bnds is not None
        assert config.r_init is not None
        assert isinstance(config.r_bnds, np.ndarray)
        assert isinstance(config.r_init, np.ndarray)


class TestConfigDefaults:
    """Tests for default values in configs."""

    def test_preprocessing_config_defaults(self):
        """Test PreprocessingConfig has correct defaults."""
        config = PreprocessingConfig()

        assert config.whitening == "PCA"
        assert config.rescale_whitening is True
        assert config.bandpass_low == 1.0
        assert config.bandpass_high == 40.0
        assert config.downsample_factor == 2
        assert config.process_segments is True

    def test_pipeline_config_defaults(self, sample_data_config):
        """Test PipelineConfig has correct defaults."""
        config = PipelineConfig(data=sample_data_config)

        assert config.skip_existing is True
        assert config.catch_errors is True
        assert config.device == "cpu"
        assert config.save_plots is False
        assert config.output_dir == "output"


class TestBackwardCompatibility:
    """Tests for backward compatibility with dict-based configs."""

    def test_pipeline_accepts_dict_format(self, sample_config_dict):
        """Test that old dict configs still work."""
        # This simulates the Pipeline.run() conversion
        from fmcg.config import PipelineConfig

        config = PipelineConfig.from_dict(sample_config_dict)
        config_dict_back = config.to_dict()

        # Should be able to access nested values both ways
        assert config.data.patient == "P001"
        assert config_dict_back["data"]["patient"] == "P001"
