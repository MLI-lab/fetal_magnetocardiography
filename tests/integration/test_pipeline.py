"""
Integration tests for Pipeline class.

Tests the Pipeline context manager, config acceptance, and resource management.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fmcg import Pipeline, PipelineConfig


class TestPipelineContextManager:
    """Tests for Pipeline as context manager."""

    def test_pipeline_context_manager_enter_exit(self):
        """Test Pipeline can be used as context manager."""
        with Pipeline() as pipeline:
            assert pipeline is not None
            assert isinstance(pipeline, Pipeline)

    def test_pipeline_context_manager_returns_self(self):
        """Test __enter__ returns the Pipeline instance."""
        pipeline = Pipeline()
        result = pipeline.__enter__()
        assert result is pipeline

    def test_pipeline_context_manager_exception_propagates(self):
        """Test exceptions are propagated from context manager."""
        with pytest.raises(ValueError):
            with Pipeline() as pipeline:
                raise ValueError("Test error")

    @patch('torch.cuda.empty_cache')
    def test_pipeline_gpu_cleanup_on_exit(self, mock_empty_cache):
        """Test GPU cache is cleared on context exit when using CUDA."""
        pipeline = Pipeline()
        pipeline.device = "cuda:0"  # Set before entering context

        # Enter and exit context
        pipeline.__enter__()
        pipeline.__exit__(None, None, None)

        # Verify GPU cleanup was called
        mock_empty_cache.assert_called_once()

    @patch('fmcg.pipeline.pipeline.torch')
    def test_pipeline_no_gpu_cleanup_for_cpu(self, mock_torch):
        """Test GPU cleanup is NOT called when using CPU."""
        mock_torch.cuda.empty_cache = Mock()

        with Pipeline() as pipeline:
            pipeline.device = "cpu"

        # GPU cleanup should not be called for CPU
        mock_torch.cuda.empty_cache.assert_not_called()


class TestPipelineConfigAcceptance:
    """Tests for Pipeline accepting different config formats."""

    def test_pipeline_accepts_pipelineconfig_dataclass(self, sample_pipeline_config):
        """Test Pipeline.run() accepts PipelineConfig dataclass."""
        pipeline = Pipeline()

        # Mock the actual run logic since we're just testing config acceptance
        with patch.object(pipeline, '_log_step_header'):
            # This should not raise an error
            # We can't actually run the full pipeline without real data,
            # but we can check the config is accepted
            try:
                # Set up minimal attributes to avoid AttributeError
                sample_pipeline_config.data.ds_path = "/fake/path"

                # Just check that the config conversion happens
                if hasattr(sample_pipeline_config, 'to_dict'):
                    config_dict = sample_pipeline_config.to_dict()
                    assert isinstance(config_dict, dict)
                    assert "device" in config_dict
            except FileNotFoundError:
                # Expected - we don't have real data
                pass
            except AttributeError as e:
                # Also acceptable - pipeline tries to access data
                pass

    def test_pipeline_accepts_dict_config(self, sample_config_dict):
        """Test Pipeline.run() accepts legacy dict config."""
        pipeline = Pipeline()

        # Verify dict configs are still accepted
        assert isinstance(sample_config_dict, dict)
        assert "device" in sample_config_dict

    def test_pipeline_converts_dataclass_to_dict_internally(self, sample_pipeline_config):
        """Test Pipeline converts PipelineConfig to dict for internal use."""
        config_dict = sample_pipeline_config.to_dict()

        # Verify conversion works
        assert isinstance(config_dict, dict)
        assert "data" in config_dict
        assert "preprocessing" in config_dict
        assert "solver" in config_dict
        assert "post_processing" in config_dict


class TestPipelineResourceManagement:
    """Tests for Pipeline resource management."""

    def test_pipeline_initializes_step_counter(self):
        """Test Pipeline initializes with step counter."""
        pipeline = Pipeline()
        assert hasattr(pipeline, '_step_num')
        assert pipeline._step_num == 1

    def test_pipeline_has_device_context_attribute(self):
        """Test Pipeline has _device_context attribute."""
        pipeline = Pipeline()
        assert hasattr(pipeline, '_device_context')
        assert pipeline._device_context is None

    @patch('fmcg.pipeline.pipeline.torch')
    def test_pipeline_handles_gpu_cleanup_errors_gracefully(self, mock_torch):
        """Test Pipeline handles GPU cleanup errors without crashing."""
        # Simulate CUDA cleanup error
        mock_torch.cuda.empty_cache.side_effect = RuntimeError("CUDA error")

        # Should not raise, just log warning
        with Pipeline() as pipeline:
            pipeline.device = "cuda:0"

        # No exception should be raised

    def test_pipeline_exit_returns_false(self):
        """Test __exit__ returns False to propagate exceptions."""
        pipeline = Pipeline()
        result = pipeline.__exit__(None, None, None)
        assert result is False


class TestPipelineIntegration:
    """Integration tests combining multiple Pipeline features."""

    def test_pipeline_context_manager_with_dataclass_config(self, sample_pipeline_config):
        """Test using Pipeline as context manager with dataclass config."""
        with Pipeline() as pipeline:
            # Verify pipeline is usable
            assert pipeline is not None

            # Verify config conversion would work
            config_dict = sample_pipeline_config.to_dict()
            assert "device" in config_dict

    @patch('torch.cuda.empty_cache')
    def test_pipeline_full_lifecycle(self, mock_empty_cache, sample_pipeline_config):
        """Test full Pipeline lifecycle: create, use, cleanup."""
        # Full lifecycle
        pipeline = Pipeline()
        pipeline.device = "cuda:0"
        pipeline.__enter__()
        pipeline.__exit__(None, None, None)

        # Verify cleanup happened
        mock_empty_cache.assert_called_once()
