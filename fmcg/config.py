"""
Configuration dataclasses for the fMCG pipeline.

This module provides structured configuration classes that replace the dictionary-based
configuration system, offering type hints, validation, and better IDE support.

Example configurations for overlapping window processing:

    # Overlapping windows only (no artifact segmentation)
    preprocessing = PreprocessingConfig(
        process_segments=False,
        enable_overlapping_windows=True,
        window_length=60.0,      # 60-second windows
        window_overlap=0.5,      # 50% overlap (30 seconds)
    )

    # Overlapping windows with artifact segmentation
    preprocessing = PreprocessingConfig(
        process_segments=True,
        min_segment_length=30.0,
        enable_overlapping_windows=True,
        window_length=45.0,      # 45-second windows
        window_overlap=15.0,     # 15-second overlap (absolute)
    )

    # Small windows for memory efficiency
    preprocessing = PreprocessingConfig(
        enable_overlapping_windows=True,
        window_length=20.0,      # 20-second windows
        window_overlap=0.75,     # 75% overlap (15 seconds)
        min_window_size_ratio=0.6,
    )
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Union, List, Dict, Any
import numpy as np


@dataclass
class DataConfig:
    """
    Configuration for data loading.

    Attributes:
        ds_path: Path to the dataset directory
        patient: Patient identifier
        series: Series identifier
        sig_group_names: Signal group name(s)
        noise_group_names: Noise group name(s)
        noise_patient: Optional noise patient identifier
        noise_series: Optional noise series identifier
        sampfrom: Start sample index (default: 0)
        sampto: End sample index (default: None for entire signal)
    """
    ds_path: str
    patient: str
    series: str
    sig_group_names: str
    noise_group_names: str
    noise_patient: Optional[str] = None
    noise_series: Optional[str] = None
    sampfrom: int = 0
    sampto: Optional[int] = None


@dataclass
class PreprocessingConfig:
    """
    Configuration for signal preprocessing.

    Attributes:
        whitening: Whitening method ("PCA", "ZCA", or None)
        rescale_whitening: Whether to rescale after whitening
        bandpass_low: Low cutoff frequency for bandpass filter (Hz)
        bandpass_high: High cutoff frequency for bandpass filter (Hz)
        bandpass_order: Order of the bandpass filter
        signal_threshold: Threshold for signal quality (std deviations)
        noise_threshold: Threshold for noise quality
        downsample_factor: Downsampling factor
        process_segments: Enable segmented processing
        min_segment_length: Minimum segment length in seconds
    """
    whitening: Optional[str] = "PCA"
    rescale_whitening: bool = True
    bandpass_low: float = 1.0
    bandpass_high: float = 40.0
    bandpass_order: int = 4
    signal_threshold: float = 2.0
    noise_threshold: float = 0.1
    downsample_factor: int = 2
    process_segments: bool = True
    min_segment_length: float = 10.0

    # Overlapping window parameters
    enable_overlapping_windows: bool = False
    window_length: float = 60.0  # Window size in seconds
    window_overlap: Union[float, int] = 0.5  # Overlap: <1.0=ratio, >=1.0=seconds
    overlap_merge_method: str = "average"  # Future: "weighted", "ola"
    min_window_size_ratio: float = 0.5  # Minimum window as fraction of target

    def __post_init__(self):
        """Validate overlapping window parameters."""
        if self.enable_overlapping_windows:
            # Validate window_length
            if self.window_length <= 0:
                raise ValueError("window_length must be positive")

            # Validate overlap
            if isinstance(self.window_overlap, float) and self.window_overlap < 1.0:
                # Ratio mode
                if not 0.0 <= self.window_overlap < 1.0:
                    raise ValueError("window_overlap ratio must be in [0, 1)")
            elif self.window_overlap >= self.window_length:
                raise ValueError("window_overlap must be less than window_length")

            # Validate min_window_size_ratio
            if not 0.0 < self.min_window_size_ratio <= 1.0:
                raise ValueError("min_window_size_ratio must be in (0, 1]")

            # Warning for small overlap
            overlap_abs = (self.window_overlap if self.window_overlap >= 1.0
                          else self.window_overlap * self.window_length)
            if overlap_abs / self.window_length < 0.1:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Small overlap ({overlap_abs:.1f}s) may cause discontinuities at window boundaries"
                )


@dataclass
class LearningRateConfig:
    """
    Configuration for learning rate scheduling.

    Attributes:
        niter: Number of iterations
        warmup: Warmup iterations
        lr_groups: Learning rates for different parameter groups [lr_m_0, lr_m_1, lr_r]
        patience_groups: Patience for learning rate reduction
        early_stopping_patience: Patience for early stopping
        early_stopping_threshold: Threshold for early stopping
        min_iter: Minimum iterations before early stopping
        cooldown_iterations: Cooldown iterations after LR reduction
    """
    niter: int = 500
    warmup: int = 50
    lr_groups: List[float] = field(default_factory=lambda: [1e-1, 1e-1, 1e-2])
    patience_groups: List[int] = field(default_factory=lambda: [20])
    early_stopping_patience: int = 50
    early_stopping_threshold: float = 1e-8
    min_iter: int = 750
    cooldown_iterations: int = 10


@dataclass
class LossConfig:
    """
    Configuration for loss function parameters.

    Attributes:
        corr_m: Correlation penalty configuration
            - weight: Weight for correlation penalty
            - lag: Maximum lag for correlation
            - window: Window size for correlation
            - step: Step size for correlation computation
        r_tv: Total variation regularization for positions
            - weight: Weight for TV regularization
    """
    corr_m: Dict[str, Union[float, int]] = field(
        default_factory=lambda: {
            "weight": 5e-1,
            "lag": 50,
            "window": 5000,
            "step": 1
        }
    )
    r_tv: Dict[str, float] = field(
        default_factory=lambda: {
            "weight": 0.0
        }
    )


@dataclass
class InitializerConfig:
    """
    Configuration for solver initialization.

    Attributes:
        method: Initialization method
        rcond: Regularization parameter for pseudo-inverse
        determine_scaling: Whether to determine scaling automatically
        update_m_scaling: Whether to update moment scaling
        field_scaling: Scaling factor for field data
    """
    method: str = "pseudo_inv_regularized"
    rcond: float = 1e0
    determine_scaling: bool = True
    update_m_scaling: bool = True
    field_scaling: float = 1e0


@dataclass
class SolverConfig:
    """
    Configuration for the inverse solver.

    Attributes:
        parallel_processing: Enable parallel processing
        max_workers: Maximum number of parallel workers
        num_dipoles: Number of dipoles to fit
        optimizer: Optimizer type ("lamb", "adam", etc.)
        m_scaling: Scaling factors for dipole moments
        lr: Learning rate configuration
        loss_params: Loss function parameters
        initializer: Initializer configuration
        r_basis: Basis function for positions
        r_bnds: Bounds for dipole positions [num_dipoles, 3, 2]
        r_init: Initial positions [1, num_dipoles, 3]
        m_basis: Optional basis configuration for moments
    """
    parallel_processing: bool = True
    max_workers: int = 4
    num_dipoles: int = 2
    optimizer: str = "lamb"
    m_scaling: List[float] = field(default_factory=lambda: [1e0, 1e0])
    lr: LearningRateConfig = field(default_factory=LearningRateConfig)
    loss_params: LossConfig = field(default_factory=LossConfig)
    initializer: InitializerConfig = field(default_factory=InitializerConfig)
    r_basis: str = "sigmoid_time_constant"
    r_bnds: Optional[np.ndarray] = None
    r_init: Optional[np.ndarray] = None
    m_basis: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default numpy arrays if not provided."""
        if self.r_bnds is None:
            self.r_bnds = np.array([
                [(-0.3, 0.3), (-0.3, -0.005), (-0.3, 0.3)],
                [(-0.3, 0.3), (-0.3, -0.005), (0.1, 0.6)]
            ])
        if self.r_init is None:
            self.r_init = np.array([[[0, -0.02, 0.0], [0, -0.05, 0.2]]])

        # Convert nested dicts to dataclasses if needed
        if isinstance(self.lr, dict):
            self.lr = LearningRateConfig(**self.lr)
        if isinstance(self.loss_params, dict):
            self.loss_params = LossConfig(**self.loss_params)
        if isinstance(self.initializer, dict):
            self.initializer = InitializerConfig(**self.initializer)


@dataclass
class AveragingConfig:
    """
    Configuration for heartbeat averaging.

    Attributes:
        method: Averaging method
        tol: Tolerance for fetal and maternal heartbeats
        min_block_length: Minimum block length for averaging
        outlier_kwargs: Parameters for outlier detection
    """
    method: str = "selective_averaging"
    tol: Dict[str, float] = field(default_factory=lambda: {"fetal": 3, "maternal": 5})
    min_block_length: int = 4
    outlier_kwargs: Dict[str, int] = field(default_factory=lambda: {"threshold": 10, "win": 4})


@dataclass
class PostProcessingConfig:
    """
    Configuration for post-processing and heartbeat detection.

    Attributes:
        method: Detection method ("subsequent", "subsequent_LMS", etc.)
        decomposition: Decomposition method ("ica", "pca", etc.)
        nlms: Use normalized LMS
        mu: Learning rate for LMS
        consider_mean: Consider mean for IBI calculation (for subsequent_LMS)
        wavelet_denoise: Enable wavelet denoising
        wavelet_denoise_threshold: Threshold for wavelet denoising
        ica_components: Number of ICA components
        n_trials: Number of trials for decomposition
        averaging: Averaging configuration
        segment_length: Optional segment length for processing
    """
    method: str = "subsequent"
    decomposition: str = "ica"
    nlms: bool = False
    mu: float = 0.001
    consider_mean: bool = False
    wavelet_denoise: bool = False
    wavelet_denoise_threshold: float = 100.0
    ica_components: int = 3
    n_trials: int = 3
    averaging: AveragingConfig = field(default_factory=AveragingConfig)
    segment_length: Optional[float] = None

    def __post_init__(self):
        """Convert nested dict to dataclass if needed."""
        if isinstance(self.averaging, dict):
            self.averaging = AveragingConfig(**self.averaging)


@dataclass
class PipelineConfig:
    """
    Top-level configuration for the fMCG pipeline.

    Attributes:
        data: Data loading configuration
        preprocessing: Preprocessing configuration
        solver: Solver configuration
        post_processing: Post-processing configuration
        skip_existing: Skip if output already exists
        catch_errors: Catch and log errors instead of raising
        device: Device for computation ("cpu", "cuda:0", etc.)
        save_reports: Save HTML/text reports
        save_plots: Save diagnostic plots
        save_data: Save processed data
        output_dir: Output directory path
    """
    data: DataConfig
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    post_processing: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    skip_existing: bool = True
    catch_errors: bool = True
    device: str = "cpu"
    save_reports: Union[bool, List[str]] = False
    save_plots: bool = False
    save_data: bool = True
    output_dir: str = "output"

    def __post_init__(self):
        """Convert nested dicts to dataclasses if needed."""
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.preprocessing, dict):
            self.preprocessing = PreprocessingConfig(**self.preprocessing)
        if isinstance(self.solver, dict):
            self.solver = SolverConfig(**self.solver)
        if isinstance(self.post_processing, dict):
            self.post_processing = PostProcessingConfig(**self.post_processing)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.

        This is useful for backward compatibility with code expecting dict configs.

        Returns:
            Dictionary representation of the configuration
        """
        def convert_value(v):
            """Recursively convert dataclasses and numpy arrays to serializable types."""
            if isinstance(v, np.ndarray):
                return v  # Keep numpy arrays as-is for compatibility
            elif hasattr(v, '__dataclass_fields__'):
                return asdict(v)
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            elif isinstance(v, list):
                return [convert_value(val) for val in v]
            return v

        result = asdict(self)
        return {k: convert_value(v) for k, v in result.items()}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """
        Create a PipelineConfig from a dictionary.

        This allows loading from legacy dict-based configs.

        Args:
            config_dict: Dictionary configuration

        Returns:
            PipelineConfig instance
        """
        return cls(**config_dict)
