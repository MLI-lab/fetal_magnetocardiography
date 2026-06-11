"""
Moving Average Epoch Averaging Filter for Heartbeat Signals

Denoises time series signals with detected heartbeat peaks using:
- Adaptive epoch extraction based on RR intervals
- Centered sliding window averaging across epochs
- Continuous signal reconstruction with overlap handling
"""

import numpy as np
from scipy import interpolate
from typing import Optional, Tuple


class EpochAveragingFilter:
    """
    Moving average epoch averaging filter for heartbeat signal denoising.
    
    Parameters
    ----------
    window_size : int
        Number of epochs to average (total window = 2*window_size + 1)
        E.g., window_size=5 means average epochs [N-5, ..., N, ..., N+5]
    epoch_pre_fraction : float, default=0.4
        Fraction of RR interval to include before peak
    epoch_post_fraction : float, default=0.6
        Fraction of RR interval to include after peak
    interpolation_points : int, default=200
        Number of points for normalizing epoch lengths
    overlap_method : str, default='weighted'
        Method for handling overlapping regions in reconstruction
        Options: 'weighted', 'average', 'last'
    """
    
    def __init__(
        self,
        window_size: int = 5,
        epoch_pre_fraction: float = 0.4,
        epoch_post_fraction: float = 0.6,
        interpolation_points: int = 200,
        overlap_method: str = 'weighted'
    ):
        self.window_size = window_size
        self.epoch_pre_fraction = epoch_pre_fraction
        self.epoch_post_fraction = epoch_post_fraction
        self.interpolation_points = interpolation_points
        self.overlap_method = overlap_method
        
    def extract_epochs(
        self, 
        signal: np.ndarray, 
        peak_indices: np.ndarray
    ) -> Tuple[list, list, list]:
        """
        Extract epochs around each peak with adaptive window sizes.
        
        Parameters
        ----------
        signal : np.ndarray
            Input time series signal (1D array of shape (T,) or 2D array of shape (T, C))
        peak_indices : np.ndarray
            Indices of detected peaks in the signal
            
        Returns
        -------
        epochs : list of np.ndarray
            Extracted epochs (varying lengths). For 2D input, each epoch has shape (samples, channels)
        epoch_starts : list of int
            Start index of each epoch in original signal
        epoch_ends : list of int
            End index of each epoch in original signal
        """
        peak_indices = np.asarray(peak_indices, dtype=int)
        n_peaks = len(peak_indices)
        
        # Get signal length (works for both 1D and 2D)
        signal_length = signal.shape[0]
        
        # Calculate RR intervals
        rr_intervals = np.diff(peak_indices)
        
        epochs = []
        epoch_starts = []
        epoch_ends = []
        
        for i, peak_idx in enumerate(peak_indices):
            # Determine local RR interval for adaptive windowing
            if i == 0:
                # First peak: use next RR interval
                local_rr = rr_intervals[0] if n_peaks > 1 else 200
            elif i == n_peaks - 1:
                # Last peak: use previous RR interval
                local_rr = rr_intervals[-1]
            else:
                # Middle peaks: use average of adjacent RR intervals
                local_rr = (rr_intervals[i-1] + rr_intervals[i]) / 2
            
            # Calculate epoch boundaries
            pre_samples = int(local_rr * self.epoch_pre_fraction)
            post_samples = int(local_rr * self.epoch_post_fraction)
            
            start_idx = max(0, peak_idx - pre_samples)
            end_idx = min(signal_length, peak_idx + post_samples)
            
            # Extract epoch (handles both 1D and 2D)
            epoch = signal[start_idx:end_idx]
            
            epochs.append(epoch)
            epoch_starts.append(start_idx)
            epoch_ends.append(end_idx)
        
        return epochs, epoch_starts, epoch_ends
    
    def normalize_epoch_lengths(self, epochs: list) -> np.ndarray:
        """
        Normalize varying-length epochs to fixed length via interpolation.
        
        Parameters
        ----------
        epochs : list of np.ndarray
            Epochs with varying lengths. Can be 1D (samples,) or 2D (samples, channels)
            
        Returns
        -------
        normalized_epochs : np.ndarray
            For 1D input: shape (n_epochs, interpolation_points)
            For 2D input: shape (n_epochs, interpolation_points, n_channels)
        """
        n_epochs = len(epochs)
        
        # Check if epochs are multi-channel
        is_multichannel = epochs[0].ndim == 2
        n_channels = epochs[0].shape[1] if is_multichannel else 1
        
        if is_multichannel:
            normalized = np.zeros((n_epochs, self.interpolation_points, n_channels))
        else:
            normalized = np.zeros((n_epochs, self.interpolation_points))
        
        for i, epoch in enumerate(epochs):
            epoch_len = epoch.shape[0]
            
            if epoch_len < 2:
                # Handle edge case of very short epochs
                if is_multichannel:
                    normalized[i, :, :] = np.mean(epoch, axis=0) if epoch_len > 0 else 0
                else:
                    normalized[i, :] = np.mean(epoch) if epoch_len > 0 else 0
                continue
            
            # Create interpolation coordinates
            x_old = np.linspace(0, 1, epoch_len)
            x_new = np.linspace(0, 1, self.interpolation_points)
            
            if is_multichannel:
                # Interpolate each channel separately
                for ch in range(n_channels):
                    f = interpolate.interp1d(x_old, epoch[:, ch], kind='cubic', 
                                            fill_value='extrapolate')
                    normalized[i, :, ch] = f(x_new)
            else:
                # Interpolate single channel
                f = interpolate.interp1d(x_old, epoch, kind='cubic', 
                                        fill_value='extrapolate')
                normalized[i, :] = f(x_new)
        
        return normalized
    
    def apply_moving_average(self, normalized_epochs: np.ndarray, return_std: bool = False) -> tuple:
        """
        Apply centered sliding window average across epochs.
        
        Parameters
        ----------
        normalized_epochs : np.ndarray
            2D array of shape (n_epochs, n_samples) for single-channel
            3D array of shape (n_epochs, n_samples, n_channels) for multi-channel
        return_std : bool, default=False
            If True, also return standard deviation across epochs
            
        Returns
        -------
        averaged_epochs : np.ndarray
            Same shape as input, each epoch averaged with neighbors
        std_epochs : np.ndarray, optional
            Standard deviation across averaged epochs (only if return_std=True)
        """
        n_epochs = normalized_epochs.shape[0]
        averaged = np.zeros_like(normalized_epochs)
        
        if return_std:
            std = np.zeros_like(normalized_epochs)
        
        for i in range(n_epochs):
            # Determine window boundaries
            start_idx = max(0, i - self.window_size)
            end_idx = min(n_epochs, i + self.window_size + 1)
            
            # Average epochs in window (works for both 2D and 3D)
            averaged[i] = np.mean(normalized_epochs[start_idx:end_idx], axis=0)
            
            if return_std:
                # Compute standard deviation across epochs in window
                std[i] = np.std(normalized_epochs[start_idx:end_idx], axis=0, ddof=1)
        
        if return_std:
            return averaged, std
        return averaged
    
    def denormalize_epochs(
        self, 
        averaged_epochs: np.ndarray,
        original_epochs: list
    ) -> list:
        """
        Convert normalized averaged epochs back to original lengths.
        
        Parameters
        ----------
        averaged_epochs : np.ndarray
            2D array (n_epochs, n_samples) or 3D array (n_epochs, n_samples, n_channels)
        original_epochs : list of np.ndarray
            Original epochs with varying lengths
            
        Returns
        -------
        denormalized_epochs : list of np.ndarray
            Averaged epochs restored to original lengths and shapes
        """
        is_multichannel = averaged_epochs.ndim == 3
        n_channels = averaged_epochs.shape[2] if is_multichannel else 1
        
        denormalized = []
        
        for i, orig_epoch in enumerate(original_epochs):
            epoch_len = orig_epoch.shape[0]
            
            if epoch_len < 2:
                # Handle very short epochs
                if is_multichannel:
                    denormalized.append(np.full_like(orig_epoch, np.mean(averaged_epochs[i], axis=0)))
                else:
                    denormalized.append(np.full_like(orig_epoch, np.mean(averaged_epochs[i])))
                continue
            
            # Interpolate back to original length
            x_norm = np.linspace(0, 1, self.interpolation_points)
            x_orig = np.linspace(0, 1, epoch_len)
            
            if is_multichannel:
                # Interpolate each channel
                denorm_epoch = np.zeros((epoch_len, n_channels))
                for ch in range(n_channels):
                    f = interpolate.interp1d(x_norm, averaged_epochs[i, :, ch], kind='cubic',
                                            fill_value='extrapolate')
                    denorm_epoch[:, ch] = f(x_orig)
                denormalized.append(denorm_epoch)
            else:
                # Single channel
                f = interpolate.interp1d(x_norm, averaged_epochs[i], kind='cubic',
                                        fill_value='extrapolate')
                denormalized.append(f(x_orig))
        
        return denormalized
    
    def compute_beat_quality(
        self,
        normalized_epochs: np.ndarray,
        averaged_epochs: np.ndarray
    ) -> dict:
        """
        Compute quality metrics for each beat based on deviation from average.
        
        Parameters
        ----------
        normalized_epochs : np.ndarray
            Original normalized epochs (n_epochs, n_samples) or (n_epochs, n_samples, n_channels)
        averaged_epochs : np.ndarray
            Averaged epochs (same shape as normalized_epochs)
            
        Returns
        -------
        quality_metrics : dict
            Dictionary containing:
            - 'rms_deviation': RMS deviation from average for each epoch
            - 'correlation': Correlation coefficient with average for each epoch  
            - 'snr_estimate': Estimated SNR for each epoch
            - 'normalized_error': Normalized error (0-1 scale) for each epoch
            - 'outlier_flags': Boolean array marking potential outliers
        """
        n_epochs = normalized_epochs.shape[0]
        is_multichannel = normalized_epochs.ndim == 3
        
        if is_multichannel:
            n_channels = normalized_epochs.shape[2]
            # Average across channels for overall metrics
            rms_deviation = np.zeros(n_epochs)
            correlation = np.zeros(n_epochs)
            snr_estimate = np.zeros(n_epochs)
            
            for i in range(n_epochs):
                # Compute per-channel metrics then average
                epoch = normalized_epochs[i]  # (n_samples, n_channels)
                avg = averaged_epochs[i]      # (n_samples, n_channels)
                
                # RMS deviation across all channels
                diff = epoch - avg
                rms_deviation[i] = np.sqrt(np.mean(diff**2))
                
                # Average correlation across channels
                corr_per_channel = []
                for ch in range(n_channels):
                    if np.std(epoch[:, ch]) > 1e-10 and np.std(avg[:, ch]) > 1e-10:
                        corr = np.corrcoef(epoch[:, ch], avg[:, ch])[0, 1]
                        corr_per_channel.append(corr)
                correlation[i] = np.mean(corr_per_channel) if corr_per_channel else 0
                
                # SNR estimate: signal power / noise power
                signal_power = np.mean(avg**2)
                noise_power = np.mean(diff**2)
                if noise_power > 1e-10:
                    snr_estimate[i] = 10 * np.log10(signal_power / noise_power)
                else:
                    snr_estimate[i] = 100  # Very high SNR
        else:
            # Single channel
            rms_deviation = np.zeros(n_epochs)
            correlation = np.zeros(n_epochs)
            snr_estimate = np.zeros(n_epochs)
            
            for i in range(n_epochs):
                epoch = normalized_epochs[i]
                avg = averaged_epochs[i]
                
                # RMS deviation
                diff = epoch - avg
                rms_deviation[i] = np.sqrt(np.mean(diff**2))
                
                # Correlation
                if np.std(epoch) > 1e-10 and np.std(avg) > 1e-10:
                    correlation[i] = np.corrcoef(epoch, avg)[0, 1]
                else:
                    correlation[i] = 0
                
                # SNR estimate
                signal_power = np.mean(avg**2)
                noise_power = np.mean(diff**2)
                if noise_power > 1e-10:
                    snr_estimate[i] = 10 * np.log10(signal_power / noise_power)
                else:
                    snr_estimate[i] = 100
        
        # Normalized error (0 = perfect match, 1 = large deviation)
        max_rms = np.percentile(rms_deviation, 95)  # Use 95th percentile as reference
        if max_rms > 1e-10:
            normalized_error = np.clip(rms_deviation / max_rms, 0, 1)
        else:
            normalized_error = np.zeros(n_epochs)
        
        # Outlier detection using modified z-score (robust to outliers)
        median_rms = np.median(rms_deviation)
        mad = np.median(np.abs(rms_deviation - median_rms))  # Median absolute deviation
        if mad > 1e-10:
            modified_z_score = 0.6745 * (rms_deviation - median_rms) / mad
            outlier_flags = np.abs(modified_z_score) > 3.5  # Threshold for outliers
        else:
            outlier_flags = np.zeros(n_epochs, dtype=bool)
        
        quality_metrics = {
            'rms_deviation': rms_deviation,
            'correlation': correlation,
            'snr_estimate': snr_estimate,
            'normalized_error': normalized_error,
            'outlier_flags': outlier_flags
        }
        
        return quality_metrics
    
    def reconstruct_signal(
        self,
        denoised_epochs: list,
        epoch_starts: list,
        epoch_ends: list,
        signal_length: int,
        original_signal: np.ndarray,
        n_channels: int = 1,
        min_weight_threshold: float = 0.01
    ) -> np.ndarray:
        """
        Reconstruct continuous signal from overlapping denoised epochs.
        
        Fills gaps (regions not covered by any epoch) with the original signal
        to avoid holes in the reconstruction.
        
        Parameters
        ----------
        denoised_epochs : list of np.ndarray
            Denoised epochs to reconstruct from
        epoch_starts : list of int
            Start indices of epochs
        epoch_ends : list of int
            End indices of epochs
        signal_length : int
            Length of original signal
        original_signal : np.ndarray
            Original signal to use for filling gaps
        n_channels : int, default=1
            Number of channels (auto-detected from epochs)
        min_weight_threshold : float, default=0.01
            Minimum weight threshold - regions below this use original signal
            
        Returns
        -------
        reconstructed_signal : np.ndarray
            Reconstructed continuous signal (T,) for single channel or (T, C) for multi-channel
        """
        # Auto-detect number of channels
        is_multichannel = denoised_epochs[0].ndim == 2
        if is_multichannel:
            n_channels = denoised_epochs[0].shape[1]
            reconstructed = np.zeros((signal_length, n_channels))
            weights = np.zeros((signal_length, n_channels))
        else:
            reconstructed = np.zeros(signal_length)
            weights = np.zeros(signal_length)
        
        for epoch, start, end in zip(denoised_epochs, epoch_starts, epoch_ends):
            epoch_len = end - start
            
            if self.overlap_method == 'weighted':
                # Use triangular weighting (higher weight at peak center)
                weight = np.bartlett(epoch_len)
                if len(weight) == 0:
                    weight = np.ones(1)
            elif self.overlap_method == 'average':
                # Equal weighting
                weight = np.ones(epoch_len)
            else:  # 'last'
                # Just overwrite (last epoch wins)
                weight = np.ones(epoch_len)
                reconstructed[start:end] = epoch
                continue
            
            # Reshape weight for broadcasting if multi-channel
            if is_multichannel:
                weight = weight[:, np.newaxis]  # (epoch_len, 1)
            
            # Accumulate weighted contributions
            reconstructed[start:end] += epoch * weight
            weights[start:end] += weight
        
        # Identify regions with insufficient coverage (gaps/holes)
        if is_multichannel:
            insufficient_coverage = np.any(weights < min_weight_threshold, axis=1)
        else:
            insufficient_coverage = weights < min_weight_threshold
        
        # Normalize by accumulated weights where we have coverage
        valid_weights = weights.copy()
        valid_weights[valid_weights == 0] = 1  # Avoid division by zero
        reconstructed /= valid_weights
        
        # Fill gaps with original signal
        if np.any(insufficient_coverage):
            if is_multichannel:
                # For multi-channel, use broadcasting
                reconstructed[insufficient_coverage, :] = original_signal[insufficient_coverage, :]
            else:
                reconstructed[insufficient_coverage] = original_signal[insufficient_coverage]
        
        return reconstructed
    
    def filter(
        self, 
        signal: np.ndarray, 
        peak_indices: np.ndarray,
        return_quality: bool = False,
        return_std: bool = False
    ) -> tuple:
        """
        Apply the complete epoch averaging filter pipeline.
        
        Automatically fills gaps in reconstruction (regions not covered by epochs)
        with the original signal to avoid holes.
        
        Parameters
        ----------
        signal : np.ndarray
            Input time series signal (T,) for single-channel or (T, C) for multi-channel
        peak_indices : np.ndarray
            Indices of detected peaks
        return_quality : bool, default=False
            If True, include quality_metrics in return tuple
        return_std : bool, default=False
            If True, include std_signal in return tuple showing beat-to-beat variability
            
        Returns
        -------
        denoised_signal : np.ndarray
            Reconstructed denoised signal with same shape as input.
            Gaps (uncovered regions) are filled with original signal.
        std_signal : np.ndarray, optional
            Standard deviation signal (only if return_std=True).
            Shows beat-to-beat variability at each time point.
            Low std = consistent beats, high std = variable/noisy beats.
            Gaps are filled with zeros (no std information available).
        quality_metrics : dict, optional
            Only returned if return_quality=True. Contains per-beat quality metrics.
        """
        # Extract epochs with adaptive windows
        epochs, starts, ends = self.extract_epochs(signal, peak_indices)
        
        # Normalize epoch lengths
        normalized = self.normalize_epoch_lengths(epochs)
        
        # Apply moving average (optionally compute std)
        if return_std:
            averaged, std_normalized = self.apply_moving_average(normalized, return_std=True)
        else:
            averaged = self.apply_moving_average(normalized, return_std=False)
        
        # Compute quality metrics if requested
        if return_quality:
            quality_metrics = self.compute_beat_quality(normalized, averaged)
            quality_metrics['peak_indices'] = peak_indices.copy()
        
        # Denormalize back to original lengths
        denoised_epochs = self.denormalize_epochs(averaged, epochs)
        
        if return_std:
            std_epochs = self.denormalize_epochs(std_normalized, epochs)
        
        # Reconstruct continuous signal (fills gaps with original signal)
        signal_length = signal.shape[0]
        denoised_signal = self.reconstruct_signal(
            denoised_epochs, starts, ends, signal_length, 
            original_signal=signal
        )
        
        if return_std:
            # Reconstruct std signal (gaps filled with zeros - no std info)
            is_multichannel = signal.ndim == 2
            zero_signal = np.zeros_like(signal)
            
            std_signal = self.reconstruct_signal(
                std_epochs, starts, ends, signal_length,
                original_signal=zero_signal  # Gaps have zero std (no information)
            )
        
        # Return appropriate tuple based on flags
        if return_std and return_quality:
            return denoised_signal, std_signal, quality_metrics
        elif return_std:
            return denoised_signal, std_signal
        elif return_quality:
            return denoised_signal, quality_metrics
        else:
            return denoised_signal


def epoch_averaging_filter(
    signal: np.ndarray,
    peak_indices: np.ndarray,
    window_size: int = 5,
    epoch_pre_fraction: float = 0.4,
    epoch_post_fraction: float = 0.6,
    return_quality: bool = False,
    return_std: bool = False,
    **kwargs
) -> tuple:
    """
    Convenience function for epoch averaging filtering.
    
    Works with both single-channel (T,) and multi-channel (T, C) signals.
    
    Parameters
    ----------
    signal : np.ndarray
        Input time series signal (T,) for single-channel or (T, C) for multi-channel
    peak_indices : np.ndarray
        Indices of detected peaks
    window_size : int, default=5
        Number of epochs on each side to average
    epoch_pre_fraction : float, default=0.4
        Fraction of RR interval before peak
    epoch_post_fraction : float, default=0.6
        Fraction of RR interval after peak
    return_quality : bool, default=False
        If True, include per-beat quality metrics in return
    return_std : bool, default=False
        If True, include continuous std signal showing beat-to-beat variability
    **kwargs
        Additional parameters passed to EpochAveragingFilter
        
    Returns
    -------
    denoised_signal : np.ndarray
        Denoised signal with same shape as input
    std_signal : np.ndarray, optional
        Continuous standard deviation signal (only if return_std=True)
    quality_metrics : dict, optional
        Per-beat quality metrics (only if return_quality=True)
        
    Examples
    --------
    >>> # Single-channel signal
    >>> signal = np.random.randn(1000)
    >>> peaks = np.array([100, 200, 300, 400, 500])
    >>> denoised = epoch_averaging_filter(signal, peaks, window_size=3)
    
    >>> # With std signal for uncertainty visualization
    >>> denoised, std = epoch_averaging_filter(signal, peaks, return_std=True)
    
    >>> # Multi-channel signal (e.g., 8-channel fMCG)
    >>> signal = np.random.randn(10000, 8)
    >>> peaks = np.array([500, 1000, 1500, 2000])
    >>> denoised, std = epoch_averaging_filter(signal, peaks, window_size=5, return_std=True)
    """
    filter_obj = EpochAveragingFilter(
        window_size=window_size,
        epoch_pre_fraction=epoch_pre_fraction,
        epoch_post_fraction=epoch_post_fraction,
        **kwargs
    )
    return filter_obj.filter(signal, peak_indices, return_quality=return_quality, return_std=return_std)
