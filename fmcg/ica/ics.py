
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks


from copy import deepcopy

def ics(ica, bp_data, sources, comp, fs, keep_sources=False, spline=False, field_subtract=True, downsample_factor_qrs=5, downsample_factor_other=20):
    ica_mixing_ = ica.mixing_.copy()

    # Either keep the selected components in the sources or zero them out
    if keep_sources:
        rec_sources = np.zeros_like(sources)
        rec_sources[:, comp] = sources[:, comp].copy()
    else:
        rec_sources = deepcopy(sources)
        rec_sources[:, comp] = 0

    # Optionally smooth source components to reduce residual components of other beats (e.g. fetal in maternal component)
    if spline and keep_sources:
        rec_sources = upDownSample(
            rec_sources, comp, fs, downsample_factor_qrs=downsample_factor_qrs, downsample_factor_other=downsample_factor_other
        )

    # Project back to the signal / field domain
    rec_meas = rec_sources @ ica_mixing_.T

    if ica.whiten:
        rec_meas += ica.mean_

    # Either subtract the reconstructed signal from the original data or return the reconstructed signal itself
    if field_subtract:
        return bp_data - rec_meas, rec_meas
    return rec_meas, bp_data - rec_meas



def upDownSample(
    maternal, maternal_indices, fs, downsample_factor_qrs=5, downsample_factor_other=20
):
    """Downsampling factors according to [1] Yu and Wakai, ‘Maternal MCG Interference Cancellation Using Splined Independent Component Subtraction’."""
    maternal_matrix = deepcopy(maternal)
    for index in maternal_indices:
        candidate = maternal_matrix[:, index]
        peak_indices, _ = find_peaks(np.abs(candidate), distance=int(0.4 * fs), prominence=2)

        # Create a mask for downsampling
        downsample_mask = np.zeros_like(candidate, dtype=bool)

        for peak in peak_indices:
            start_qrs = max(0, peak - int(0.05 * fs))
            end_qrs = min(len(candidate), peak + int(0.05 * fs))
            downsample_mask[start_qrs:end_qrs] = True

        # Downsample
        downsampled_signal = []
        indices = []
        for i in range(len(candidate)):
            if downsample_mask[i]:
                if i % int(downsample_factor_qrs * fs / 520.8) == 0:
                    downsampled_signal.append(candidate[i])
                    indices.append(i)
            else:
                if i % int(downsample_factor_other * fs / 520.8) == 0:
                    downsampled_signal.append(candidate[i])
                    indices.append(i)

        # Convert to numpy arrays
        downsampled_signal = np.array(downsampled_signal)
        indices = np.array(indices)

        # Perform cubic spline interpolation
        cs = CubicSpline(indices, downsampled_signal)
        upsampled_signal = cs(np.arange(len(candidate)))

        # Replace the original signal with the upsampled signal
        maternal_matrix[:, index] = upsampled_signal

    return maternal_matrix