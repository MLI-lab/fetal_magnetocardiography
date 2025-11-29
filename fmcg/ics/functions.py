"""
Implementation of a splined independent component subtraction approach for maternal signal cancellation in fMCG. The implementation is predominantly based on [1].

[1] S. Yu and R. T. Wakai, 'Maternal MCG interference cancellation using splined independent component subtraction', IEEE Trans Biomed Eng, vol. 58, no. 10, pp. 2835-2843, Oct. 2011, doi: 10.1109/TBME.2011.2160635.


@author: Serden Eranil
@author: Jonas Emrich
"""

import numpy as np
from sklearn.decomposition import FastICA
from scipy.signal import find_peaks
from scipy.stats import kurtosis
import neurokit2 as nk
import matplotlib.pyplot as plt
from copy import deepcopy
import pywt
from scipy.interpolate import CubicSpline
import logging

logger = logging.getLogger(__name__)

def apply_fastICA(dd, n_comp=None, algo="parallel", seed=42):
    ica = FastICA(
        n_components=n_comp, algorithm=algo, max_iter=500, tol=1e-5, random_state=seed
    )
    S_ = ica.fit_transform(dd)
    return S_, ica


def plotICA(
    sources,
    ica=None,
    kurt_threshold=2,
    offset=10000,
    span=3000,
    fs=1000,
    mse_th=1,
    plt_save=False,
    plot=True,
):
    kurtosis_values = [kurtosis(sources[:, j]) for j in range(sources.shape[1])]
    kurt_list = [
        j for j, kurt in enumerate(kurtosis_values) if kurt_threshold < kurt < 200
    ]
    channels_to_plot = []
    mse_list = []
    for chan in kurt_list:
        signal = sources[offset : offset + span, chan]

        import neurokit2 as nk

        signal_, _ = nk.ecg_invert(signal, fs)
        signal_clean = nk.ecg_clean(signal_, sampling_rate=fs, method="vg")
        peaks = nk.ecg_findpeaks(signal_clean, sampling_rate=fs, method="vg")[
            "ECG_R_Peaks"
        ]
        # peaks, _ = find_peaks(np.abs(signal), distance=200, prominence=1)
        if len(peaks) < 1 or peaks is None:
            continue
        peak_heights = signal[peaks]
        mean_height = np.mean(peak_heights)
        mse = np.mean((peak_heights - mean_height) ** 2)

        # Normalize MSE by the square of the mean height to make it scale-invariant
        normalized_mse = mse / (mean_height**2)

        # print(chan, normalized_mse)

        # Check if normalized MSE is below threshold (smaller is better)
        if normalized_mse < mse_th:
            channels_to_plot.append(chan)
            mse_list.append(normalized_mse)

    if not channels_to_plot:
        return -1, channels_to_plot

    if plot:
        fig, axes = plt.subplots(
            len(channels_to_plot),
            1,
            sharex=True,
            figsize=(12, 2 * len(channels_to_plot)),
            dpi=150,
        )
        fig.subplots_adjust(hspace=0.3)

        if len(channels_to_plot) == 1:
            axes = [axes]

    maxPeaks = 0
    compF = -1
    for i, j in enumerate(channels_to_plot):
        signal = sources[offset : offset + span, j]
        # peaks, _ = find_peaks(np.abs(signal), distance=400, prominence=2)
        signal_, _ = nk.ecg_invert(signal, fs)
        signal_clean = nk.ecg_clean(signal_, sampling_rate=fs, method="vg")
        peaks = nk.ecg_findpeaks(signal_clean, sampling_rate=fs, method="vg")[
            "ECG_R_Peaks"
        ]
        if len(peaks) > maxPeaks:
            maxPeaks = len(peaks)
            compF = j

        if plot:
            axes[i].plot(np.arange(span) / fs, signal, label=f"Component {j}")
            axes[i].plot(peaks / fs, signal[peaks], "rx", markersize=8)
            axes[i].set_ylabel("Amplitude")
            axes[i].legend(loc="upper right")
            axes[i].text(
                0.02,
                0.95,
                f"Kurtosis: {kurtosis_values[j]:.2f}\nNorm. MSE: {mse_list[i]:.2f}\nPeaks: {len(peaks)}",
                transform=axes[i].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )
            axes[i].grid(True, linestyle="--", alpha=0.7)

    if plot:
        axes[-1].set_xlabel("Time [s]")
        params = ica.get_params()
        plt.suptitle(
            f"ICA selected: {len(channels_to_plot)} components with kurtosis > {kurt_threshold}, algo:{params['algorithm']}, comps:{params['n_components']}, mse_th = {mse_th}",
            fontsize=16,
        )

        plt.tight_layout()
        if plt_save:
            # Save the plot as SVG (vector format)
            plt.savefig("data_comparison.svg", format="svg", bbox_inches="tight")
        plt.show()
    maternals = deepcopy(channels_to_plot)
    maternals.remove(compF)
    return compF, maternals


def plotHR(signal, compF=None, h=2, d=400, fs=1000, minBpm=-1, maxBpm=-1, saveplt=False, plot=True):

    def _find_peaks(signal):
        # Find all peaks
        signal, _ = nk.ecg_invert(signal, fs)
        signal_clean = nk.ecg_clean(signal, sampling_rate=fs, method="vg")
        return nk.ecg_findpeaks(signal_clean, sampling_rate=fs, method="vg")["ECG_R_Peaks"]

    if compF is not None and signal.ndim > 1 and len(compF) > 1:
        logger.debug("Selecting the Fetal Component with lowest sdnn")
        i = np.argmin([np.std(np.diff(_find_peaks(signal[:, f]))) for f in compF])
        peaks = _find_peaks(signal[:, compF[i]])
        logger.debug(f"Selected component {compF[i]} with sdnn {np.std(np.diff(peaks))}")
    else:
        peaks = _find_peaks(signal[:, compF]) if compF is not None else _find_peaks(signal)

    # Calculate intervals and heart rates for all peaks
    all_intervals = np.diff(peaks) / fs
    heart_rates = 60 / all_intervals

    # Filter peaks based on heart rate criteria
    if minBpm != -1 and maxBpm != -1:
        valid_indices = np.where((heart_rates >= minBpm) & (heart_rates <= maxBpm))[0]
        peaks = peaks[:-1][
            valid_indices
        ]  # Exclude the last peak as it doesn't have a corresponding interval
        heart_rates = heart_rates[valid_indices]

    avg = np.mean(heart_rates)

    if plot:
        # Create the plot
        plt.figure(figsize=(12, 6))

        # Plot vertical lines
        x_positions = np.arange(len(heart_rates))
        plt.vlines(x_positions, ymin=0, ymax=heart_rates, colors="b", linewidth=1)

        # Customize the plot
        plt.xlabel("Beat Number")
        plt.ylabel(f"Heart Rate (BPM)")
        plt.title(f"Heart Rate avg = {avg:.2f} - Fetal")
        plt.grid(True, linestyle="--", alpha=0.9)

        num_ticks = 30  # Adjust this number to control how many ticks you want
        tick_locations = np.linspace(0, len(heart_rates) - 1, num_ticks, dtype=int)
        plt.xticks(tick_locations, tick_locations)
        plt.yticks(np.linspace(60, 150, 20, dtype=int), np.linspace(60, 150, 20, dtype=int))

        # Set y-axis to start from 100
        plt.ylim(bottom=60)

        # Add markers at the top of each line
        plt.plot(x_positions, heart_rates, "ro", markersize=2)

        # Plot all data points (they're all between minBpm and maxBpm now)
        plt.plot(
            x_positions, heart_rates, "go", markersize=4, label="Between Min and Max BPM"
        )

        # plt.savefig('heartRate.svg',  format='svg', bbox_inches='tight')
        plt.legend()
        plt.tight_layout()

        plt.show()

    return heart_rates, peaks


def upDownSample(
    maternal, maternal_indices, downsample_factor_qrs=5, downsample_factor_other=20
):
    maternal_matrix = deepcopy(maternal)
    for index in maternal_indices:
        candidate = maternal_matrix[:, index]
        peak_indices, _ = find_peaks(np.abs(candidate), distance=400, prominence=2)

        # Create a mask for downsampling
        downsample_mask = np.zeros_like(candidate, dtype=bool)

        for peak in peak_indices:
            start_qrs = max(0, peak - 50)
            end_qrs = min(len(candidate), peak + 50)
            downsample_mask[start_qrs:end_qrs] = True

        # Downsample
        downsampled_signal = []
        indices = []
        for i in range(len(candidate)):
            if downsample_mask[i]:
                if i % downsample_factor_qrs == 0:
                    downsampled_signal.append(candidate[i])
                    indices.append(i)
            else:
                if i % downsample_factor_other == 0:
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


def identify_hr_range_segments(
    heart_rates, min_hr, max_hr, plot=False, min_segment_length=5, plt_save=False
):
    range_segments = []
    start_index = None

    for i, hr in enumerate(heart_rates):
        if min_hr <= hr <= max_hr:
            if start_index is None:
                start_index = i
        else:
            if start_index is not None:
                if i - start_index >= min_segment_length:
                    range_segments.append((start_index, i))
                start_index = None

    # Check if the last segment extends to the end
    if start_index is not None and len(heart_rates) - start_index >= min_segment_length:
        range_segments.append((start_index, len(heart_rates)))
    if plot:
        plt.figure(figsize=(15, 6))
        plt.plot(heart_rates, "b-", label="Heart Rate")
        for start, end in range_segments:
            plt.axvspan(start, end, color="green", alpha=0.3)
        plt.axhline(y=min_hr, color="r", linestyle="--", label="Min HR")
        plt.axhline(y=max_hr, color="r", linestyle="--", label="Max HR")
        plt.title(f"Heart Rate with {min_hr}-{max_hr} BPM Range Segments Highlighted")
        plt.xlabel("Beat Number")
        plt.ylabel("Heart Rate (BPM)")
        plt.legend()
        if plt_save:
            plt.savefig("rangeSegments.svg", format="svg", bbox_inches="tight")
        plt.show()
    return range_segments


def avg_channels_hr_range(
    data,
    peaks,
    heart_rates,
    window_size,
    denoise=False,
    plot=False,
    min_hr=0,
    max_hr=200,
    segment_length=5,
    plt_save=False,
    return_num_beats=False,
):
    if plot:
        plt.figure(figsize=[20, 10])

    avg = []
    std = []
    for ch in range(0, data.shape[1]):
        averaged_waveform, stds, num_beats = __avg_hr_range(
            data[:, ch],
            peaks,
            heart_rates,
            window_size=window_size,
            min_hr=min_hr,
            max_hr=max_hr,
            s_length=segment_length,
        )
        # if ch == 0:
        #     print(f"Averaged {num_beats} beats")
        if denoise:
            averaged_waveform = wavelet_denoise(averaged_waveform)
        if averaged_waveform is not None and plot:
            time = np.arange(len(averaged_waveform)) - window_size // 2
            plt.plot(time, averaged_waveform, color="k")
            plt.fill_between(
                time,
                averaged_waveform - stds,
                averaged_waveform + stds,
                color="gray",
                alpha=0.5,
            )
        avg.append(averaged_waveform)
        std.append(stds)
    if plot:
        plt.title(f"Averaged Waveform from Heart Rate Range {min_hr}-{max_hr} BPM")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        if plt_save:
            plt.savefig("averages.svg", format="svg", bbox_inches="tight")
        plt.show()

    if return_num_beats:
        return np.array(avg), np.array(std), num_beats
    return avg, std


def wavelet_denoise(averaged_waveform, wavelet="db4", level=5):
    coeffs = pywt.wavedec(averaged_waveform, wavelet, level=level)
    coeffs[1:] = [pywt.threshold(c, value=0.7, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)


def __avg_hr_range(
    data, peaks, heart_rates, window_size, min_hr=0, max_hr=200, s_length=5
):
    half_window = window_size // 2
    range_segments = identify_hr_range_segments(
        heart_rates, min_hr, max_hr, min_segment_length=s_length
    )

    summed_waveforms = []
    for start, end in range_segments:
        segment_peaks = peaks[start:end]
        for peak in segment_peaks:
            if peak - half_window < 0 or peak + half_window >= len(data):
                continue
            window = data[peak - half_window : peak + half_window]
            summed_waveforms.append(window)

    if not summed_waveforms:
        print(f"Warning: No valid segments found")
        return np.full(window_size-1, np.nan), np.full(window_size-1, np.nan), 0


    averaged_waveform = np.nanmean(summed_waveforms, axis=0)
    std = np.nanstd(summed_waveforms, axis=0)
    return averaged_waveform, std, len(summed_waveforms)
