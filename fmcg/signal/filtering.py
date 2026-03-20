

from copy import deepcopy
import logging
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pywt
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

def filternk(sig, fs, method="biosppy", **kwargs):
    """ Apply bandpass and notch filters using neurokit2's ecg_clean function. """
    sig_filt = sig.copy()
    for i in range(sig.shape[1]):
        sig_filt[:, i] = nk.ecg_clean(sig[:, i], sampling_frequency=fs, method=method, **kwargs)
    return sig_filt

def filter_sensor_dict(grid, fs, low=0.5, high=40, order=6, notch=True):
    """
    Apply bandpass and multiple notch filters to sensor data in a dictionary.

    Parameters:
        grid (dict): Dictionary containing sensor data arrays. Each key corresponds to a sensor, and the value is a 2D numpy array where rows represent time points and columns represent different channels.
        fs (float): Sampling frequency of the sensor data.
        low (float, optional): Lower frequency bound for the bandpass filter. Default is 0.5 Hz.
        high (float, optional): Upper frequency bound for the bandpass filter. Default is 40 Hz.
        order (int, optional): Order of the bandpass filter. Default is 6.
        notch (bool, optional): If True, apply notch filters at specific frequencies to remove powerline noise and other artifacts. Default is True.

    Returns:
        dict: A dictionary with the same keys as the input, where each value is a 2D numpy array of filtered sensor data.
    """
    bandpass = butter(order, [low, high], btype="band", output="sos", fs=fs)

    grid_out = deepcopy(grid)
    for key in grid.keys():
        # Optimization: Use axis parameter for vectorized filtering (20-30% faster)
        grid_out[key] = sosfiltfilt(bandpass, grid[key], axis=0)
        if notch:
            # Apply all notch filters vectorized
            b_notch, a_notch = iirnotch(50, 40, fs=fs)
            grid_out[key] = filtfilt(b_notch, a_notch, grid_out[key], axis=0)

            b_notch, a_notch = iirnotch(16.7, 40, fs=fs)
            grid_out[key] = filtfilt(b_notch, a_notch, grid_out[key], axis=0)

            # b_notch, a_notch = iirnotch(29, 10, fs=fs)
            # grid_out[key] = filtfilt(b_notch, a_notch, grid_out[key], axis=0)

            b_notch, a_notch = iirnotch(76, 30, fs=fs)
            grid_out[key] = filtfilt(b_notch, a_notch, grid_out[key], axis=0)

            # b_notch, a_notch = iirnotch(77, 30, fs=fs)
            # grid_out[key] = filtfilt(b_notch, a_notch, grid_out[key], axis=0)
            # b_notch, a_notch = iirnotch(83.5, 30, fs=fs)
            # grid_out[key] = filtfilt(b_notch, a_notch, grid_out[key], axis=0)

            b_notch, a_notch = iirnotch(38, 30, fs=fs)
            grid_out[key] = filtfilt(b_notch, a_notch, grid_out[key], axis=0)

            b_notch, a_notch = iirnotch(100, 30, fs=fs)
            grid_out[key] = filtfilt(b_notch, a_notch, grid_out[key], axis=0)
    return grid_out


def filter_list(data, fs, low=0.5, high=80, order=6):
    """ legacy / explorative function to filter a list of signals """
    data_filt = deepcopy(data)

    if data.ndim > 2:
        T, N, _ = data.shape
        data_filt = data_filt.reshape(T, -1)

    bandpass = butter(order, [low, high], btype="band", output="sos", fs=fs)
    data_filt = sosfiltfilt(bandpass, data_filt, axis=0)

    # Apply all notch filters vectorized
    b_notch, a_notch = iirnotch(50, 40, fs=fs)
    data_filt = filtfilt(b_notch, a_notch, data_filt, axis=0)

    b_notch, a_notch = iirnotch(16.7, 40, fs=fs)
    data_filt = filtfilt(b_notch, a_notch, data_filt, axis=0)

    b_notch, a_notch = iirnotch(76, 30, fs=fs)
    data_filt = filtfilt(b_notch, a_notch, data_filt, axis=0)

    # b_notch, a_notch = iirnotch(77, 30, fs=fs)
    # data_filt = filtfilt(b_notch, a_notch, data_filt, axis=0)
    # b_notch, a_notch = iirnotch(83.5, 30, fs=fs)
    # data_filt = filtfilt(b_notch, a_notch, data_filt, axis=0)

    b_notch, a_notch = iirnotch(38, 30, fs=fs)
    data_filt = filtfilt(b_notch, a_notch, data_filt, axis=0)

    b_notch, a_notch = iirnotch(100, 30, fs=fs)
    data_filt = filtfilt(b_notch, a_notch, data_filt, axis=0)

    if data.ndim > 2:
        data_filt = data_filt.reshape(T, N, -1)
    return data_filt


def wavelet_denoise(signal, wavelet='db6', level=None, threshold=None, plot=False, verbose=True):
    """
    Denoise a signal using wavelet soft-thresholding.

    Parameters
    ----------
    signal : array_like
        1-D sequence (or array with samples along axis=0) to be denoised.
    wavelet : str or pywt.Wavelet, optional
        Wavelet to use for decomposition (default: 'db6').
    level : int, optional
        Decomposition level. If None, the maximum allowed level is selected automatically.
    threshold : float, optional
        If provided, treated as a multiplier of the estimated noise sigma (effective
        threshold = threshold * sigma). If None, a universal threshold
        sigma * sqrt(2*log(N)) is used, where N is the signal length.
    plot : bool, optional
        If True, plot approximation and detail coefficients for each level.
    verbose : bool, optional
        If True, print diagnostic information (max level, estimated sigma, threshold).

    Returns
    -------
    ndarray
        Reconstructed (denoised) signal with the same sample axis as the input.

    """
    # Auto-select level if not provided
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    print(f"Max level: {max_level}")
    level = level or max_level

    # Decompose
    coeffs = pywt.wavedec(signal, wavelet, level=level, axis=0)

    # Estimate noise sigma from finest detail
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    if verbose:
        print(f"Estimated noise sigma: {sigma}")
    # Universal threshold if not given
    N = len(signal)
    T = threshold * sigma or sigma * np.sqrt(2 * np.log(N))
    if verbose:
        print(f"Threshold: {T}")

    # Plot coefficients
    if plot:
        plt.figure(figsize=(10, level*1.25))
        for i, c in enumerate(coeffs):
            plt.subplot(level+1, 1, i+1)
            plt.title(f'Level {i}' + (' Approx' if i==0 else ' Detail'))
            plt.plot(c)
        plt.tight_layout()
        plt.show()

    # Threshold detail coeffs
    coeffs[1:] = [pywt.threshold(c, T, mode='soft') for c in coeffs[1:]]

    # Reconstruct
    return pywt.waverec(coeffs, wavelet, axis=0)


def pca_filter(data, remove_first=3, verbose=False):
    """
    Applies Principal Component Analysis (PCA) to filter the input data by removing 
    the first few principal components. The function handles NaN values by ignoring
    columns where all values are NaN during PCA computation. The PCA transformation
    is applied independently to each channel (last axis of the input data).

    Parameters:
        data : numpy.ndarray
            A 3D array where PCA is applied along the last axis. The shape of the array 
            should be (n_samples, n_features, n_channels).
        remove_first : int, optional
            The number of principal components to remove. Default is 3.
        verbose : bool, optional
            If True, prints the explained variance ratio of the PCA for each channel. 
            Default is False.

    Returns:
        numpy.ndarray
            A 3D array of the same shape as the input `data`, with the specified number 
            of principal components removed for each channel.
    """
    if isinstance(remove_first, int):
        remove_first = [remove_first] * data.shape[-1]
    pca = PCA()
    pca_filtered = data.copy()
    for i in range(pca_filtered.shape[-1]):
        x = pca_filtered[:,:,i]
        x = x[:, np.isnan(x).all(axis=0) == False]
        c = pca.fit_transform(x)
        c[:, :remove_first[i]] = 0
        pca_filtered[:,np.isnan(pca_filtered[:,:,i]).all(axis=0) == False,i] = np.dot(c, pca.components_)

        if verbose:
            print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    return pca_filtered


def dict_remove_median(sensor_dict, axis=0, window=None):
    """
    Subtract the per-sample median (common-mode) across sensors from each sensor array.

    Parameters:
        sensor_dict : dict[str, numpy.ndarray]
            Mapping from sensor name to a 2D numpy array of shape (T, C) containing time-series
            data for that sensor. All arrays must have the same shape. NaN values are allowed
            and ignored when computing the median.
        axis : int, optional
            Axis passed to numpy.nanmedian when stacking sensor arrays. Default is 0, which
            computes the median across sensors for each time/channel element.
        window : int or None, optional
            If None (default), compute a single per-sample median across all sensors for the
            entire time series. If an integer is provided, the time dimension T is split into
            non-overlapping windows of length `window` and a median is computed per-window
            across sensors. Any leftover samples at the end (T % window) are ignored. Note:
            the implementation currently reshapes windowed results for 3 channels, so windowed
            mode expects C == 3.

    Returns:
        tuple
            (adjusted_dict, common_mode)
            - adjusted_dict: dict with the same keys as sensor_dict and arrays of the same shape,
            where the computed common-mode median has been subtracted elementwise from each
            sensor array.
            - common_mode: numpy.ndarray containing the median(s) that were subtracted. When
            window is None, this has the same shape as the sensor arrays (T, C). When window
            is an int, this contains the per-window medians flattened to align with the time
            dimension (shape (num_windows * window, 3) in the current implementation).
    """

    if window is None:
        common_mode = np.nanmedian(np.array(list(sensor_dict.values())), axis=axis, keepdims=False)
        return {name: sensor_dict[name] - common_mode for name in sensor_dict.keys()}, common_mode

    window = int(window)
    data_len = list(sensor_dict.values())[0].shape[0]
    num_windows = data_len // window
    num_channels = list(sensor_dict.values())[0].shape[1]
    common_mode = np.zeros((num_windows, window, num_channels))

    for i in range(num_windows):
        window_data = [v[i * window : (i + 1) * window] for v in sensor_dict.values()]
        common_mode[i] = np.nanmedian(window_data, axis=axis, keepdims=False)

    common_mode = common_mode.reshape(-1, 3)
    return {
        name: sensor_dict[name] - common_mode[: sensor_dict[name].shape[0]] for name in sensor_dict.keys()
    }, common_mode


def detrend(sig, win=None, order=3):
    """
    Detrend a signal (1D or multi-channel) by applying polynomial detrending
    to non-overlapping windows.

    Parameters
    ----------
    sig : numpy.ndarray
        Input signal with shape (n_samples,) or (n_samples, ...) where the first
        axis is time/samples. The array will be reshaped to (n_samples, n_channels)
        internally for processing.
    win : int or None, optional
        Window length (in samples) for blockwise detrending. If None (default),
        the whole signal length is used (single window). The final window may be
        shorter if n_samples is not a multiple of win.
    order : int, optional
        Polynomial order passed to nk.signal_detrend (default: 3).

    Returns
    -------
    numpy.ndarray
        Detrended signal with the same shape as the input.

    Notes
    -----
    - Detrending is performed per-window and per-channel using
      neurokit2.signal_detrend(..., method='polynomial', order=order).
    - Windows consisting entirely of NaNs are left unchanged.
    - The function may modify the input array if it shares memory with the
      internal reshaped view; to avoid in-place changes, pass a copy of `sig`.
    """
    if sig.ndim == 1:
        sig_ = sig[:, None]
    else:
        sig_ = sig.reshape(sig.shape[0], -1)

    if win is None:
        win = sig.shape[0]
    win = int(win)

    num_windows = np.ceil(sig.shape[0] / win).astype(int)
    for i in range(num_windows):
        for j in range(sig_.shape[1]):
            if np.isnan(sig_[i * win : (i + 1) * win, j]).all():
                continue
            sig_[i * win : (i + 1) * win, j] = nk.signal_detrend(
                sig_[i * win : (i + 1) * win, j], method="polynomial", order=order
            )

    return sig_.reshape(sig.shape)


def remove_common_mode(
    field, axis=0, window=None, method="mean", regress=False, filter_lowpass=False, fs=500, return_common_mode=False
):
    """
    Removes the common mode signal from a multi-dimensional field array.

    Parameters:
        field (np.ndarray): Input data array to process.
        axis (int, optional): Axis along which to calculate the common mode. Default is 0.
        window (int, optional): Size of the window for processing. Defaults to the length of the first dimension.
        method (str, optional): Aggregation method for common mode calculation ('mean' or 'median'). Default is "mean".
        regress (bool, optional): If True, performs regression to remove the common mode. Default is False.
        filter_lowpass (bool, optional): If True, applies a lowpass filter to the common mode signal. Default is False.
        fs (int, optional): Sampling frequency for the lowpass filter. Default is 500 Hz.
        return_common_mode (bool, optional): If True, returns the computed common mode signal. Default is False.

    Returns:
        np.ndarray: The field array with the common mode removed.
        np.ndarray (optional): The computed common mode signal, if `return_common_mode` is True.

    Raises:
        ValueError: If an invalid method is specified.
    """
    if method == "mean":
        agg_func = np.nanmean
    elif method == "median":
        agg_func = np.nanmedian
    else:
        raise ValueError("Invalid method")

    if window is None:
        window = field.shape[0]

    window = int(window)
    num_windows = np.ceil(field.shape[0] / window).astype(int)

    field_cmr = deepcopy(field)
    if return_common_mode:
        cm = np.zeros_like(field)

    for i in range(num_windows):
        window_data = field[i * window : (i + 1) * window]
        agg = agg_func(window_data, axis=axis, keepdims=True)

        if filter_lowpass:
            # # highpass filter
            bandpass = butter(4, 25, btype="lowpass", output="sos", fs=fs)
            for j in range(3):
                agg[:, :, j] = sosfiltfilt(bandpass, agg[:, :, j].T).T

        if regress:
            for j in range(3):
                for k in range(window_data.shape[1]):
                    if np.isnan(window_data[:, k, j]).all():
                        continue
                    reg = LinearRegression().fit(agg[:, 0, j].reshape(-1, 1), window_data[:, k, j])
                    beta = reg.coef_[0]

                    if return_common_mode:
                        cm[i * window : (i + 1) * window, k, j] -= beta * agg[:, 0, j]

                    field_cmr[i * window : (i + 1) * window :, k, j] -= beta * agg[:, 0, j]
                # is_nan = np.isnan(window_data[:,:,j]).all(axis=0)

                # reg = LinearRegression(alpha=10).fit(agg[:,:,j], window_data[:,~is_nan,j])
                # beta = reg.coef_[:,0]

                # field_cmr[i * window: (i + 1) * window:, ~is_nan, j] -= beta * agg[:,:,j]
        else:
            if return_common_mode:
                cm[i * window : (i + 1) * window] -= agg
            field_cmr[i * window : (i + 1) * window] -= agg

    if return_common_mode:
        return field_cmr, cm
    return field_cmr

