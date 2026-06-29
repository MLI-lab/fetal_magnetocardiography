import numpy as np

from fmcg.analysis.hr import compute_hr
from fmcg.analysis.heartbeat_averaging import average_heartbeats
from fmcg.utils.plotting.plot_utils import plot_sensor_signals
from fmcg.utils.utils import reduced2full


def construct_template(data, sources, fs, comps, window_size=0.6, bpm_tol=5,
                      plot=False, axis_mask=None, subtract_mean=True,
                      t_start=None, t_end=None):
    """
    Construct spatial template from cardiac beat averaging.

    Detects cardiac cycles from source components, averages field measurements
    over beats with similar heart rates, and returns the spatiotemporal template.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_channels)
        Sensor measurements to construct template from
    sources : ndarray, shape (n_samples, n_components)
        Source signals (e.g., from ICA) for heart rate detection
    fs : float
        Sampling frequency in Hz
    comps : list of int
        Component indices to use for heart rate detection
    window_size : float, optional
        Window size as fraction of cardiac cycle (default: 0.6)
    bpm_tol : float, optional
        BPM tolerance for selecting similar beats (default: 5)
    plot : bool, optional
        Whether to plot the template (default: False)
    axis_mask : ndarray of bool, optional
        Mask for valid sensor locations, used for plotting
    subtract_mean : bool, optional
        Whether to subtract temporal mean from template (default: True)
    t_start : float, optional
        Start time in seconds (default: None, uses beginning)
    t_end : float, optional
        End time in seconds (default: None, uses end)

    Returns
    -------
    topography : ndarray, shape (n_timepoints, n_channels)
        Spatial template from averaged cardiac beats
    """
    assert data.ndim == 2, "Data must be 2D (n_samples, n_channels)"

    # Convert time indices to sample indices
    t_start = 0 if t_start is None else int(t_start * fs)
    t_end = data.shape[0] if t_end is None else int(t_end * fs)

    # Detect heart rate and peaks from sources
    heartRate, peaks = compute_hr(sources[t_start:t_end][:, comps], fs, plot=False)

    # Average segments across similar heartbeats
    topography, _, _ = average_heartbeats(peaks, data[t_start:t_end], fs, interval=window_size)

    if subtract_mean:
        topography -= topography.mean(axis=0)

    # Plot if requested
    if plot:
        if axis_mask is None:
            raise ValueError("To plot the topography, please provide an axis_mask "
                           "indicating the sensor locations.")
        full_topography = reduced2full(topography, axis_mask)
        plot_sensor_signals(
            full_topography,
            np.arange(full_topography.shape[0]) / fs,
            ylabel="Reconstructed Field [pT]",
            xlim=[0, full_topography.shape[0] / fs],
            sharey=True,
        )

    return topography