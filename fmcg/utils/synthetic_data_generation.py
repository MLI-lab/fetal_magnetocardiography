from fmcg.utils import plot_utils, utils


import numpy as np
from scipy.signal import decimate

from fmcg.utils.data import load_vcg_data
from fmcg.utils.utils import rot, resample


def add_aligned_noise(
    field_data,
    time=None,
    noise_data=None,
    fs_field=None,
    fs_noise=None,
    noise_type="recording",
    snr_db=0,
    win_length_seconds=5,
    sine_frequency=4.0,  # For constant sine wave noise
    plot=False,
    seed=None,
):
    """
    Add aligned noise to a multichannel time-series field.

    Parameters
    ----------
    field_data : ndarray, shape (N, M, C)
        Clean field time-series data, where N is number of time samples, M is number of sensors,
        and C is number of components per sensor. May contain NaNs for missing data.
    time : ndarray, optional
        1D array of time instants in seconds corresponding to field_data. If None, it is
        generated as np.arange(0, N / fs_field, 1 / fs_field). Length should match the
        first dimension of field_data when provided.
    noise_data : ndarray, optional
        Recorded noise time-series to be added when noise_type == "recording". Must have
        shape (N_noise, M, C) (sensor/component dims must match field_data). If provided
        with fs_noise differing from fs_field it will be decimated by an integer factor.
    fs_field : float
        Sampling frequency (Hz) of field_data.
    fs_noise : float, optional
        Sampling frequency (Hz) of noise_data. Required when noise_type == "recording".
    noise_type : {"recording", "gaussian", "sine"}, default "recording"
        Type of noise to add:
        - "recording": use provided recorded noise_data (aligned/decimated to fs_field).
        - "gaussian": generate band-independent Gaussian noise via utils.add_gaussian_noise_to_field.
        - "sine": add a single-frequency sine wave (same across sensors/components).
    snr_db : float, default 0
        Desired signal-to-noise ratio in decibels (dB). The implementation converts this
        value to an amplitude scaling applied to the chosen noise before adding to field.
    win_length_seconds : float, default 5
        Window length in seconds used for local power estimation when computing scaling
        for the requested SNR. Converted internally to samples using fs_field.
    sine_frequency : float, default 4.0
        Frequency (Hz) of the sine wave when noise_type == "sine".
    plot : bool, default False
        If True, plot the original and noisy field signals using plot_utils.plot_sensor_signals.
    seed : int or None, optional
        Random seed forwarded to utils.add_gaussian_noise_to_field when noise_type == "gaussian".

    Returns
    -------
    noisy_field : ndarray
        Noisy version of field_data. For noise_type == "recording", the returned time-series
        is truncated to the minimum length of field_data and the (possibly decimated) noise_data.
        For other noise types the returned array has the same shape as field_data.

    Notes
    -----
    - For "recording", if fs_noise != fs_field the code attempts to decimate noise_data by
      factor int(fs_noise / fs_field). The caller should ensure fs_noise is an integer
      multiple of fs_field to avoid aliasing or incorrect decimation factors.
    - For "sine", a scalar sine waveform is created and broadcast to all sensors/components.
      The sine is scaled so that its power over the initial window (win_length_seconds)
      yields the requested SNR relative to the field power computed over the same window.
    - For "gaussian" and "recording" noise types, internal helper functions
      (utils.add_gaussian_noise_to_field, utils.add_noise_to_field) perform windowed SNR
      scaling and addition; behavior and parameters (e.g., handling of NaNs) follow those helpers.
    - If time is generated internally, its length will be floor(N) samples; ensure fs_field
      corresponds to the temporal sampling of field_data.

    """

    if noise_type not in ["recording", "gaussian", "sine"]:
        raise ValueError("noise_type must be one of 'recording', 'gaussian', or 'sine'")

    # Calculate window length in samples
    win = int(win_length_seconds * fs_field)

    if time is None:
        time = np.arange(0, field_data.shape[0] / fs_field, 1 / fs_field)

    if noise_type == "recording":
        if noise_data is None or fs_noise is None:
            raise ValueError(
                "For noise_type='recording', noise_data and fs_noise must be provided"
            )

        # Align sampling rates
        if fs_noise != fs_field:
            decimation_factor = int(fs_noise / fs_field)
            noise_data = decimate(noise_data, decimation_factor, axis=0)

        # Ensure both signals have compatible shapes for sensors and components
        if noise_data.shape[1:] != field_data.shape[1:]:
            raise ValueError(
                f"Sensor dimensions mismatch: noise {noise_data.shape[1:]} vs field {field_data.shape[1:]}"
            )

        # Truncate to match lengths
        min_length = min(field_data.shape[0], noise_data.shape[0])
        field_data_trunc = field_data[:min_length].copy()
        noise_data_trunc = noise_data[:min_length].copy()

        # Add noise
        noisy_field = utils.add_noise_to_field(
            field_data_trunc, noise_data_trunc, snr_db=snr_db, win=win
        )

    elif noise_type == "gaussian":
        # Use existing gaussian noise function
        noisy_field = utils.add_gaussian_noise_to_field(
            field_data, snr_db=snr_db, seed=seed, win=win
        )

    elif noise_type == "sine":
        # Create sine wave and reshape to match field dimensions
        noise = np.sin(2 * np.pi * sine_frequency * time).reshape(-1, 1, 1)

        # Scale noise based on SNR
        field_power = np.nanmean(field_data[:win] ** 2)
        noise_power = np.mean(noise[:win] ** 2)
        scaling = np.sqrt(field_power / noise_power) * 10 ** (-snr_db / 20)

        noisy_field = field_data + scaling * noise

    if plot:
        plot_utils.plot_sensor_signals(
            field_data,
            time,
            ylabel="Clean Field [pT]",
            xlim=[time[0], time[-1]],
            # savename="../plots/noise_gauss_SNR3dB.pdf",
        )

        plot_utils.plot_sensor_signals(
            noisy_field,
            time,
            ylabel="Noisy Field [pT]",
            xlim=[time[0], time[-1]],
            # savename="../plots/noise_gauss_SNR3dB.pdf",
        )

    return noisy_field


def add_noise_to_field(field, noise, snr_db=3, win=None):
    """
    Adds noise to the given field data to achieve a specified SNR in decibels (dB).

    Parameters
    ----------
    field : np.ndarray
        The true field data (signal).
    noise : np.ndarray
        The noise data (same shape as field).
    snr_db : float
        Desired SNR in decibels.

    Returns
    -------
    np.ndarray
        Field with noise added at the specified SNR.
    """
    # Calculate signal and noise power
    signal_power = utils.sig_power(field, win=win)
    noise_power = utils.sig_power(noise, win=win)
    print(f"Signal power: {signal_power}, Noise power: {noise_power}")

    # Desired noise power to achieve the target SNR
    target_noise_power = signal_power / (10 ** (snr_db / 10))

    # Scale the noise to get the desired noise power
    scale_factor = np.sqrt(target_noise_power / noise_power)

    return field + noise * scale_factor


def add_gaussian_noise_to_field(field, snr_db=3, win=None, noise_mean=0, seed=None):
    """
    Adds Gaussian noise to the given field data to achieve a specified SNR in decibels (dB).

    Parameters
    ----------
    field : np.ndarray
        The true field data (signal).
    snr_db : float
        Desired SNR in decibels.

    Returns
    -------
    np.ndarray
        Field with Gaussian noise added at the specified SNR.
    """

    signal_power = utils.sig_power(field, win=win)

    # Desired noise power to achieve the target SNR
    target_noise_power = signal_power / (10 ** (snr_db / 10))

    # Generate Gaussian noise
    rng = np.random.default_rng(seed)
    noise = rng.normal(noise_mean, np.sqrt(target_noise_power), field.shape)

    return field + noise


def generate_synthetic_dipole_signals(
    decimation_factor=2,
    plot=False,
    sampfrom=0,
    sampto=None,
    fetal_scaling=0.0005,  # .0001#.0002
    fetal_time_offset=0.5,
    fetal_downsample_factor=0.6,
    maternal_rot=[0, 0, 0],  # in deg
    fetal_rot=[180, 0, 0],
    fetal_pos=[[-0.02, -0.05, -0.02]],
    maternal_pos=[[-0.05, -0.05, 0.44]],
    vcg_dataset_path=None,
):
    """
    Generate synthetic dipole signals for fetal and maternal dipole sources.
    The magnetic dipole moments will be returned in uA m^2, the positions in m.

    Parameters:
        decimation_factor (int): Factor by which to downsample the original 1000Hz VCG signal.
        plot (bool): Whether to plot the VCG signals.
        sampfrom (float): Start time of the sample in seconds.
        sampto (float): End time of the sample in seconds.
        fetal_scaling (float): Scaling factor for the fetal signal.
        fetal_time_offset (float): Time offset for the fetal signal in seconds.
        fetal_downsample_factor (float): Downsample factor for the fetal signal.
        maternal_rot (list): Rotation angles for the maternal signal in degrees.
        fetal_rot (list): Rotation angles for the fetal signal in degrees.
        fetal_pos (list): Position of the fetal dipole.
        maternal_pos (list): Position of the maternal dipole.
        vcg_dataset_path (str): Path to the VCG dataset directory. Required parameter.
    Returns:
        dict: A dictionary containing the following keys:
        - m_true (ndarray): The true dipole moments.
        - r_true (ndarray): The true dipole positions.
        - fetal_signal (ndarray): The synthetic fetal signal.
        - maternal_signal (ndarray): The synthetic maternal signal.
        - fetal_ecg (ndarray): The resampled fetal ECG signal.
        - maternal_ecg (ndarray): The resampled maternal ECG signal.
        - time (ndarray): The time array for the selected segment.
        - fs (float): The sampling frequency after decimation.
        - num_dipoles (int): The number of dipoles.
    """
    num_dipoles = 2

    ## Load VCG data
    filtered_signals, ecg_II, fs = load_vcg_data(base_path=vcg_dataset_path)

    # downsample VCG signal
    filtered_signals = decimate(filtered_signals, decimation_factor, axis=0)
    ecg_II = decimate(ecg_II, decimation_factor, axis=0)
    fs /= decimation_factor

    # Convert time values to indices
    sampfrom_idx = int(sampfrom * fs)
    fetal_offset_idx = int(fetal_time_offset * fs)
    if sampto is None:
        sampto_idx = filtered_signals.shape[0]
    else:
        sampto_idx = int(sampto * fs)

    if plot:
        plot_utils.plot_vcg(filtered_signals, ecg_II, fs, plt_HS_coor=False)

    ## Create dipole model
    # convert frank leads to heart shield ccordinates
    filtered_signals = np.array(
        [-filtered_signals[:, 0], -filtered_signals[:, 2], -filtered_signals[:, 1]]
    ).T

    # Extract maternal signal from selected time range
    maternal_signal = filtered_signals[sampfrom_idx:sampto_idx]
    maternal_ecg = ecg_II[sampfrom_idx:sampto_idx]

    # Calculate fetal signal start index (with both time offset and additional offset parameter)
    fetal_start_idx = sampfrom_idx + fetal_offset_idx

    # Calculate how many samples of the original fetal signal we need to maintain the
    # correct length after resampling
    original_fetal_samples_needed = int(len(maternal_signal) / fetal_downsample_factor)

    # Extract fetal signal with adjusted start position
    fetal_signal = filtered_signals[
        fetal_start_idx : fetal_start_idx + original_fetal_samples_needed
    ]
    fetal_ecg = ecg_II[
        fetal_start_idx : fetal_start_idx + original_fetal_samples_needed
    ]

    # Resample the fetal signal to match desired length after resampling
    fetal_signal = resample(fetal_scaling * fetal_signal, len(maternal_signal), axis=0)
    fetal_ecg = resample(fetal_ecg, len(maternal_signal), axis=0)

    # Create appropriate time array for the selected segment
    time = np.arange(sampfrom, sampto, 1 / fs)

    # rotate the cardiac vectors
    fetal_signal = rot(fetal_signal, rot=fetal_rot)
    maternal_signal = rot(maternal_signal, rot=maternal_rot)

    # construct the output
    m_true = np.array([fetal_signal, maternal_signal]).transpose(1, 0, 2)[
        :, :num_dipoles
    ]
    r_true = np.array([fetal_pos, maternal_pos]).transpose(1, 0, 2)[:, :num_dipoles]

    if plot:
        plot_utils.plot_vcg(maternal_signal, maternal_ecg, fs, sig_name="mMCG")
        plot_utils.plot_vcg(
            fetal_signal / fetal_scaling,
            fetal_ecg,
            fs=fs,
            fetal_downsample_factor=fetal_downsample_factor,
            sig_name="fMCG",
        )

    return dict(
        m_true=m_true,
        r_true=r_true,
        fetal_signal=fetal_signal,
        maternal_signal=maternal_signal,
        fetal_ecg=fetal_ecg,
        maternal_ecg=maternal_ecg,
        time=time,
        fs=fs,
        num_dipoles=num_dipoles,
    )