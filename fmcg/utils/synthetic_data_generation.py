import numpy as np
from scipy.signal import decimate, resample

from fmcg.data import load_vcg_data
from fmcg.utils import utils
from fmcg.utils.utils import rot
from fmcg.fitting.field_model import ForwardModel
from fmcg.utils.plotting import plot_utils


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

        # If noise is shorter than field, tile it to cover the full length.
        # If field is shorter, truncate noise to match.
        n_field = field_data.shape[0]
        n_noise = noise_data.shape[0]
        if n_noise < n_field:
            repeats = (n_field // n_noise) + 1
            noise_data = np.tile(noise_data, (repeats, 1, 1))[:n_field]
        field_data_trunc = field_data.copy()
        noise_data_trunc = noise_data[:n_field].copy()

        # Add noise
        noisy_field = add_noise_to_field(
            field_data_trunc, noise_data_trunc, snr_db=snr_db, win=win
        )

    elif noise_type == "gaussian":
        # Use existing gaussian noise function
        noisy_field = add_gaussian_noise_to_field(
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


def sample_physiological_position(
    baseline,
    bnds,
    sigma=0.05,
    seed=None,
    constraints=None,
):
    """
    Sample a single physiological position using Gaussian perturbation with rejection sampling.

    Parameters
    ----------
    baseline : array (3,)
        Baseline position [x, y, z] in meters
    bnds : array (3, 2)
        Bounds for each coordinate [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    sigma : float
        Gaussian standard deviation for perturbations (meters)
    seed : int or None
        Random seed
    constraints : dict or None
        Additional constraints to check. Keys: 'min_z', 'max_z', 'other_position' (for separation)

    Returns
    -------
    position : array (3,)
        Sampled position satisfying all constraints
    """
    rng = np.random.default_rng(seed)
    max_attempts = 1000

    for _ in range(max_attempts):
        # Sample perturbation
        pos = baseline + rng.normal(0, sigma, size=3)

        # Check bounds
        within_bounds = all(
            bnds[i, 0] <= pos[i] <= bnds[i, 1] for i in range(3)
        )
        if not within_bounds:
            continue

        # Check additional constraints if provided
        if constraints:
            if 'min_z' in constraints and pos[2] < constraints['min_z']:
                continue
            if 'max_z' in constraints and pos[2] > constraints['max_z']:
                continue
            if 'other_position' in constraints:
                other = constraints['other_position']
                separation = np.linalg.norm(pos - other)
                min_sep = constraints.get('min_separation', 0.05)
                if separation < min_sep:
                    continue

        return pos

    # If rejection sampling fails, return baseline
    return baseline


def generate_dipole_movement(
    r_init,
    bnds,
    n_samples,
    movement_type='random_walk',
    step_size=0.0001,
    tau=2000,
    momentum=0.95,
    seed=None,
    fs=1000,
    lowpass_cutoff=None,
):
    """
    Generate time-varying dipole positions with physiological constraints.

    Parameters
    ----------
    r_init : array (3,) or (n_dipoles, 3)
        Initial position(s) in meters
    bnds : array (3, 2) or (n_dipoles, 3, 2)
        Position bounds
    n_samples : int
        Number of time samples
    movement_type : str
        - 'static': No movement (constant position)
        - 'random_walk': Brownian-like random walk with reflecting boundaries
        - 'constrained_random_walk': Smoother random walk with temporal correlation
        - 'ou_process': Ornstein-Uhlenbeck overdamped process (realistic fetal movement)
    step_size : float
        Standard deviation of position changes per time step (meters)
        For 'ou_process': controls diffusion strength (sigma). Typical: 0.02 for ~2cm/s drift
    tau : int or float
        For 'constrained_random_walk': temporal correlation length (samples)
        For 'ou_process': time constant in seconds (how long a drift lasts). Typical: 2.0 seconds
    momentum : float
        Velocity momentum parameter (0-1) for 'constrained_random_walk'.
        Higher values preserve velocity direction. Default 0.95 gives physiological motion.
    seed : int or None
        Random seed for reproducibility
    fs : float
        Sampling frequency in Hz (used for 'ou_process' to compute dt). Default 1000 Hz.
    lowpass_cutoff : float or None
        If provided, apply Butterworth lowpass filter with this cutoff frequency (Hz).
        Typical: 5 Hz to smooth out high-frequency jitter while preserving slow drift.
        Only applies to 'ou_process' and 'constrained_random_walk'.

    Returns
    -------
    r_trajectory : array (n_samples, n_dipoles, 3)
        Time-varying positions
    """
    rng = np.random.default_rng(seed)

    # Handle single dipole case
    if r_init.ndim == 1:
        r_init = r_init.reshape(1, 3)
        bnds = bnds.reshape(1, 3, 2)

    n_dipoles = r_init.shape[0]
    r_trajectory = np.zeros((n_samples, n_dipoles, 3))
    r_trajectory[0] = r_init

    if movement_type == 'static':
        # No movement - replicate initial position
        for t in range(1, n_samples):
            r_trajectory[t] = r_init
        return r_trajectory

    elif movement_type == 'random_walk':
        # Simple random walk with reflecting boundaries
        for t in range(1, n_samples):
            # Random step for each dipole
            steps = rng.normal(0, step_size, size=(n_dipoles, 3))
            new_pos = r_trajectory[t - 1] + steps

            # Reflect at boundaries
            for d in range(n_dipoles):
                for i in range(3):
                    if new_pos[d, i] < bnds[d, i, 0]:
                        new_pos[d, i] = 2 * bnds[d, i, 0] - new_pos[d, i]
                    elif new_pos[d, i] > bnds[d, i, 1]:
                        new_pos[d, i] = 2 * bnds[d, i, 1] - new_pos[d, i]

            r_trajectory[t] = new_pos

    elif movement_type == 'constrained_random_walk':
        # Smoother random walk with temporal correlation (Ornstein-Uhlenbeck-like) + momentum
        velocity = np.zeros((n_dipoles, 3))

        for t in range(1, n_samples):
            # Update velocity with momentum, decay, and random forcing
            random_force = rng.normal(0, step_size, size=(n_dipoles, 3))
            velocity = momentum * velocity + (1 - 1/tau) * random_force
            new_pos = r_trajectory[t - 1] + velocity

            # Reflect at boundaries and reverse velocity
            for d in range(n_dipoles):
                for i in range(3):
                    if new_pos[d, i] < bnds[d, i, 0]:
                        new_pos[d, i] = 2 * bnds[d, i, 0] - new_pos[d, i]
                        velocity[d, i] *= -0.5  # Reverse and dampen
                    elif new_pos[d, i] > bnds[d, i, 1]:
                        new_pos[d, i] = 2 * bnds[d, i, 1] - new_pos[d, i]
                        velocity[d, i] *= -0.5

            r_trajectory[t] = new_pos

    elif movement_type == 'ou_process':
        # Ornstein-Uhlenbeck process for overdamped fetal movement in amniotic fluid
        # This simulates the high-viscosity environment where movement stops quickly
        # after muscle contraction, creating smooth ~2 cm/s drifts without momentum
        
        dt = 1.0 / fs  # Time step in seconds
        theta = 1.0 / tau  # Stiffness parameter (mean-reversion rate)
        sigma = step_size  # Diffusion strength (controls drift speed)
        mu = r_init.copy()  # Rest/central position (starting position)
        
        for t in range(1, n_samples):
            # 1. Calculate drift force (mean-reverting to rest position)
            drift = theta * (mu - r_trajectory[t - 1]) * dt
            
            # 2. Calculate diffusion (random walk component)
            diffusion = sigma * np.sqrt(dt) * rng.normal(size=(n_dipoles, 3))
            
            # 3. Update position
            new_pos = r_trajectory[t - 1] + drift + diffusion
            
            # 4. Apply soft boundary constraints (sticky walls, not bouncing)
            # This feels more like a womb than hard reflections
            for d in range(n_dipoles):
                for i in range(3):
                    # Clip to boundaries - creates natural "stalling" at limits
                    new_pos[d, i] = np.clip(new_pos[d, i], bnds[d, i, 0], bnds[d, i, 1])
            
            r_trajectory[t] = new_pos

    else:
        raise ValueError(
            f"Unknown movement_type '{movement_type}'. "
            f"Choose from: 'static', 'random_walk', 'constrained_random_walk', 'ou_process'"
        )

    # Apply low-pass filter if requested (removes high-frequency jitter)
    if lowpass_cutoff is not None and movement_type != 'static':
        from scipy.signal import butter, filtfilt
        
        # Design Butterworth filter
        nyquist = fs / 2
        normalized_cutoff = lowpass_cutoff / nyquist
        b, a = butter(4, normalized_cutoff, btype='low')
        
        # Apply filter to each dipole and coordinate independently
        for d in range(n_dipoles):
            for i in range(3):
                r_trajectory[:, d, i] = filtfilt(b, a, r_trajectory[:, d, i])
        
        # Re-apply boundary constraints after filtering (filter may violate bounds)
        for d in range(n_dipoles):
            for i in range(3):
                r_trajectory[:, d, i] = np.clip(r_trajectory[:, d, i], bnds[d, i, 0], bnds[d, i, 1])

    return r_trajectory


def apply_vcg_drift(m_source, peaks, drift, drift_scale=1.0):
    """Replay a measured VCG drift trajectory onto a clean 3D cardiac moment source.

    The full measured drift (orientation ``units`` + relative magnitude ``scale`` w.r.t. ``mean_dir``;
    produced by :func:`fmcg.analysis.vcg.vcg_drift` on a real recording) is **resampled** to the
    number of beats in ``peaks``, then applied beat-wise as the minimal rotation
    ``mean_dir -> units[i]`` (geodesically scaled by ``drift_scale``) and amplitude scaling
    ``1 + drift_scale*(scale[i]-1)``. Rotations are slerp-interpolated and scales linearly
    interpolated to every sample for smooth per-sample drift.

    Parameters
    ----------
    m_source : (T, 3) clean cardiac moment trajectory (before drift).
    peaks    : (n_beats,) synthetic R-peak sample indices.
    drift    : dict from ``vcg_drift`` — uses ``units``, ``scale``, ``mean_dir``.
    drift_scale : float — 0 = stationary, 1 = real magnitude, k = exaggerated.

    Returns
    -------
    (T, 3) drifted moment trajectory.
    """
    from scipy.spatial.transform import Rotation, Slerp
    from fmcg.analysis.vcg import axis_rotation

    m_source = np.asarray(m_source, dtype=float)
    peaks = np.asarray(peaks)
    T = m_source.shape[0]
    if len(peaks) < 2 or drift_scale == 0:
        return m_source.copy()

    # real per-beat drift rotations (mean_dir -> peak orientation)
    units_r, scale_r, mean_dir = drift["units"], drift["scale"], drift["mean_dir"]
    nr, n = len(units_r), len(peaks)
    real_rots = Rotation.from_matrix([axis_rotation(mean_dir, u) for u in units_r])

    # slerp-resample the full real trajectory -> n synthetic beats, then geodesically scale
    resamp = Slerp(np.arange(nr), real_rots)(np.linspace(0.0, nr - 1, n))
    beat_rots = Rotation.from_rotvec(drift_scale * resamp.as_rotvec())
    scale_n = np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(scale_r)), scale_r)
    beat_scales = 1.0 + drift_scale * (scale_n - 1.0)

    # interpolate to every sample (clamp outside the [first, last] peak range)
    idx = np.clip(np.arange(T), peaks[0], peaks[-1])
    R_t = Slerp(peaks, beat_rots)(idx)
    s_t = np.interp(idx, peaks, beat_scales)
    return R_t.apply(m_source) * s_t[:, None]


def generate_synthetic_fmcg_recording(
    r_sensors,
    # VCG source configuration
    vcg_fetal_record="patient104/s0306lre",
    vcg_maternal_record="patient104/s0306lre",
    vcg_dataset_path=None,
    # Position sampling
    fetal_pos_baseline=np.array([0.0, -0.05, 0.0]),
    maternal_pos_baseline=np.array([0.0, -0.05, 0.15]),
    fetal_pos_bounds=np.array([[-0.05, 0.15], [-0.05, 0.0], [-0.05, 0.05]]),
    maternal_pos_bounds=np.array([[-0.1, 0.15], [-0.1, 0.0], [0.1, 0.3]]),
    position_sampling_sigma=0.02,
    min_separation=0.05,
    # Dipole movement
    enable_movement=False,
    movement_type='constrained_random_walk',
    movement_step_size=0.00001,
    movement_tau=2000,
    movement_momentum=0.98,
    movement_lowpass_cutoff=None,
    # VCG drift (replay measured real drift; see fmcg.analysis.vcg.vcg_drift)
    vcg_drift=None,
    drift_scale=1.0,
    # VCG processing
    fetal_moment_amplitude_nAm2=60.0,
    maternal_moment_amplitude_nAm2=6000.0,
    moment_amplitude_variation=0.2,
    fetal_time_offset=0.5,
    fetal_downsample_factor=0.6,
    fetal_rot=None,
    maternal_rot=None,
    fetal_rot_bounds=np.array([[-180.0, 180.0], [-180.0, 180.0], [-180.0, 180.0]]),
    maternal_rot_bounds=np.array([[-180.0, 180.0], [-180.0, 180.0], [-180.0, 180.0]]),
    decimation_factor=2,
    sampfrom=0,
    sampto=None,
    # Noise configuration
    snr_db=10,  # 10 dB SNR (less aggressive noise than 3 dB)
    noise_type='gaussian',
    noise_data=None,
    fs_noise=None,
    gaussian_noise_fraction=None,  # If set, adds Gaussian noise = fraction * RMS(field_clean) on top of snr_db noise
    # Sensor axis mask
    axis_mask=None,
    # Output
    seed=None,
    return_clean=False,
    return_sources=False,
    device="cpu",
):
    """
    Generate a complete synthetic fMCG recording with physiological constraints.
    
    This function combines VCG-based cardiac signals, physiological position sampling,
    optional dipole movement, forward modeling, and noise addition to create realistic
    synthetic fMCG data for benchmarking reconstruction algorithms.

    Parameters
    ----------
    r_sensors : array (n_sensors, 3)
        Sensor array positions in meters
    vcg_fetal_record : str
        PTB database record for fetal VCG (e.g., "patient104/s0306lre")
    vcg_maternal_record : str
        PTB database record for maternal VCG
    vcg_dataset_path : str or None
        Path to PTB VCG database (if None, loads from PhysioNet)
    fetal_pos_baseline : array (3,)
        Baseline fetal position [x, y, z] in meters
    maternal_pos_baseline : array (3,)
        Baseline maternal position in meters
    fetal_pos_bounds : array (3, 2)
        Fetal position bounds [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    maternal_pos_bounds : array (3, 2)
        Maternal position bounds
    position_sampling_sigma : float
        Gaussian std for position perturbation (meters)
    min_separation : float
        Minimum fetal-maternal separation (meters)
    enable_movement : bool
        Whether to simulate time-varying dipole positions
    movement_type : str
        Type of movement ('static', 'random_walk', 'constrained_random_walk', 'ou_process')
    movement_step_size : float
        Movement step size per time sample (meters). For 'ou_process', controls diffusion (sigma).
    movement_tau : int or float
        For 'constrained_random_walk': temporal correlation length (samples).
        For 'ou_process': time constant in seconds.
    movement_momentum : float
        Velocity momentum for 'constrained_random_walk' (0-1, higher=smoother)
    movement_lowpass_cutoff : float or None
        If provided, apply Butterworth lowpass filter at this frequency (Hz) to smooth trajectory.
        Typical: 5 Hz removes jitter while preserving slow drift. Only for moving dipoles.
    vcg_drift : dict or None, default None
        If provided, replays a measured real VCG drift trajectory onto the fetal cardiac vector
        (orientation + magnitude nonstationarity). Produce it with
        :func:`fmcg.analysis.vcg.vcg_drift` on a real recording. See
        :func:`apply_vcg_drift`. Grounds synthetic nonstationarity in real fetal dynamics.
    drift_scale : float, default 1.0
        Magnitude knob for ``vcg_drift``: 0 = stationary, 1 = real drift, k = exaggerated.
    fetal_moment_amplitude_nAm2 : float
        Target magnetic dipole moment amplitude for fetal signal in nanoAmpere·meter² (nA·m²).
        Typical fMCG: 10-50 nA·m². Default 20 nA·m².
    maternal_moment_amplitude_nAm2 : float
        Target magnetic dipole moment amplitude for maternal signal in nA·m².
        Typical adult MCG: 3000-10000 nA·m². Default 6000 nA·m².
    moment_amplitude_variation : float
        Random variation around target amplitudes (fraction). E.g., 0.2 = ±20%.
        Sampled uniformly as amplitude * (1 ± variation). Default 0.2.
    fetal_time_offset : float
        Time offset between fetal and maternal signals (seconds)
    fetal_downsample_factor : float
        Fetal heart rate relative to maternal (typically 0.5-0.7)
    fetal_rot : array (3,) or None
        Fetal dipole orientation [rx, ry, rz] in degrees. If None, sampled
        uniformly from fetal_rot_bounds using the realization RNG (seed-deterministic).
    maternal_rot : array (3,) or None
        Maternal dipole orientation in degrees. If None, sampled uniformly
        from maternal_rot_bounds using the realization RNG (seed-deterministic).
    fetal_rot_bounds : array (3, 2)
        Rotation sampling bounds in degrees for fetal_rot when fetal_rot is None.
        Rows are [min, max] for x/y/z Euler angles.
    maternal_rot_bounds : array (3, 2)
        Rotation sampling bounds in degrees for maternal_rot when maternal_rot is None.
        Rows are [min, max] for x/y/z Euler angles.
    decimation_factor : int
        Downsample factor from 1000 Hz
    sampfrom : float
        Start time (seconds)
    sampto : float or None
        End time (seconds)
    snr_db : float
        Signal-to-noise ratio in dB
    noise_type : str
        'gaussian' or 'recording'
    noise_data : array or None
        Recorded noise for noise_type='recording'
    fs_noise : float or None
        Sampling frequency of noise_data
    axis_mask : ndarray or None
        (n_sensors, 3) boolean/int array; 1/True = observed axis, 0/False = unobserved.
        When provided, unobserved channels are set to NaN in field_clean *before* noise
        addition so that sig_power() uses the same observed channels for both signal and
        noise, yielding correct SNR scaling. Unobserved channels in field_measured are
        also NaN, matching real sensor data structure.
        Typically taken from MeasurementConfig.axis_mask.
    seed : int or None
        Random seed for reproducibility
    return_clean : bool
        If True, also return clean (noiseless) combined field as 'field_clean'
    return_sources : bool
        If True, also return per-source ground-truth fields as 'field_fetal' and
        'field_maternal' (shape (n_samples, n_sensors, 3) in pT, NaN on masked axes).
        These are computed via the same ForwardModel using m_true/r_true, so
        field_fetal + field_maternal == field_clean (within floating-point precision).

    Returns
    -------
    data : dict
        Dictionary containing:
        - 'field_measured': (n_samples, n_sensors, 3) - Noisy field measurements in pT
        - 'field_clean': (n_samples, n_sensors, 3) - Clean field (if return_clean=True)
        - 'field_fetal': (n_samples, n_sensors, 3) - GT fetal field in pT (if return_sources=True)
        - 'field_maternal': (n_samples, n_sensors, 3) - GT maternal field in pT (if return_sources=True)
        - 'm_true': (n_samples, 2, 3) - True magnetic moments in μA·m²
        - 'r_true': (n_samples, 2, 3) - True positions in meters
        - 'time': (n_samples,) - Time array in seconds
        - 'fs': float - Sampling frequency in Hz
        - 'metadata': dict - Configuration metadata
    """
    rng = np.random.default_rng(seed)

    # Sample initial positions with physiological constraints
    print("Sampling physiological positions...")

    # Sample fetal position first
    r_fetal_init = sample_physiological_position(
        baseline=fetal_pos_baseline,
        bnds=fetal_pos_bounds,
        sigma=position_sampling_sigma,
        seed=rng.integers(0, 2**31) if seed is not None else None,
        constraints={'max_z': fetal_pos_bounds[2, 1]},  # Fetal not too deep
    )

    # Sample maternal position with separation constraint
    r_maternal_init = sample_physiological_position(
        baseline=maternal_pos_baseline,
        bnds=maternal_pos_bounds,
        sigma=position_sampling_sigma,
        seed=rng.integers(0, 2**31) if seed is not None else None,
        constraints={
            'min_z': maternal_pos_bounds[2, 0],  # Maternal deeper than fetal
            'other_position': r_fetal_init,
            'min_separation': min_separation,
        },
    )

    print(f"  Fetal position: {r_fetal_init * 100} cm")
    print(f"  Maternal position: {r_maternal_init * 100} cm")
    print(f"  Separation: {np.linalg.norm(r_fetal_init - r_maternal_init) * 100:.1f} cm")

    # Load VCG data for fetal and maternal separately
    print(f"Loading VCG data...")
    print(f"  Fetal: {vcg_fetal_record}")
    fetal_vcg, fetal_ecg_II, fs_vcg = load_vcg_data(
        record=vcg_fetal_record,
        base_path=vcg_dataset_path,
        plot=False,
    )

    print(f"  Maternal: {vcg_maternal_record}")
    maternal_vcg, maternal_ecg_II, _ = load_vcg_data(
        record=vcg_maternal_record,
        base_path=vcg_dataset_path,
        plot=False,
    )

    # Downsample VCG signals
    fetal_vcg = decimate(fetal_vcg, decimation_factor, axis=0)
    maternal_vcg = decimate(maternal_vcg, decimation_factor, axis=0)
    fetal_ecg_II = decimate(fetal_ecg_II, decimation_factor, axis=0)
    maternal_ecg_II = decimate(maternal_ecg_II, decimation_factor, axis=0)
    fs = fs_vcg / decimation_factor

    # Convert time values to indices
    sampfrom_idx = int(sampfrom * fs)
    fetal_offset_idx = int(fetal_time_offset * fs)
    if sampto is None:
        sampto_idx = min(len(fetal_vcg), len(maternal_vcg))
    else:
        sampto_idx = int(sampto * fs)

    # Convert Frank leads to Heart Shield coordinates
    fetal_vcg = np.array([
        -fetal_vcg[:, 0], -fetal_vcg[:, 2], -fetal_vcg[:, 1]
    ]).T
    maternal_vcg = np.array([
        -maternal_vcg[:, 0], -maternal_vcg[:, 2], -maternal_vcg[:, 1]
    ]).T

    # Convert moment amplitudes from nA·m² to μA·m² (factor of 1000)
    # VCG signals are in mV and are NUMERICALLY treated as μA·m² (no unit conversion)
    # This matches the old generate_synthetic_dipole_signals() behavior
    fetal_moment_target = fetal_moment_amplitude_nAm2 / 1000.0  # nA·m² to μA·m²
    maternal_moment_target = maternal_moment_amplitude_nAm2 / 1000.0
    
    # Sample random variation for this recording
    if moment_amplitude_variation > 0:
        fetal_variation = rng.uniform(1 - moment_amplitude_variation, 1 + moment_amplitude_variation)
        maternal_variation = rng.uniform(1 - moment_amplitude_variation, 1 + moment_amplitude_variation)
        fetal_moment_target *= fetal_variation
        maternal_moment_target *= maternal_variation
    
    # Calculate required samples
    n_samples = sampto_idx - sampfrom_idx

    # Extract and process maternal signal (with tiling if needed)
    maternal_available = len(maternal_vcg) - sampfrom_idx
    if n_samples <= maternal_available:
        # Enough samples available
        maternal_signal = maternal_vcg[sampfrom_idx:sampto_idx]
    else:
        # Need to tile VCG to reach desired duration
        print(f"  ⚠ Maternal VCG tiling: need {n_samples} samples, only {maternal_available} available")
        maternal_usable = maternal_vcg[sampfrom_idx:]  # All available from start point
        n_repeats = int(np.ceil(n_samples / len(maternal_usable)))
        maternal_tiled = np.tile(maternal_usable, (n_repeats, 1))
        maternal_signal = maternal_tiled[:n_samples]
        print(f"    → Tiled {n_repeats} times to reach {len(maternal_signal)} samples")

    # Extract and resample fetal signal (different heart rate, with tiling if needed)
    fetal_start_idx = sampfrom_idx + fetal_offset_idx
    original_fetal_samples_needed = int(n_samples / fetal_downsample_factor)
    fetal_available = len(fetal_vcg) - fetal_start_idx

    if original_fetal_samples_needed <= fetal_available:
        # Enough samples available
        fetal_signal = fetal_vcg[
            fetal_start_idx : fetal_start_idx + original_fetal_samples_needed
        ]
    else:
        # Need to tile VCG to reach desired duration
        print(f"  ⚠ Fetal VCG tiling: need {original_fetal_samples_needed} samples, only {fetal_available} available")
        fetal_usable = fetal_vcg[fetal_start_idx:]  # All available from start point
        n_repeats = int(np.ceil(original_fetal_samples_needed / len(fetal_usable)))
        fetal_tiled = np.tile(fetal_usable, (n_repeats, 1))
        fetal_signal = fetal_tiled[:original_fetal_samples_needed]
        print(f"    → Tiled {n_repeats} times to reach {len(fetal_signal)} samples")

    # Resample fetal signal to match output length
    fetal_signal = resample(fetal_signal, n_samples, axis=0)
    
    # Compute peak magnitude of VCG signals for scaling (corresponds to R-peak)
    # Target amplitudes refer to peak cardiac dipole moment, not mean
    fetal_magnitudes = np.linalg.norm(fetal_signal, axis=1)
    maternal_magnitudes = np.linalg.norm(maternal_signal, axis=1)
    fetal_vcg_max_mag = np.max(fetal_magnitudes)
    maternal_vcg_max_mag = np.max(maternal_magnitudes)
    
    # Scale signals: VCG (mV) is numerically treated as dipole moment (μA·m²)
    # Target amplitude refers to peak magnitude (R-peak)
    fetal_scaling = fetal_moment_target / fetal_vcg_max_mag
    maternal_scaling = maternal_moment_target / maternal_vcg_max_mag
    
    fetal_signal = fetal_scaling * fetal_signal
    maternal_signal = maternal_scaling * maternal_signal
    
    print(f"  Moment amplitude (with random variation):")
    print(f"    Fetal: {fetal_moment_target:.3f} μA·m² = {fetal_moment_target*1000:.1f} nA·m² (peak magnitude: {fetal_vcg_max_mag:.3f} mV, scaling: {fetal_scaling:.4f})")
    print(f"    Maternal: {maternal_moment_target:.3f} μA·m² = {maternal_moment_target*1000:.1f} nA·m² (peak magnitude: {maternal_vcg_max_mag:.3f} mV, scaling: {maternal_scaling:.4f})")
    print(f"    Ratio: {maternal_moment_target/fetal_moment_target:.1f}×")
    
    # Resolve dipole orientations (seed-deterministic per realization).
    if fetal_rot is None:
        fetal_bounds = np.asarray(fetal_rot_bounds, dtype=float)
        if fetal_bounds.shape != (3, 2):
            raise ValueError(f"fetal_rot_bounds must have shape (3, 2), got {fetal_bounds.shape}")
        fetal_rot = rng.uniform(fetal_bounds[:, 0], fetal_bounds[:, 1])
    else:
        fetal_rot = np.asarray(fetal_rot, dtype=float)

    if maternal_rot is None:
        maternal_bounds = np.asarray(maternal_rot_bounds, dtype=float)
        if maternal_bounds.shape != (3, 2):
            raise ValueError(
                f"maternal_rot_bounds must have shape (3, 2), got {maternal_bounds.shape}"
            )
        maternal_rot = rng.uniform(maternal_bounds[:, 0], maternal_bounds[:, 1])
    else:
        maternal_rot = np.asarray(maternal_rot, dtype=float)

    if fetal_rot.shape != (3,):
        raise ValueError(f"fetal_rot must have shape (3,), got {fetal_rot.shape}")
    if maternal_rot.shape != (3,):
        raise ValueError(f"maternal_rot must have shape (3,), got {maternal_rot.shape}")

    print(f"  Fetal rotation [deg]: {np.round(fetal_rot, 2)}")
    print(f"  Maternal rotation [deg]: {np.round(maternal_rot, 2)}")

    # Apply rotations
    fetal_signal = rot(fetal_signal, rot=fetal_rot)
    maternal_signal = rot(maternal_signal, rot=maternal_rot)

    # Construct magnetic moments
    m_true = np.stack([fetal_signal, maternal_signal], axis=1)  # (n_samples, 2, 3)

    # VCG drift: replay a measured real drift trajectory onto the fetal cardiac vector
    if vcg_drift is not None and drift_scale != 0:
        from fmcg.analysis.hr import compute_hr
        _, fetal_peaks = compute_hr(m_true[:, 0, :], int(fs), plot=False)
        print(f"Applying VCG drift replay (drift_scale={drift_scale}, {len(fetal_peaks)} fetal beats)...")
        m_true[:, 0, :] = apply_vcg_drift(m_true[:, 0, :], fetal_peaks, vcg_drift, drift_scale)

    # Generate dipole trajectories
    if enable_movement:
        print(f"Generating dipole movement ({movement_type})...")
        r_init = np.array([r_fetal_init, r_maternal_init])  # (2, 3)
        bnds = np.array([fetal_pos_bounds, maternal_pos_bounds])  # (2, 3, 2)

        r_true = generate_dipole_movement(
            r_init=r_init,
            bnds=bnds,
            n_samples=n_samples,
            movement_type=movement_type,
            step_size=movement_step_size,
            tau=movement_tau,
            momentum=movement_momentum,
            seed=rng.integers(0, 2**31) if seed is not None else None,
            fs=fs,
            lowpass_cutoff=movement_lowpass_cutoff,
        )  # (n_samples, 2, 3)
    else:
        # Static positions
        r_true = np.tile(
            np.array([r_fetal_init, r_maternal_init])[None, :, :],
            (n_samples, 1, 1)
        )

    # Forward model: compute magnetic field at sensors
    print(f"Computing forward model on {device}...")
    forward_model = ForwardModel(r_sensors, device=device)
    field_clean = forward_model.forward_numpy(m_true, r_true)  # (n_samples, n_sensors, 3)

    # With mu0_4pi=0.1 and moments in μA·m², forward model outputs field in pT directly
    # No unit conversion needed

    # Apply axis mask: set unobserved channels to NaN so that (a) sig_power() for SNR
    # scaling uses only the same observed channels as the noise, and (b) field_measured
    # has NaN on unobserved axes matching real sensor data structure.
    if axis_mask is not None:
        valid = np.asarray(axis_mask, dtype=bool)[np.newaxis, :, :]  # (1, n_sensors, 3)
        field_clean = np.where(valid, field_clean, np.nan)

    # Add noise (snr_db=None → skip noise, field_measured equals field_clean)
    time = np.arange(n_samples) / fs
    if snr_db is None:
        field_measured = field_clean.copy()
    else:
        print(f"Adding noise (SNR={snr_db} dB, type={noise_type})...")
        field_measured = add_aligned_noise(
            field_data=field_clean,
            time=time,
            noise_data=noise_data,
            fs_field=fs,
            fs_noise=fs_noise,
            noise_type=noise_type,
            snr_db=snr_db,
            seed=rng.integers(0, 2**31) if seed is not None else None,
        )

    # Add Gaussian noise relative to clean signal (independent of real noise level)
    # Reference: P99 of |field_clean| tracks R-peak amplitude scale, so fraction=0.05
    # means sigma = 5% of a representative peak — more intuitive than RMS for cardiac signals.
    if gaussian_noise_fraction is not None:
        reference_amplitude = np.nanpercentile(np.abs(field_clean), 99)
        sigma = gaussian_noise_fraction * reference_amplitude
        gaussian_seed = rng.integers(0, 2**31) if seed is not None else None
        gaussian_rng = np.random.default_rng(gaussian_seed)
        gaussian_noise = gaussian_rng.normal(0, sigma, field_measured.shape)
        field_measured = field_measured + gaussian_noise
        print(f"Added Gaussian noise: fraction={gaussian_noise_fraction}, ref_p99={reference_amplitude:.4e}, std={sigma:.4e}")

    # Prepare output
    data = {
        'field_measured': field_measured,
        'm_true': m_true,
        'r_true': r_true,
        'time': time,
        'fs': fs,
        'metadata': {
            'vcg_fetal_record': vcg_fetal_record,
            'vcg_maternal_record': vcg_maternal_record,
            'r_fetal_init': r_fetal_init,
            'r_maternal_init': r_maternal_init,
            'fetal_scaling': fetal_scaling,
            'fetal_downsample_factor': fetal_downsample_factor,
            'fetal_rot_deg': fetal_rot,
            'maternal_rot_deg': maternal_rot,
            'enable_movement': enable_movement,
            'movement_type': movement_type if enable_movement else 'static',
            'drift_scale': drift_scale if vcg_drift is not None else 0.0,
            'snr_db': snr_db,
            'noise_type': noise_type,
            'gaussian_noise_fraction': gaussian_noise_fraction,
            'seed': seed,
            'duration': n_samples / fs,
            'n_samples': n_samples,
        }
    }

    if return_clean:
        data['field_clean'] = field_clean

    if return_sources:
        field_fetal_gt = forward_model.forward_numpy(m_true[:, [0], :], r_true[:, [0], :])
        field_maternal_gt = forward_model.forward_numpy(m_true[:, [1], :], r_true[:, [1], :])
        if axis_mask is not None:
            field_fetal_gt = np.where(valid, field_fetal_gt, np.nan)
            field_maternal_gt = np.where(valid, field_maternal_gt, np.nan)
        data['field_fetal'] = field_fetal_gt
        data['field_maternal'] = field_maternal_gt

    print(f"✅ Generated synthetic recording: {n_samples} samples ({n_samples/fs:.1f} s) at {fs} Hz")

    return data