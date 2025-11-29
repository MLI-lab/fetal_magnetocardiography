import os
import glob
import logging

import numpy as np
import wfdb
from nptdms import TdmsFile
from scipy.signal import butter, filtfilt

from .plotting import plot_utils

logger = logging.getLogger(__name__)

def load_vcg_data(
    record="patient104/s0306lre", base_path="/jonas/fMCG/datasets/ptbdb", plot=False
):
    """
    Load and preprocess VCG (Vectorcardiogram) data from a specified record.

    Parameters:
        record (str): The path to the record file within the dataset. Default is "patient104/s0306lre".
        base_path (str): The base path to the dataset directory. Default is "/jonas/fMCG/datasets/ptbdb".
        plot (bool): If True, plots the VCG signals using plot_utils. Default is False.

    Returns:
        tuple: A tuple containing:
            - filtered_signals (ndarray): The bandpass filtered VCG signals.
            - ecg_II (ndarray): The ECG lead II signal.
            - fs (int): The sampling frequency of the record.
    """
    record = wfdb.rdrecord(f"{base_path}/{record}", physical=True)
    fs = record.fs

    # retrieve ordered indices of the signals in the record
    indices = np.argsort(np.argsort(["vx", "vy", "vz"]))
    sorted_indices = np.where(np.in1d(record.sig_name, ["vx", "vy", "vz"]))[0][indices]

    # retrieve the signals
    signals = record.p_signal[:, sorted_indices]

    ecg_II = record.p_signal[:, 1]

    # Bandpass filter the signals
    lowcut = 0.5
    highcut = 40.0
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype="band")
    filtered_signals = filtfilt(b, a, signals, axis=0)

    if plot:
        plot_utils.plot_vcg(filtered_signals, ecg_II, fs)
    return filtered_signals, ecg_II, fs


def load_structured_patient_data_and_noise(
    SystemConfig,
    MeasurementConfig,
    ds_path=None,
    patient="P077",
    series="S01",
    files="all",
    sig_group_names="R001",
    noise_group_names="R004",
    noise_patient=None,
    noise_series=None,
    offset=5000,
    print_groups=False,
):
    """
    Loads patient data and noise data from TDMS files, synchronizes the data, and returns the data along with time vector and sampling frequency.

    Parameters:
        SystemConfig (dataclass): Configuration object containing system-specific settings, including coordinate mappings and sensor types.
        MeasurementConfig (dataclass): Configuration object containing measurement-specific settings, including sensor positions and channel mappings.
        ds_path (str): Path to the dataset directory. Required parameter.
        patient (str): Patient identifier. Default is "P077".
        series (str): Series identifier. Default is "S01".
        files (str or list): List of file identifiers to load. If "all", load all files in the series. Default is "all".
        sig_group_names (str or list): Group names for signal data within the TDMS files.
        noise_group_names (str or list): Group names for noise data within the TDMS files.
        noise_patient (str, optional): Patient identifier for noise data. If None, uses the same patient as signal data. Default is None.
        noise_series (str, optional): Series identifier for noise data. If None, uses the same series as signal data. Default is None.
        offset (int): Number of samples to truncate from the beginning. Defaults to 5000.
        print_groups (bool): If True, prints available groups in the TDMS file. Default is False.

    Returns:
        tuple: A tuple containing:
            - data_dict (dict): Dictionary where keys are channel names and values are numpy arrays of the channel data.
            - time (numpy.ndarray): Time vector corresponding to the data.
            - fs (int): Sampling frequency in Hz.
            - noise_dict (dict): Dictionary where keys are channel names and values are numpy arrays of the noise data.
    """
    if ds_path is None:
        raise ValueError("ds_path must be provided. Please specify the path to the dataset directory.")

    # Load signal data
    sig_data_dict, time, fs = load_patient_data(
        ds_path=ds_path,
        patient=patient,
        series=series,
        files=files,
        group_names=sig_group_names,
        offset=offset,
        print_groups=print_groups,
    )

    # Load noise data
    noise_data_dict, _, _ = load_patient_data(
        ds_path=ds_path,
        patient=patient if noise_patient is None else noise_patient,
        series=series if noise_series is None else noise_series,
        files=files,
        group_names=noise_group_names,
        # offset=offset,
        # print_groups=print_groups
    )

    sig_data_dict = structure_data(sig_data_dict, SystemConfig, MeasurementConfig)
    noise_data_dict = structure_data(noise_data_dict, SystemConfig, MeasurementConfig)

    return sig_data_dict, time, fs, noise_data_dict


def load_patient_data(
    ds_path=None,
    patient="P077",
    series="S01",
    files="all",
    group_names="R001",
    offset=5000,
    print_groups=False,
):
    """
    Loads patient data from TDMS files, synchronizes the data, and returns the data along with time vector and sampling frequency.

    Parameters:
        ds_path (str): Path to the dataset directory. Required parameter.
        patient (str): Patient identifier. Default is "P077".
        series (str): Series identifier. Default is "S01".
        files (str or list): List of file identifiers to load. If "all", load all files in the series. Default is "all".
        group_names (str or list): Group names within the TDMS files. If a single string is provided, it is used for all files.
            In case of processing multiple TDMS files, different goups can be specified by passing a list. In this case the groups
            and files are processed sequentially, each group element corresponding to a file. Default is "R001".
        offset (int): Number of samples to truncate from the beginning. Defaults to 5000.
        print_groups (bool): If True, prints available groups in the TDMS file. Default is False.

    Returns:
        tuple: A tuple containing:
            - data_dict (dict): Dictionary where keys are channel names and values are numpy arrays of the channel data.
            - time (numpy.ndarray): Time vector corresponding to the data.
            - fs (int): Sampling frequency in Hz.

    Notes:
        - The function synchronizes the data by truncating to the latest starting channel and ensuring all channels have the same length.
        - The function reads data from TDMS files using the `TdmsFile` class and scales the data appropriately.
        - The time vector is created based on the synchronized data length and the sampling frequency.
        - The function assumes a fixed voltage-to-flux scaling factor of 2.7e-3 V/pT.
        - The sampling frequency is assumed to be 1000 Hz.
    """
    if ds_path is None:
        raise ValueError("ds_path must be provided. Please specify the path to the dataset directory.")

    volt2flux_scaling = 2.7e-3  # Voltage to flux scaling factor (V/pT)
    fs = 1000  # Sampling frequency in Hz

    # Collect all file paths to process
    pre = ""
    if patient is not None and series is not None:
        pre = f"{patient}/{patient}_{series}_"
    if files == "all":
        file_list = glob.glob(f"{ds_path}/{pre}*.tdms")
    else:
        file_list = [f"{ds_path}/{pre}{file}.tdms" for file in files]

    # Ensure group_names is a list, even if a single string is provided
    if isinstance(group_names, str):
        group_names = [group_names] * len(file_list)

    # Initialize dictionaries to store waveform start times and data
    wf_start_times = {}
    data_dict = {}

    # Iterate over all files and their corresponding group names
    for file_path, group_name in zip(file_list, group_names):
        tdms_file = TdmsFile.read(file_path)
        if print_groups:
            print(
                f"{os.path.basename(file_path)}: available groups {[g.name for g in tdms_file.groups()]}"
            )
        logger.debug(
            f"{os.path.basename(file_path)}: available groups {[g.name for g in tdms_file.groups()]}"
        )
        # Collect data from all channels in the group
        for i, channel in enumerate(tdms_file[group_name].channels()):
            data_dict[channel.name] = channel.read_data(scaled=True) / volt2flux_scaling
            wf_start_times[channel.name] = channel.properties["wf_start_time"]
            # Ensure the sampling frequency is consistent
            assert fs == 1 / channel.properties["wf_increment"]

    # Synchronize the data by truncating to the latest starting channel
    latest_start_time = max(wf_start_times.values())
    for key in data_dict.keys():
        time_diff = (latest_start_time - wf_start_times[key]).astype(
            "timedelta64[us]"
        ).astype(
            float
        ) / 1_000_000  # Convert to seconds
        n_samples = int(time_diff * fs)  # Calculate the number of samples to truncate
        data_dict[key] = data_dict[key][n_samples:]

    # Truncate the data at the end to ensure all channels have the same length
    shortest_length = min([len(data_dict[key]) for key in data_dict.keys()])
    data_dict = {
        key: data_dict[key][:shortest_length][offset:] for key in data_dict.keys()
    }

    # Create a time vector based on the synchronized data length and sampling frequency
    time = np.arange(data_dict[list(data_dict.keys())[0]].shape[0]) / fs

    return data_dict, time, fs


def structure_data(data_dict, SystemConfig, MeasurementConfig):
    """
    Converts sensor data loaded with `utils.load_patient_data` into a 3D grid format, by mapping the recorded DAC
    channels to the corresponding sensor and position, and aligning all coordinate systems to that of the holder.

    Parameters:
        data_dict (dict): A dictionary where keys are channel names and values are numpy arrays of sensor data (e.g., loaded with
            `utils.load_patient_data`).
        SystemConfig (dataclass): Configuration object containing system-specific settings, including coordinate mappings and sensor types.
        MeasurementConfig (dataclass): Configuration object containing measurement-specific settings, including sensor positions and channel mappings.

    Returns:
        dict: A dictionary where keys are sensor names, in descending order as installed in the holder, and values are 3D
            numpy arrays (samples x 3 axes)
    """

    sig_len = list(data_dict.values())[0].shape[
        0
    ]  # Get the length of the signal from the first channel
    empty_count = 0
    # Iterate over all grid / sensor positions and their sensor orientation (represented by the coordinate mapping)
    grid = {}
    for quspin, coor_mapping in zip(
        np.array(MeasurementConfig.quspin_positions).flat,
        SystemConfig.gen3quspin_coor_to_plate_coor,
    ):
        if quspin == "":
            empty_count += 1
            quspin = f"Empty {empty_count}"

        # Initialize a 3D grid for each sensor (samples x 3 axes) with NaNs
        grid[quspin] = np.ones((sig_len, 3)) * np.nan

        if quspin == f"Empty {empty_count}":
            continue

        # Iterate over all DAC channels, each recording a single axis of a sensor
        for channel_idx, channel in enumerate(MeasurementConfig.quspin_mapping[quspin]):
            if channel is not None:
                # Find the index of the current axis in the plate coordinates
                plate_coor_index = np.abs(coor_mapping).tolist().index(channel_idx + 1)
                # Determine the sign of the axis direction
                sign = np.sign(coor_mapping[plate_coor_index])
                # Adjust the axis direction if the sensor is a Gen2 sensor
                if SystemConfig.is_gen2[quspin]:
                    sign *= SystemConfig.gen3_to_gen2[channel_idx]
                # Map the data from the DAC channel to the corresponding position in the grid
                grid[quspin][:, plate_coor_index] = data_dict[channel] * sign

    return grid


def generate_array_coordinates(grid_shape, grid_spacing, y=0):
    """
    Generate a 2D array of coordinates for a grid centered around the origin.

    Parameters:
        grid_shape (tuple): A tuple (n_rows, n_cols) representing the shape of the grid.
        grid_spacing (float): The spacing between grid points.
        y (float, optional): The y-coordinate for all points in the grid. Default is 0.

    Returns:
        numpy.ndarray: A 2D array of shape (n_rows * n_cols, 3) containing the coordinates of the grid points.
    """
    # Calculate the offsets to center the grid
    x_offset = (grid_shape[0] - 1) * grid_spacing / 2
    z_offset = (grid_shape[1] - 1) * grid_spacing / 2

    # Create centered grid coordinates
    grid_coordinates = np.array(
        [
            [x * grid_spacing - x_offset, y, z * grid_spacing - z_offset]
            for z in range(grid_shape[1] - 1, -1, -1)
            for x in range(grid_shape[0])
        ]
    )
    return grid_coordinates


def parse_tdms_filename(filename, remove_prefix=True):
    """
    Parse TDMS filename of format P052_S01_D2024-06-20_G29.tdms
    Returns dict with patient, series, date, and group information

    Parameters:
        filename (str): The TDMS filename to parse.
        remove_prefix (bool): Whether to remove the leading characters from the parsed values. E.g. 'P052' becomes '052'.
    Returns:
        dict: A dictionary with the parsed patient, series, date, and gestation information.
    """
    # Remove path and extension
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]

    # Split by underscore
    parts = name_without_ext.split("_")

    i = 1 if remove_prefix else 0
    return {
        "patient": parts[0][i:],  # P052
        "series": parts[1][i:],  # S01
        "date": parts[2][i:],  # 2024-06-20 (remove 'D' prefix)
        "gestation": parts[3][i:],  # G29
    }
