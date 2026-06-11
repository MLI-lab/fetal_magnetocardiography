import os
import glob
import logging
import datetime
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import wfdb
from nptdms import TdmsFile
from scipy.signal import butter, filtfilt
import pickle
from typing import Optional, Any, Dict

from fmcg.signal.artifact_removal import remove_outlier_dict
from fmcg.signal.filtering import filter_sensor_dict
from fmcg.signal.whitening import apply_whitening

from .plotting import plot_utils
from .utils import get_config_for_date

logger = logging.getLogger(__name__)

def load_vcg_data(
    record="patient104/s0306lre", base_path=None, plot=False
):
    """
    Load and preprocess VCG (Vectorcardiogram) data from a specified record.

    Parameters:
        record (str): The path to the record file within the dataset. Default is "patient104/s0306lre".
        base_path (str): The base path to the dataset directory. Required parameter - must point to your local ptbdb dataset directory.
        plot (bool): If True, plots the VCG signals using plot_utils. Default is False.

    Returns:
        tuple: A tuple containing:
            - filtered_signals (ndarray): The bandpass filtered VCG signals.
            - ecg_II (ndarray): The ECG lead II signal.
            - fs (int): The sampling frequency of the record.
    """
    # If no local base path is provided, try to load the record from the
    # PhysioNet-hosted PTB-DB. Otherwise read from the provided local path.
    if base_path is None:
        # Load record from the PhysioNet-hosted PTB Diagnostic ECG Database.
        # Database URL: https://physionet.org/content/ptbdb/1.0.0/
        # wfdb strips directory components from record_name, so we need to
        # include any subdirectory (e.g., "patient104") in pn_dir.
        record_dir = os.path.dirname(record)  # e.g., "patient104"
        record_base = os.path.basename(record)  # e.g., "s0306lre"
        if record_dir:
            pn_dir = f"ptbdb/1.0.0/{record_dir}"
        else:
            pn_dir = "ptbdb/1.0.0"
        try:
            record = wfdb.rdrecord(record_base, physical=True, pn_dir=pn_dir)
        except Exception as e:
            logger.warning(
                "Failed to load record '%s' from PhysioNet (pn_dir=%s): %s",
                record_base, pn_dir, e
            )
            raise ValueError(
                f"Could not load record '{record}' from PhysioNet. "
                "Provide a local base_path or check the record name. "
                f"Error: {e}"
            ) from e
    else:
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
    SystemConfig=None,
    MeasurementConfig=None,
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
    return_configs=False,
):
    """
    Loads patient data and noise data from TDMS files, synchronizes the data, and returns the data along with time vector and sampling frequency.

    If SystemConfig and MeasurementConfig are None (recommended), they will be automatically determined from the
    measurement date in the TDMS filename using get_config_for_date(). Explicit config passing is deprecated.

    Parameters:
        SystemConfig (dataclass, optional): DEPRECATED. Configuration object containing system-specific settings.
            If None, automatically determined from measurement date. Default is None.
        MeasurementConfig (dataclass, optional): DEPRECATED. Configuration object containing measurement-specific settings.
            If None, automatically determined from measurement date. Default is None.
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
        return_configs (bool): If True, returns configuration information in a dict. Default is False.

    Returns:
        tuple: If return_configs=False (default):
            - data_dict (dict): Dictionary where keys are channel names and values are numpy arrays of the channel data.
            - time (numpy.ndarray): Time vector corresponding to the data.
            - fs (int): Sampling frequency in Hz.
            - noise_dict (dict): Dictionary where keys are channel names and values are numpy arrays of the noise data.

        If return_configs=True:
            - data_dict, time, fs, noise_dict, config (dict)

            Where config dict contains:
                - 'SystemConfig': System configuration object (used for both signal and noise)
                - 'MeasurementConfig': Measurement configuration object (used for both signal and noise)
                - 'date': Measurement date string from signal data (YYYY-MM-DD)
                - 'r_sensors': Sensor positions array
                - 'axis_mask': Axis mask array
    """
    if ds_path is None:
        raise ValueError("ds_path must be provided. Please specify the path to the dataset directory.")

    # Issue deprecation warning if configs were explicitly provided
    if SystemConfig is not None or MeasurementConfig is not None:
        import warnings
        warnings.warn(
            "Explicitly passing SystemConfig and MeasurementConfig to load_structured_patient_data_and_noise "
            "is deprecated and will be ignored. Configs will be automatically determined.",
            DeprecationWarning,
            stacklevel=2
        )

    # Load signal data with auto-detected configs
    sig_data_dict, time, fs, config_sig = load_patient_data(
        ds_path=ds_path,
        patient=patient,
        series=series,
        files=files,
        group_names=sig_group_names,
        offset=offset,
        print_groups=print_groups,
        return_configs=True
    )

    # Load noise data with auto-detected configs
    noise_data_dict, _, _, config_noise = load_patient_data(
        ds_path=ds_path,
        patient=patient if noise_patient is None else noise_patient,
        series=series if noise_series is None else noise_series,
        files=files,
        group_names=noise_group_names,
        return_configs=True
        # offset=offset,  # Note: offset not applied to noise data
        # print_groups=print_groups
    )

    # Check if signal and noise configs are consistent (compare the actual config objects, not dates)
    if (config_sig['SystemConfig'] is not config_noise['SystemConfig'] or
        config_sig['MeasurementConfig'] is not config_noise['MeasurementConfig']):
        import warnings
        warnings.warn(
            f"Signal and noise data have different configurations (signal date={config_sig['date']}, "
            f"noise date={config_noise['date']}). Using signal configs for both.",
            UserWarning,
            stacklevel=2
        )

    sig_data_dict = structure_data(sig_data_dict, config_sig['SystemConfig'], config_sig['MeasurementConfig'])
    noise_data_dict = structure_data(noise_data_dict, config_noise['SystemConfig'], config_noise['MeasurementConfig'])

    if return_configs:
        config = {
            'SystemConfig': config_sig['SystemConfig'],
            'MeasurementConfig': config_sig['MeasurementConfig'],
            'date': config_sig['date'],
            'r_sensors': config_sig['r_sensors'],
            'axis_mask': config_sig['axis_mask']
        }
        return sig_data_dict, time, fs, noise_data_dict, config

    return sig_data_dict, time, fs, noise_data_dict


def load_patient_data(
    ds_path=None,
    patient="P077",
    series="S01",
    files="all",
    group_names="R001",
    offset=5000,
    print_groups=False,
    return_configs=False,
):
    """
    Loads patient data from TDMS files, synchronizes the data, and returns the data along with time vector and sampling frequency.

    Automatically determines SystemConfig and MeasurementConfig from the measurement date in the TDMS filename.

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
        return_configs (bool): If True, returns auto-detected configuration information in a dict. Default is False.

    Returns:
        tuple: If return_configs=False (default):
            - data_dict (dict): Dictionary where keys are channel names and values are numpy arrays of the channel data.
            - time (numpy.ndarray): Time vector corresponding to the data.
            - fs (int): Sampling frequency in Hz.

        If return_configs=True:
            - data_dict, time, fs, config (dict)

            Where config dict contains:
                - 'SystemConfig': System configuration object
                - 'MeasurementConfig': Measurement configuration object
                - 'date': Measurement date string (YYYY-MM-DD)
                - 'r_sensors': Sensor positions array
                - 'axis_mask': Axis mask array

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

    if return_configs:
        date = os.path.basename(file_path).split("_")[2]  # Extract the date part 
        #remove the leading 'D' if present
        if date.startswith('D'):
            date = date[1:]
        MeasurementConfig, SystemConfig, axis_mask, r_sensors = get_config_for_date(date)
        config = {
            'SystemConfig': SystemConfig,
            'MeasurementConfig': MeasurementConfig,
            'date': date,
            'r_sensors': r_sensors,
            'axis_mask': axis_mask
        }
        return data_dict, time, fs, config

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


def load_pipeline_results_compact(path: str) -> Dict[str, Any]:
    """
    Load the compact pipeline payload or adapt a legacy out_data file into the compact form.

    Returns a dict with keys:
      - 'shared_reference': dict with peaks/hr/fs for fetal and maternal
      - 'methods': dict mapping method_name -> ndarray (time x channels) or None
    """
    # Try pickle-based payload first
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception:
        obj = None

    if isinstance(obj, dict):
        # Already compact
        if "shared_reference" in obj and "methods" in obj:
            return {
                "shared_reference": obj["shared_reference"],
                "methods": {k: (np.asarray(v) if v is not None else None) for k, v in obj["methods"].items()},
            }

        # Legacy out_data
        if "comparison_methods" in obj:
            cm = obj["comparison_methods"]
            shared = cm.get("shared_reference", {})
            methods_map: Dict[str, Optional[np.ndarray]] = {}
            for name, payload in cm.get("methods", {}).items():
                if isinstance(payload, dict) and "signal" in payload:
                    methods_map[name] = np.asarray(payload["signal"])
                else:
                    methods_map[name] = None
            return {"shared_reference": shared, "methods": methods_map}

    # Try loading as npz (np.savez) fallback
    try:
        arr = np.load(path, allow_pickle=True)
        files = set(arr.files)
        if "shared_reference" in files:
            shared = arr["shared_reference"].item() if arr["shared_reference"].shape == () else arr["shared_reference"]
            methods_map: Dict[str, Optional[np.ndarray]] = {}
            if "methods" in files:
                methods_obj = arr["methods"].item() if arr["methods"].shape == () else arr["methods"]
                if isinstance(methods_obj, dict):
                    for k, v in methods_obj.items():
                        methods_map[k] = np.asarray(v)
            else:
                # discover keys named method__{name}
                for key in arr.files:
                    if key.startswith("method__"):
                        mname = key.split("__", 1)[1]
                        methods_map[mname] = np.asarray(arr[key])
            return {"shared_reference": shared, "methods": methods_map}
    except Exception:
        pass

    raise ValueError(f"Unsupported pipeline results format: {path}")


def expand_compact_to_legacy(compact_payload: Dict[str, Any], fm_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Expand a compact payload into a minimal legacy-style `out_data` dict suitable for plotting consumers.

    If `fm_payload` is provided it will be used as a base to copy other needed fields.
    """
    legacy = dict(fm_payload) if fm_payload is not None else {}
    shared = compact_payload.get("shared_reference", {})
    methods = compact_payload.get("methods", {}) or {}
    legacy_cmp: Dict[str, Any] = {"shared_reference": shared, "methods": {}}
    for name, arr in methods.items():
        legacy_cmp["methods"][name] = {"signal": np.asarray(arr) if arr is not None else None}
    legacy["comparison_methods"] = legacy_cmp
    return legacy


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
        "patient": parts[0][i:] if len(parts) > 0 else None,  # P052
        "series": parts[1][i:] if len(parts) > 1 else None,  # S01
        "date": parts[2][i:] if len(parts) > 2 else None,  # 2024-06-20 (remove 'D' prefix)
        "gestation": parts[3][i:] if len(parts) > 3 else None,  # G29
    }


def get_whitening_matrix_from_data(params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sig_data_dict_raw, _, fs, noise_data_dict_raw, cfg = load_structured_patient_data_and_noise(
        **{
            key: params[key]
            for key in params.keys()
            & {
                "ds_path",
                "patient",
                "series",
                "sig_group_names",
                "noise_group_names",
                "noise_patient",
                "noise_series",
            }
        },
        files="all",
        print_groups=False,
        return_configs=True,
    )

    sensor_dict = filter_sensor_dict(
        sig_data_dict_raw,
        fs,
        low=params["bandpass_low"],
        high=params["bandpass_high"],
        order=4,
    )
    noise_sensor_dict = filter_sensor_dict(
        noise_data_dict_raw,
        fs,
        low=params["bandpass_low"],
        high=params["bandpass_high"],
        order=4,
    )

    sensor_dict, axis_mask = remove_outlier_dict(
        sensor_dict, physical_range_threshold=1, verbose=False
    )
    noise_sensor_dict, axis_mask_noise = remove_outlier_dict(
        noise_sensor_dict, physical_range_threshold=0.1, verbose=False
    )
    axis_mask = axis_mask & axis_mask_noise

    field_maps = np.array([value for value in sensor_dict.values()]).transpose(1, 0, 2)
    noise_maps = np.array([value for value in noise_sensor_dict.values()]).transpose(1, 0, 2)

    field_maps, whitening_matrix = apply_whitening(
        field_maps,
        noise_maps,
        axis_mask,
        method=params["whitening"],
        return_whitening_matrix=True,
        rescale=params["rescale_whitening"],
    )

    r_sensor_pos = np.asarray(cfg["r_sensors"])
    return whitening_matrix, axis_mask, field_maps, r_sensor_pos


def find_matching_file(
    file_list: Sequence[str], patient: str, series: str, signal_group: str, filter_str: str = ""
) -> str:
    matches = [
        file_path
        for file_path in file_list
        if f"{patient}_{series}_{signal_group}" in file_path and filter_str in file_path
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected 1 file for {patient}_{series}_{signal_group}, found {len(matches)}"
        )
    return matches[0]
