import contextlib
import importlib
from datetime import datetime
import io
import sys
import logging
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import tqdm

import torch
import torch.nn as torch_nn

def get_config_for_date(date_str):
    # 1. Clean and parse the date safely
    # Removes non-numeric prefix characters like 'v2025...' or ' 2025...'
    clean_date_str = date_str.lstrip('D ') 
    current_date = datetime.strptime(clean_date_str, "%Y-%m-%d").date()

    # 2. Define your configuration timeline (Newest to Oldest)
    # This makes it easy to add a new config: just add one line here!
    configs = [
        (datetime(2025, 9, 5).date(),  "fmcg.configs.measurement_config_2025_09_04"),
        (datetime(2024, 7, 5).date(),  "fmcg.configs.measurement_config_2024_07_05"),
        (datetime(2024, 3, 7).date(),  "fmcg.configs.measurement_config_2024_03_07"),
        (datetime(2023, 12, 5).date(),  "fmcg.configs.measurement_config_2023_12_05"),
    ]

    # Sort configs by date (newest first) to ensure correct matching
    configs.sort(key=lambda x: x[0], reverse=True)

    # 3. Find the first config that matches (moving backward in time)
    for threshold, module_path in configs:
        if current_date >= threshold:
            module = importlib.import_module(module_path)
            logger = logging.getLogger(__name__)
            logger.info(f"Using config {module_path} for date {current_date}")
            print(f"Using config {module_path} for date {current_date}")
            return module.MeasurementConfig, module.SystemConfig, getattr(module, 'axis_mask', None), getattr(module, 'r_sensors', None) 

    # If no config matched (date is before all thresholds), raise an error
    raise ValueError(f"No configuration found for date {current_date}. Date is before all config thresholds.")

class TqdmLoggingHandler(logging.Handler):
    """Logging handler that sends formatted log records through tqdm.tqdm.write.

    This handler is intended for console applications that display tqdm progress
    bars. Using tqdm.tqdm.write preserves progress-bar rendering by writing log
    messages on a new line without interfering with the bar.

    Parameters
    ----------
    level : int, optional
        Logging level for the handler (default: logging.NOTSET).

    Behavior
    --------
    On emit, the handler formats the record with its formatter, writes the result
    with tqdm.tqdm.write, and flushes. Any exceptions raised during emit are
    delegated to logging.Handler.handleError.
    """

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


@contextlib.contextmanager
def silence():
    """Temporarily suppress sys.stdout within a with-block.

    Replaces sys.stdout with an in-memory buffer (io.StringIO) while the context
    is active and restores the original stdout on exit (even if an exception is raised).
    Only affects sys.stdout (not sys.stderr). Requires the 'sys' and 'io' modules.
    Intended for use as a generator-based context manager (e.g. `with silence():`).
    """
    sys.stdout, old = io.StringIO(), sys.stdout
    try:
        yield
    finally:
        sys.stdout = old


def dict2channels(sensor_dict):
    """
    Convert a dictionary of sensor data into a 2D array of channels.

    Parameters:
        sensor_dict (dict): A dictionary where keys are sensor names and values are numpy arrays of sensor data of shape (samples, channels).
        Sensor data arrays should have the same number of samples. Each sensor data array may contain NaN values.


    Returns:
        tuple: A tuple containing:
            - all_channels (numpy.ndarray): A 2D array where each column represents a sensor channel with NaNs removed.
            - used_channels (numpy.ndarray): A boolean array indicating which channels were used (True) and which were not
            (False). If a channel is all NaN, it is excluded from the `all_channels` array and is marked as False in this array.
    """
    all_sensors = np.concatenate([v[:, None] for v in sensor_dict.values()], axis=1)
    all_channels = all_sensors.reshape(all_sensors.shape[0], -1)
    used_channels = ~np.isnan(all_channels).all(axis=0)
    all_channels = all_channels[:, used_channels]
    return all_channels, used_channels


def channels2dict(all_values, used_channels, grid, num_channels=3):
    """
    Convert channel values into a dictionary of sensor data.

    Parameters
    ----------
        all_values (np.ndarray): A array of shape (number of samples, number of nan reduced channels) containing the sensor data.
        used_channels (np.ndarray): A boolean array of same length as all_values indicating which channels were used, i.e.
        contained no nans, (True) and which were not (False).
        grid (dict): A dictionary where keys are sensor names and values are their respective positions.
        num_channels (int, optional): The number of channels per sensor. Default is 3.

    Returns
    ----------
        dict: A dictionary where keys are sensor names and values are 2D arrays of shape
            (number of samples, num_channels) containing the sensor data.
    """
    reshaped_values = np.full([all_values.shape[0], used_channels.shape[0]], np.nan)
    reshaped_values[:, used_channels] = all_values
    sensor_dict = {
        name: reshaped_values[:, i * num_channels : (i + 1) * num_channels]
        for i, name in enumerate(grid.keys())
    }
    return sensor_dict


def sig_power(x, root=False, win=None):
    """
    Calculate the signal power of an input array, optionally using a sliding window.
    Args:
        x (np.ndarray): Input signal array.
        root (bool, optional): If True, return the square root of the signal power. Defaults to False.
        win (int, optional): Window size for calculating signal power. If None, the entire signal is used. Defaults to None.
    Returns:
        float: The median signal power across windows, or the entire signal if no window is specified.
    """

    def rmsd(x, root=False):
        """Calculate the root mean square deviation (RMSD) of a signal."""
        # Check if the input is empty or all NaN
        if x.size == 0 or np.all(np.isnan(x)):
            return 0

        # Calculate the column-wise mean, ignoring NaNs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_means = np.nanmean(x, axis=0)

        # If every column‐wise mean is NaN, then there was no data to center.
        if np.all(np.isnan(col_means)):
            return 0

        # Center and compute squared deviations.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Take the mean of the squared diffs, ignoring NaNs.
            var = np.nanmean((x - col_means) ** 2, axis=(0, -1))

        e = np.nanmedian(var)

        return np.sqrt(e) if root else e

    if win is None:
        win = x.shape[0]
    win = int(win)
    # Calculate signal power in windows
    num_windows = max(x.shape[0] // win, 1)
    signal_power = np.nanmedian(
        [rmsd(x[i * win : (i + 1) * win], root=root) for i in range(num_windows)]
    )
    return signal_power


def sanity_check(sensor_dict, time, xlim=[50, 52], ylim=None, savename=None, show_median=True):
    """
    Perform a sanity check on sensor data by plotting the median and 1-sigma bounds for each axis component.

    Parameters
    ----------
        sensor_dict (dict): A dictionary where keys are sensor labels and values are 3D numpy arrays representing sensor data.
        time (numpy.ndarray): A 1D numpy array representing the time points corresponding to the sensor data.
        xlim (tuple): A tuple of two floats representing the time range to plot.
        ylim (tuple, optional): A tuple of two floats representing the y-axis limits for the plot. Default is None, which
            auto-scales the y-axis.
        savename (str, optional): If provided, saves the plot to the specified filename.

    The resulting plot shows the field measurements for each axis over the specified time range.
    """
    # Extract sensor labels and components for each axis
    all_components = np.array([v for v in sensor_dict.values()])
    sensor_labels = [
        np.array(list(sensor_dict.keys()))[
            ~np.isnan(all_components[:, :, i]).all(axis=1)
        ]
        for i in range(all_components.shape[-1])
    ]
    xyz = [
        all_components[~np.isnan(all_components[:, :, i]).all(axis=1), :, i].T
        for i in range(all_components.shape[-1])
    ]

    # Truncate the components based on the time array
    mask = (time >= xlim[0]) & (time <= xlim[1])
    time_truncated = time[mask]
    xyz_truncated = [components[mask, :] for components in xyz]

    # Plot the median with 1-sigma bounds for each component in separate subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    names = ["x", "y", "z"]
    for i, components in enumerate(xyz_truncated):
        times = np.repeat(time_truncated[:, None], len(sensor_labels[i]), axis=1)
        axs[i].plot(times, components, alpha=0.3, label=sensor_labels[i])
        if show_median:
            axs[i].plot(
                time_truncated, np.median(components, axis=1), color="black", label="median"
            )
        axs[i].set_title(names[i])
        axs[i].grid()
        axs[i].legend(title="Sensor:", loc="upper right")
        axs[i].set_ylim(ylim)
    fig.supxlabel("Time [s]")
    fig.supylabel("Field [pT]")
    plt.suptitle("Field Measurements")
    if savename is not None:
        plt.savefig(savename)
        plt.close()
    plt.tight_layout()
    plt.show()


class MutualInformation(torch_nn.Module):
    """
    Computes the mutual information between two input tensors using kernel density estimation.
    """

    def __init__(self, sigma=0.1, num_bins=256, normalize=True):
        """
        Initializes the MutualInformation module.

        Args:
            sigma (float, optional): Standard deviation for Gaussian kernel. Default is 0.1.
            num_bins (int, optional): Number of histogram bins. Default is 256.
            normalize (bool, optional): Whether to normalize mutual information. Default is True.
        """
        super(MutualInformation, self).__init__()

        self.sigma = sigma
        self.num_bins = num_bins
        self.normalize = normalize
        self.epsilon = torch.finfo(torch.float32).eps

        self.bins = torch_nn.Parameter(
            torch.linspace(0, 1, num_bins).float(), requires_grad=False
        )

    def marginalPdf(self, values):
        """
        Computes the marginal probability density function (PDF) for given values using a Gaussian kernel.
        Args:
            values (torch.Tensor): Input tensor of values for which to compute the marginal PDF.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - pdf: Normalized marginal PDF for each value.
                - kernel_values: Raw kernel values before normalization.
        """

        residuals = values - self.bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5 * (residuals / self.sigma).pow(2))

        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
        pdf = pdf / normalization

        return pdf, kernel_values

    def jointPdf(self, kernel_values1, kernel_values2):
        """
        Computes the joint probability density function (PDF) from two sets of kernel values.

        Args:
            kernel_values1 (torch.Tensor): Tensor of kernel values with shape (batch_size, n, m).
            kernel_values2 (torch.Tensor): Tensor of kernel values with shape (batch_size, m, p).

        Returns:
            torch.Tensor: Normalized joint PDF tensor with shape (batch_size, n, p).
        """

        joint_kernel_values = torch.matmul(
            kernel_values1.transpose(1, 2), kernel_values2
        )
        normalization = (
            torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1) + self.epsilon
        )
        pdf = joint_kernel_values / normalization

        return pdf

    def getMutualInformation(self, input1, input2):
        """
        Computes the mutual information between two input tensors.

        The inputs are first normalized to the [0, 1] range. Marginal and joint probability density functions (PDFs) are estimated using kernel methods.
        Entropies for each input and their joint distribution are calculated, and mutual information is derived as:
            MI = H(input1) + H(input2) - H(input1, input2)
        Optionally, the result can be normalized.

        Args:
            input1 (torch.Tensor): First input tensor.
            input2 (torch.Tensor): Second input tensor.

        Returns:
            torch.Tensor: Mutual information value(s) between input1 and input2.
        """
        # normalize inputs
        input1 = 1 * (input1 - input1.min()) / (input1.max() - input1.min())
        input2 = 1 * (input2 - input2.min()) / (input2.max() - input2.min())

        assert input1.shape == input2.shape

        x1 = input1
        x2 = input2

        pdf_x1, kernel_values1 = self.marginalPdf(x1)
        pdf_x2, kernel_values2 = self.marginalPdf(x2)
        pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

        H_x1 = -torch.sum(pdf_x1 * torch.log2(pdf_x1 + self.epsilon), dim=1)
        H_x2 = -torch.sum(pdf_x2 * torch.log2(pdf_x2 + self.epsilon), dim=1)
        H_x1x2 = -torch.sum(pdf_x1x2 * torch.log2(pdf_x1x2 + self.epsilon), dim=(1, 2))

        mutual_information = H_x1 + H_x2 - H_x1x2

        if self.normalize:
            mutual_information = 2 * mutual_information / (H_x1 + H_x2)

        return mutual_information

    def forward(self, input1, input2):
        """Computes the mean mutual information between two input tensors.

        Args:
            input1 (torch.Tensor): First input tensor of shape (B, C, H, W).
            input2 (torch.Tensor): Second input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar representing the mean mutual information.
        """
        return self.getMutualInformation(input1, input2).mean()


class Logger:
    """
    Logger class for tracking and visualizing training metrics and parameters.
    Attributes:
        history (dict): Stores logged values for each metric.
        parameter_labels (list, optional): Labels for parameters to display in plots.
        ylogaxis (list): Metrics to plot on a logarithmic y-axis.
    Methods:
        reset():
            Clears the history of logged metrics.
        log(**kwargs):
            Logs values for given metrics. Each metric is stored as a list.
        plot(savename=None, plt_param=False):
            Plots the logged metrics and parameters.
            - If plt_param is True, plots parameter evolution and gradients.
            - Supports plotting mean, std, max, and min for gradients.
            - Plots other metrics in subplots, with log scale for specified metrics.
            - Optionally saves the plot to a file if savename is provided.
    """

    def __init__(
        self,
        parameter_labels=None,
        ylogaxis=["mse", "regularization", "loss", "condition_number"],
    ):
        self.history = {}
        self.parameter_labels = parameter_labels
        self.ylogaxis = ylogaxis

    def reset(self):
        self.history = {}

    def log(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)

    def plot(self, savename=None, plt_param=False):
        if plt_param:
            if "parameters" in self.history:
                fig, ax = plt.subplots(figsize=(7.11, 3), dpi=200)
                ax.plot(self.history["parameters"])
                ax.set_title("parameters")
                ax.set_xlabel("Iteration")
                ax.grid()
                if (
                    self.parameter_labels
                    and np.array(self.history["parameters"]).shape[-1] <= 12
                ):
                    ax.legend(self.parameter_labels)
                plt.tight_layout()
                plt.show()

            # if "grad" in self.history and self.parameter_labels:
            #     fig, axs = plt.subplots(2, 2, figsize=(7.11, 4), sharex=True, sharey=True)

            #     for i in range(2 * len(self.parameter_labels) // 6):
            #         ax = axs[i % 2, i // 2]
            #         for j in range(3):
            #             ax.plot(
            #                 np.abs(np.array(self.history["grad"])[:, i * 3 + j]),
            #                 label=self.parameter_labels[i * 3 + j],
            #             )
            #         if self.parameter_labels and np.array(self.history["grad"]).shape[-1] <= len(
            #             self.parameter_labels
            #         ):
            #             ax.legend()
            #         ax.grid()
            #         ax.set_yscale("log")
            #     plt.suptitle("grad")
            #     plt.tight_layout()
            #     plt.show()
            if "grad_m" in self.history and "grad_r" in self.history:
                # fig, axs = plt.subplots(2, 2, figsize=(7.11, 4), sharex=True, sharey=True)

                fig, axs = plt.subplots(
                    2, 2, figsize=(7.11, 4), sharex=True, sharey=True
                )
                for i, k in enumerate(["grad_m", "grad_r"]):
                    if k not in self.history:
                        continue

                    grad_ = np.array(self.history[k])
                    if grad_.ndim == 4:
                        axis = (1, -1)
                    elif grad_.ndim == 2:
                        grad_ = grad_.reshape(grad_.shape[0], 1, 2, -1)
                        axis = (1, -1)
                    else:
                        raise NotImplementedError("Unknown grad shape")

                    for j in range(grad_.shape[-2]):
                        ax = axs[i, j]
                        if grad_.ndim == 4:
                            # plot mean, std, max and min grad
                            ax.plot(
                                np.mean(np.abs(grad_[:, :, j]), axis=axis),
                                label=f"mean",
                            )
                            ax.fill_between(
                                np.arange(grad_.shape[0]),
                                np.mean(np.abs(grad_[:, :, j]), axis=axis)
                                - np.std(np.abs(grad_[:, :, j]), axis=axis),
                                np.mean(np.abs(grad_[:, :, j]), axis=axis)
                                + np.std(np.abs(grad_[:, :, j]), axis=axis),
                                alpha=0.2,
                                label=f"std",
                            )
                            ax.plot(
                                np.max(np.abs(grad_[:, :, j]), axis=axis),
                                label=f"max",
                            )
                            ax.plot(
                                np.min(np.abs(grad_[:, :, j]), axis=axis),
                                label=f"min",
                            )
                            ax.set_yscale("log")
                            ax.grid()
                            ax.legend()
                            ax.set_xlabel("Iteration")
                            ax.set_ylabel(
                                f'{["Fetal", "Maternal"][j]} {["Magn. Moment", "Position"][i]}'
                            )

        keys = [
            k
            for k in self.history
            if k != "parameters" and k != "grad" and k != "grad_m" and k != "grad_r"
        ]

        if "loss" in keys:
            keys.remove("loss")
            keys.insert(0, "loss")
        if "lr" in keys:
            keys.remove("lr")
            keys.insert(1, "lr")

        if "mse" in keys:
            keys.remove("mse")

        n = len(keys)
        cols = 3
        rows = int(np.ceil(n / cols))

        fig, axs = plt.subplots(rows, cols, figsize=(7.11, 2 * rows), dpi=200)
        axs = axs.flatten()
        for i, k in enumerate(keys):

            ax = axs[i]
            ax.plot(self.history[k])
            ax.set_title(
                [
                    (
                        "MSE"
                        if l == "mse"
                        else (
                            "Loss"
                            if l == "loss"
                            else (
                                "Learning Rate"
                                if l == "lr"
                                else (
                                    "Regularization"
                                    if l == "regularization"
                                    else (
                                        "Condition Number"
                                        if l == "condition_number"
                                        else l
                                    )
                                )
                            )
                        )
                    )
                    for l in [k]
                ][0]
            )
            ax.set_xlabel("Iteration")
            ax.grid()
            if k in self.ylogaxis:
                ax.set_yscale("log")

        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])
        plt.tight_layout()
        if savename:
            plt.savefig(savename, dpi=500, pad_inches=0)
            plt.close()
        plt.show()


def rot(M, rot=[0, 0, 0], rot_axes="xyz", degrees=True):
    """
    Rotate a matrix M by specified angles.

    Parameters:
        M (array-like): The matrix to be rotated.
        rot (list or array-like, optional): The rotation angles. Default is [0, 0, 0].
        rot_axes (str, optional): The axes about which to rotate. Default is "xyz".
        degrees (bool, optional): If True, the rotation angles are given in degrees. If False, in radians. Default is True.

    Returns:
        numpy.ndarray: The rotated matrix.
    """

    return M @ R.from_euler(rot_axes, rot, degrees=degrees).as_matrix()
