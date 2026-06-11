import logging
import warnings
from sklearn.decomposition import PCA, FastICA
from tqdm import tqdm
import neurokit2 as nk
import matplotlib.pyplot as plt
import numpy as np
import padasip as pa
import pandas as pd
from scipy.optimize import linear_sum_assignment

from fmcg.analysis.ecg_segment import ecg_segment

from ..utils import utils
from ..utils.plotting import plot_utils, plt_config
from ..analysis.hr import detect_hr_outlier

logger = logging.getLogger(__name__)

# =============================================================================
# Main Processing Functions
# =============================================================================


def detect_heartbeats(
    m_hat,
    r_hat,
    fs,
    verbose=True,
    method="subsequent",
    decomposition="ica",
    show_plots=True,
    outlier_kwargs={"win": 4, "threshold": 30},
    savename=None,
    log_dict=None,
    n_trials=1,
    ica_components=3,
    segment_length=None,
    mu=0.001,
    nlms=False,
    consider_mean=False,
    seed=None,
    **kwargs,
):
    """
    Process dipole moments to extract fetal and maternal heartbeat information.
    This function processes dipole moment data to extract fetal and maternal
    components, detect peaks, and calculate heart rates.
    It supports multiple decomposition methods and provides optional visualization.

    Parameters:
    -----------
    m_hat : numpy.ndarray
        A 2D array containing dipole moment data. The first column corresponds
        to fetal data, and the second column corresponds to maternal data.
    r_hat : numpy.ndarray
        A 2D array containing position data. The first column corresponds
        to fetal data, and the second column corresponds to maternal data.
    fs : float
        Sampling frequency of the input data in Hz.
    verbose : bool, optional
        If True, prints detailed information during processing. Default is True.
    method : str, optional
        The method used for processing. Options are:
        - "subsequent": Decomposes components sequentially.
        - "lsap": Matches components using linear assignment.
        Default is "subsequent".
    decomposition : str, optional
        The decomposition method to use. Default is "ica" (Independent Component Analysis).
    show_plots : bool, optional
        If True, displays plots for visualization. Default is True.
    outlier_kwargs : dict, optional
        A dictionary containing parameters for outlier detection. Default is {"win": 4, "threshold": 30}.
    savename : str, optional
        If provided, saves the plots with this filename. Default is None.
    log_dict : dict, optional
        A dictionary to log information during processing. Default is None.
    n_trials : int, optional
        The number of trials for decomposition. Default is 1.
    segment_length : float, optional
        Length of each segment in seconds for processing. If None, processes the entire signal. Default is None.
    ica_components : int, optional
        The number of components to extract using ICA. Default is 3.
    mu : float, optional
        Learning rate for LMS adaptive filter. Default is 0.001.
    nlms : bool, optional
        If True, use normalized LMS. Default is False.
    consider_mean : bool, optional
        If True, consider mean for IBI calculation in subsequent_LMS method. Default is False.

    Returns:
    --------
    result : dict
        A dictionary containing processed results for fetal and maternal components:
        - "dipole_moments": Original dipole moment data.
        - "components": Processed components (if applicable).
        - "peaks": Detected R-peaks in the ECG signal.
        - "hr": Calculated heart rate in beats per minute (bpm).
    """

    ### Processing in segments ###
    if segment_length is not None:
        segment_samples = int(segment_length * fs)
        num_segments = int(np.ceil(m_hat.shape[0] / segment_samples))
        results = []

        # Process each segment
        for i in (
            bar := tqdm(
                range(num_segments), desc="Processing segments", total=num_segments
            )
        ):
            start = i * segment_samples
            end = min((i + 1) * segment_samples, m_hat.shape[0])
            logger.info(
                f"Processing segment {i + 1}/{num_segments} (samples {start/fs}s:{end/fs}s)"
            )
            bar.set_description(
                f"Processing segment {i + 1}/{num_segments} ({start/fs:.2f}s:{end/fs:.2f}s)"
            )
            # Prepare postfix dictionary for tqdm bar
            postfix = {}
            for k in ["fetal", "maternal"]:
                if len(results) > 0:
                    postfix[f"{k}_HR"] = f"{results[-1][k]['hr'].mean():.2f}"
                else:
                    postfix[f"{k}_HR"] = "N/A"
            bar.set_postfix(postfix)

            # Recursive call to process_dipole_moments for each segment
            segment_result = detect_heartbeats(
                m_hat[start:end],
                r_hat[start:end],
                fs,
                verbose=verbose,
                method=method,
                decomposition=decomposition,
                show_plots=False,
                outlier_kwargs=outlier_kwargs,
                savename=None,
                log_dict=log_dict,
                n_trials=n_trials,
                ica_components=ica_components,
                segment_length=None,  # Avoid next level of recursion!
                mu=mu,
                nlms=nlms,
                consider_mean=consider_mean,
                **kwargs,
            )

            # Add segment index for tracking
            for k in segment_result.keys():
                segment_result[k]["segment_idx"] = i
                segment_result[k]["segment_start"] = start

            results.append(segment_result)

        # Store segment results for later beat collection
        combined_result = {}
        for k in ["fetal", "maternal"]:
            combined_result[k] = {
                "dipole_moments": np.concatenate(
                    [r[k]["dipole_moments"] for r in results]
                ),
                "components": np.concatenate([r[k]["components"] for r in results]),
                "peaks": np.concatenate(
                    [r[k]["peaks"] + r[k]["segment_start"] for r in results]
                ),
                "position": np.concatenate([r[k]["position"] for r in results]),
                "outlier": np.concatenate([r[k]["outlier"] for r in results]),
                "segment_results": [
                    r[k] for r in results
                ],  # Store individual segment results
            }

        # Recompute hr
        for k in combined_result.keys():
            combined_result[k]["hr"] = nk.ecg_rate(
                combined_result[k]["peaks"],
                sampling_rate=fs,
                desired_length=len(combined_result[k]["components"]),
            )

        if show_plots:
            plot_component_analysis(combined_result, fs, savename=savename)

        return combined_result

    ### Original processing logic for non-segmented data ###
    result = {
        "fetal": {
            "dipole_moments": m_hat[:, 0],
            "position": r_hat[:, 0],
        },
        "maternal": {
            "dipole_moments": m_hat[:, 1],
            "position": r_hat[:, 1],
        },
    }

    if method == "subsequent":
        result = decompose_subsequently(
            result,
            fs=fs,
            verbose=verbose,
            method=decomposition,
            plot=False,
            log_dict=log_dict,
            n_trials=n_trials,
            ica_components=ica_components,
            consider_mean=True,
            seed=seed,
        )
    elif method == "subsequent_LMS":
        result = decompose_and_LMS(
            result,
            fs=fs,
            n_components=ica_components,
            n_trials=n_trials,
            mu=mu,
            plot=show_plots,
            log_dict=log_dict,
            verbose=verbose,
            nlms=nlms,
            consider_mean=consider_mean,
            seed=seed,
        )
    elif method == "lsap":
        comp = decompose_and_match_components(
            m_hat, method=decomposition, n_components=2, plot=show_plots
        )
        result["fetal"]["components"] = comp[0][:, 0]
        result["maternal"]["components"] = comp[1][:, 1]
    else:
        raise NotImplementedError(f"Method {method} not implemented")

    # Process all dipoles
    for k in result.keys():
        # Invert components to ensure positive peaks
        inv, is_inverted = nk.ecg_invert(
            result[k]["components"], sampling_rate=fs, show=False
        )
        result[k]["components"] = inv

        # Detect peaks in the signal
        filtered = nk.ecg_clean(inv, sampling_rate=fs, method="vg")
        _, peaks_dict = nk.ecg_peaks(
            filtered, sampling_rate=fs, correct_artifacts=False, method="vg"
        )
        result[k]["peaks"] = peaks_dict["ECG_R_Peaks"]
        #result[k]["peaks"] = correct_peaks(np.linalg.vector_norm(result[k]["dipole_moments"], axis=1), result[k]["peaks"], fs=fs, window=20/fs)

        # Compute heart rate
        result[k]["hr"] = nk.ecg_rate(
            result[k]["peaks"], sampling_rate=fs, desired_length=len(inv)
        )
        # if log_dict is not None:
        #     log_dict[f"hr_{k}"] = result[k]["hr"].mean()

        # hr = 60 / (np.diff(result[k]["peaks"]) / fs)
        # tol = kwargs.get("tol", 3)
        # result[k]["peak_selection_mask"] = (np.median(hr) - tol < hr) & (hr < np.median(hr) + tol)

        result[k]["outlier"] = detect_hr_outlier(
            result[k]["peaks"],
            fs,
            win=outlier_kwargs.get("win", 4),
            threshold=outlier_kwargs.get("threshold", 30),
            plot=False,
        )

        logger.info(
            f"{k} - Peaks: {len(result[k]['peaks'])} - Heart Rate: {result[k]['hr'].mean():.2f} bpm"
        )

        if show_plots:
            _plot_segmented_dipole_magnitudes(
                result[k]["dipole_moments"], result[k]["peaks"], fs
            )

    if show_plots:
        plot_component_analysis(result, fs, savename=savename)

    return result


def segment_and_average_heartbeats(
    data_dict,
    fs,
    method="remove_outlier",
    ratio_pre=0.5,
    interval=2.5,
    verbose=True,
    log_dict=None,
    **kwargs,
):
    results = {}
    for key in data_dict.keys():
        # Ensure all values under data_dict[key] are numpy arrays
        for subkey in data_dict[key]:
            if not isinstance(data_dict[key][subkey], np.ndarray):
                data_dict[key][subkey] = np.array(data_dict[key][subkey])

        m_hat_i = 1e3 * data_dict[key]["dipole_moments"].copy()
        peaks = data_dict[key]["peaks"]
        hr = data_dict[key]["hr"][peaks[1:]]
        r_hat_i = data_dict[key].get("position", None)
        is_outlier = data_dict[key]["outlier"][1:]

        if "artifacts" in data_dict[key]:
            is_artifact = np.array(data_dict[key]["artifacts"])[peaks[1:]]
            is_outlier = is_outlier | is_artifact

        if r_hat_i is not None:
            r_hat_i = r_hat_i.copy()

        # Dynamically build the result dictionary for each axis
        result = {}
        axis_labels = [r"M_x", r"M_y", r"M_z"]
        for i, label in enumerate(axis_labels):
            result[label] = {
                "signal": m_hat_i[:, i],
                "peaks": peaks,
                "position": r_hat_i[:, i] if r_hat_i is not None else None,
            }

        # get outlier mask depending on method
        if method == "remove_outlier":
            mask = ~is_outlier
        elif method == "selective_averaging":
            tol = kwargs.get("tol", 3)

            if isinstance(tol, dict):
                tol = tol.get(key, 3)

            # from scipy.stats import mode
            # mask = (mode(hr[~is_outlier])[0] - tol < hr) & (hr < mode(hr[~is_outlier])[0] + tol) & ~is_outlier
            mask = (
                (np.median(hr[~is_outlier]) - tol < hr)
                & (hr < np.median(hr[~is_outlier]) + tol)
                & ~is_outlier
            )

            # Add threshold in index/time dimension
            min_block_length = kwargs.get(
                "min_block_length", 5
            )  # Minimum block length in number of beats
            block_starts = np.where(np.diff(mask.astype(int)) == 1)[0] + 1
            block_ends = np.where(np.diff(mask.astype(int)) == -1)[0] + 1

            if mask[0]:
                block_starts = np.insert(block_starts, 0, 0)
            if mask[-1]:
                block_ends = np.append(block_ends, len(mask))

            for start, end in zip(block_starts, block_ends):
                if end - start < min_block_length:
                    mask[start:end] = False
        else:
            raise NotImplementedError(f"Method {method} not implemented")

        if mask.sum() == 0:
            logger.warning(f"{key}: No heartbeats selected for averaging")
            continue

        # segment and average each component
        for k in result.keys():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=nk.misc.NeuroKitWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)
                result[k]["signal_cleaned"] = nk.ecg_clean(
                    result[k]["signal"], sampling_rate=fs, method="vg"
                )
                _, result[k]["is_inverted"] = nk.ecg_invert(
                    result[k]["signal_cleaned"], sampling_rate=fs, show=False
                )

            beats = ecg_segment(
                result[k]["signal_cleaned"],
                rpeaks=result[k]["peaks"],
                sampling_rate=fs,
                show=False,
                ratio_pre=ratio_pre,
                interval=interval,
            )

            if result[k]["position"] is not None:
                pos = ecg_segment(
                    result[k]["position"],
                    rpeaks=result[k]["peaks"],
                    sampling_rate=fs,
                    show=False,
                    ratio_pre=ratio_pre,
                    interval=interval,
                )

            # remove outliers from beats and position
            beats = dict((k, beats[k]) for k, o in zip(beats.keys(), mask) if o)
            if result[k]["position"] is not None:
                pos = dict((k, pos[k]) for k, o in zip(pos.keys(), mask) if o)

            logger.info(
                f"{k}: Selected {len(beats)} heartbeats for averaging with mean HR {np.mean(hr[mask]):.2f} bpm"
            )

            result[k]["selected_peaks_mask"] = mask

            # perform averaging
            if len(beats) == 0:
                # result[k][f"beats"] = pd.DataFrame()
                result[k][f"mean_beat"] = pd.DataFrame()
                result[k][f"std_beat"] = pd.DataFrame()
            else:
                # result[k][f"beats"] = df = nk.epochs_to_df(beats)
                df = nk.epochs_to_df(beats)
                result[k][f"mean_beat"] = df.groupby("Time")["Signal"].mean()
                result[k][f"std_beat"] = df.groupby("Time")["Signal"].std()

            if result[k]["position"] is not None and len(pos) > 0:
                # result[k][f"seg_position"] = df = nk.epochs_to_df(pos)
                df = nk.epochs_to_df(pos)
                result[k][f"mean_position"] = df.groupby("Time")["Signal"].mean()
                result[k][f"std_position"] = df.groupby("Time")["Signal"].std()

        if log_dict is not None:
            log_dict[f"center_hr_{key}"] = np.round(np.mean(hr[~is_outlier]), 2)
            log_dict[f"selected_hr_{key}"] = np.round(np.mean(hr[mask]), 2)
            log_dict[f"selected_beats_{key}"] = len(peaks[1:][mask])

        results[key] = result

    return results


def segment_and_average_fields(
    data_dict,
    mdl,
    fs,
    W=None,
    axis_mask=None,
    savename=None,
    method="remove_outlier",
    **kwargs,
):
    field_dict = {}
    for i, c in enumerate(data_dict.keys()):
        field_dict[c] = {"mean": [], "std": []}

        B = mdl.forward_model.forward_linear(
            data_dict[c]["position"][:, None, :],
            data_dict[c]["dipole_moments"][:, None, :],
            as_numpy=True,
        )
        if W is not None and axis_mask is not None:
            B = B[:, axis_mask] @ W.T
        else:
            B = B.reshape(data_dict[c]["position"].shape[0], -1)

        fig, ax = plt.subplots(figsize=(7.11, 2.5), dpi=500)
        for k in range(B.shape[-1]):
            peaks = data_dict[c]["peaks"]
            # inv_corr, is_inverted = nk.ecg_invert(B[:, k], sampling_rate=fs, show=False)
            #corrected_peaks = correct_peaks(B[:, k], peaks.copy(), fs, window=0.02, inverted=is_inverted)
            heartbeats = ecg_segment(
                B[:, k],
                rpeaks=peaks,
                sampling_rate=fs,
                show=False,
                ratio_pre=0.5,
                interval=2.5,
            )

            if method == "remove_outlier":
                # if outlier_kwargs is None:
                #     outlier_kwargs = {}
                # is_outlier = detect_hr_outlier(
                #     peaks,
                #     fs,
                #     win=outlier_kwargs.get("win", 4),
                #     threshold=outlier_kwargs.get("threshold", 30),
                #     plot=False,
                # )
                is_outlier = data_dict[c]["outlier"]
                heartbeats = dict(
                    (k, heartbeats[k])
                    for k, o in zip(heartbeats.keys(), is_outlier)
                    if not o
                )

            elif method == "selective_averaging":
                # if outlier_kwargs is None:
                #     outlier_kwargs = {}
                # is_outlier = detect_hr_outlier(
                #     peaks,
                #     fs,
                #     win=outlier_kwargs.get("win", 4),
                #     threshold=outlier_kwargs.get("threshold", 30),
                #     plot=False,
                # )[1:]
                is_outlier = data_dict[c]["outlier"][1:]

                hr = 60 / (np.diff(peaks) / fs)
                tol = kwargs.get("tol", 3)
                mask = (
                    (np.median(hr[~is_outlier]) - tol < hr)
                    & (hr < np.median(hr[~is_outlier]) + tol)
                    & ~is_outlier
                )

                # Add threshold in index/time dimension
                min_block_length = kwargs.get(
                    "min_block_length", 5
                )  # Minimum block length in number of beats
                block_starts = np.where(np.diff(mask.astype(int)) == 1)[0] + 1
                block_ends = np.where(np.diff(mask.astype(int)) == -1)[0] + 1

                if mask[0]:
                    block_starts = np.insert(block_starts, 0, 0)
                if mask[-1]:
                    block_ends = np.append(block_ends, len(mask))

                for start, end in zip(block_starts, block_ends):
                    if end - start < min_block_length:
                        mask[start:end] = False
                heartbeats = dict(
                    (k, heartbeats[k]) for k, o in zip(heartbeats.keys(), mask) if o
                )

                logger.info(f"{k}: Selected {len(heartbeats)} heartbeats for averaging")
                # plt.plot(hr, '.-')
                # plt.plot(np.arange(len(hr))[mask], hr[mask], 'ro')

            df = nk.epochs_to_df(heartbeats)
            mean_heartbeat = df.groupby("Time")["Signal"].mean()
            std_heartbeat = df.groupby("Time")["Signal"].std()
            field_dict[c]["mean"].append(mean_heartbeat)
            field_dict[c]["std"].append(std_heartbeat)

            ax.axvline(x=0, color="grey", linestyle="--")
            # Plot average heartbeat
            ax.plot(
                mean_heartbeat.index * 1e3,
                mean_heartbeat,
                color="k",
                linewidth=1 / np.log2(np.log2(1 + B.shape[-1])),
                zorder=3,
            )
            # ax.fill_between(
            #     mean_heartbeat.index * 1e3,
            #     mean_heartbeat - std_heartbeat,
            #     mean_heartbeat + std_heartbeat,
            #     alpha=0.2,
            #     zorder=2,
            # )

            ax.grid(True)
            ax.grid(
                which="minor", linestyle=":", linewidth=0.5
            )  # Add gridlines between major ticks
            ax.minorticks_on()  # Enable minor ticks without adding labels
            ax.set_title(rf"{c.capitalize()}")
            ax.set_xlabel("Time [ms]")
            ax.set_ylabel(r"Reconstructed Field [pT]")
            if savename is not None:
                savename_ = savename.replace(".pdf", f"_{c}.pdf")
                plt.savefig(
                    savename_, dpi=fig.dpi, bbox_inches="tight", pad_inches=0.01
                )
                plt.close()
    return field_dict


# =============================================================================
# Auxiliary Functions
# =============================================================================


def correct_peaks(sig, peaks, fs, window=0.02, inverted=False):
    """
    Adjusts the positions of detected peaks in a signal by finding the closest local
    maximum (or minimum if inverted) within a specified window.

    Parameters:
        sig (array-like): The input signal in which peaks are to be corrected.
        peaks (array-like): Indices of the detected peaks in the signal.
        fs (float): Sampling frequency of the signal, used to calculate the window size.
        window (float, optional): The time window (in seconds) around each peak to search
            for the local maximum or minimum. Default is 0.02 seconds.
        inverted (bool, optional): If True, the function searches for the local minimum
            instead of the local maximum. Default is False.

    Returns:
        array-like: The corrected indices of the peaks in the signal.
    """

    for i, peak in enumerate(peaks):
        # Find the closest local maximum within the window
        if i == 0:
            l = 0
        else:
            l = peaks[i - 1] + int(window * fs)
        r = peaks[i] + int(window * fs)
        if len(sig[l:r]) > 0:
            if inverted:
                # Find the local minimum in the inverted signal
                peaks[i] = np.argmin(sig[l:r]) + l
            else:
                # Find the local maximum in the signal
                peaks[i] = np.argmax(sig[l:r]) + l

    return peaks


def decompose_and_match_components(m_hat, method="pca", n_components=2, plot=True):
    """
    Decomposes fetal and maternal signals using PCA or ICA, matches corresponding components,
    and optionally plots the results. Matching is performed by solving the linear sum assignment
    constructed of the correlation coefficients.

    Parameters:
    -----------
    m_hat : numpy.ndarray
        A 2D array where the first column represents the fetal signal and the second column
        represents the maternal signal.
    method : str, optional
        The decomposition method to use. Options are "pca" (Principal Component Analysis)
        or "ica" (Independent Component Analysis). Default is "pca".
    n_components : int, optional
        The number of components to extract during decomposition. Default is 2.
    plot : bool, optional
        If True, plots the separated components and their matches. Default is True.

    Returns:
    --------
    f : numpy.ndarray
        A 2D array containing the matched fetal components. Each column corresponds to a
        matched component.
    m : numpy.ndarray
        A 2D array containing the matched maternal components. Each column corresponds to a
        matched component.

    Notes:
    ------
    The returned arrays `fetal` and `maternal` have shape `(n_samples, n_components)`.
    """

    # Normalize the fetal and maternal signals
    m_f = m_hat[:, 0] / np.linalg.norm(m_hat[:, 0])
    m_m = m_hat[:, 1] / np.linalg.norm(m_hat[:, 1])

    # decompose each signal
    if method == "pca":
        decomposition = PCA(n_components=n_components)
    elif method == "ica":
        decomposition = FastICA(n_components=n_components)
    else:
        raise NotImplementedError(f"Method {method} not implemented")

    fetal = np.zeros((m_f.shape[0], 2))
    maternal = np.zeros((m_m.shape[0], 2))
    for i, (m, name) in enumerate(zip([m_f, m_m], ["Fetal", "Maternal"])):
        # decompose
        components = decomposition.fit_transform(m)

        if plot:
            fig = plt.figure(figsize=(7.11, 2))
            plt.plot(components)
            plt.title(f"{name} Components")
            plt.show()

        # compute correlations to both reconstructed signals
        # find corresponding components between fetal and maternal signals
        c_corr_f = np.abs(
            np.corrcoef(components, m_f, rowvar=False)[:n_components, n_components:]
        ).mean(axis=1)
        c_corr_m = np.abs(
            np.corrcoef(components, m_m, rowvar=False)[:n_components, n_components:]
        ).mean(axis=1)
        # c_corr_f /= c_corr_f.sum()
        # c_corr_m /= c_corr_m.sum()
        logger.debug(f"Correlation of {name} to f: {c_corr_f}")
        logger.debug(f"Correlation of {name} to m: {c_corr_m}")
        corr = np.stack([c_corr_f, c_corr_m], axis=1)
        logger.debug(f"Correlation of {name} to other: {corr}")

        # solve assignment problem
        row_ind, col_ind = linear_sum_assignment(corr, maximize=True)
        # col_ind = np.argmax(corr, axis=0)
        fetal[:, i] = components[:, col_ind[0]]
        maternal[:, i] = components[:, col_ind[1]]

        logger.info(
            f"Fetal signal {col_ind[0]} and maternal signal {col_ind[1]} in {name} components"
        )

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(7.11, 5), sharex=True)
        ax[0].plot(fetal, label="Fetal Components")
        ax[1].plot(maternal, label="Maternal Components")
        ax[0].set_title("Fetal Assigned Components")
        ax[1].set_title("Maternal Assigned Components")
    return fetal, maternal


def _ica(
    result,
    key,
    fs,
    in_key="dipole_moments",
    n_components=2,
    n_trials=1,
    consider_mean=True,
    plot=False,
    log_dict=None,
    seed=None,
):
    best_components = None
    best_ibi_std = np.ones(n_components) * np.inf
    if consider_mean:
        best_ibi_mean = np.ones(n_components) * np.inf

    # Create seeded RNG if seed provided, otherwise use global numpy random state
    rng = np.random.RandomState(seed) if seed is not None else np.random

    for _ in range(n_trials):  # Perform decomposition n_trials times
        fastica = FastICA(
            n_components=n_components, random_state=rng.randint(0, 10000)
        )
        components = fastica.fit_transform(result[key][in_key])

        num_peaks = []
        ibi_std = []
        ibi_mean = []
        for c_ in components.T:
            # Clean and detect peaks
            inv, is_inv = nk.ecg_invert(c_, sampling_rate=fs)
            filtered = nk.ecg_clean(inv, sampling_rate=fs, method="vg")
            _, peaks_dict = nk.ecg_peaks(filtered, sampling_rate=fs, method="vg")
            peaks = peaks_dict["ECG_R_Peaks"]
            #peaks = correct_peaks(np.linalg.vector_norm(result[key]["dipole_moments"], axis=1), peaks, fs=fs, window=20/fs)

            # num_peaks.append(peaks_dict["ECG_R_Peaks"].shape[0])
            ibi = np.diff(peaks)

            std = np.std(ibi)
            std = np.nan_to_num(std, nan=np.inf)
            if std == 0:
                std = np.inf
            ibi_std.append(std)

            if consider_mean:
                mean = np.mean(ibi)
                mean = np.nan_to_num(mean, nan=np.inf)
                if mean == 0:
                    mean = np.inf
                ibi_mean.append(mean)

        # Check if this run has the lowest ibi_std
        if min(best_ibi_std) > min(ibi_std):
            logger.debug(f"IBI std: {ibi_std}")
            best_ibi_std = ibi_std
            if consider_mean:
                best_ibi_mean = ibi_mean
            best_components = components

        if plot and consider_mean:
            fig, ax = plt.subplots(
                n_components, 1, figsize=(7.11, 2 * n_components), sharex=True
            )
            for i in range(n_components):
                ax[i].plot(
                    components[:, i],
                    label=f"Component {i + 1} ({ibi_mean[i]:.2f} +/- {ibi_std[i]:.2f})",
                )
                ax[i].set_title(
                    f"{key.capitalize()} Component {i + 1} - IBI Mean: {ibi_mean[i]:.2f} +/- IBI Std: {ibi_std[i]:.2f}"
                )
                ax[i].grid()
                ax[i].legend()
            plt.tight_layout()
            plt.show()

    if key is not None:
        logger.info(f"{key.capitalize()} - Best IBI Std: {best_ibi_std}")

    # select component
    if consider_mean:
        # Select the two components with the lowest IBI std
        lowest_std_indices = np.argsort(best_ibi_std)[:2]
        # Get their IBI means
        selected_means = np.array(best_ibi_mean)[lowest_std_indices]
        selected_indices = lowest_std_indices

        # Define realistic HR region (e.g., 40-200 bpm, convert to sample)
        min_ibi = 60 * fs / 300  # max HR
        max_ibi = 60 * fs / 20  # min HR

        # Filter out components outside realistic IBI range
        valid = (selected_means > min_ibi) & (selected_means < max_ibi)
        if valid.any():
            logger.debug(
                f"Valid components found within realistic IBI range: {selected_means[valid]}"
            )
            if len(valid) > 1 and np.diff(selected_means[valid]) < 5:
                # If components are too similar, select the first one (with lower std)
                chosen_idx = selected_indices[valid][0]
                logger.debug(
                    f"! Components {selected_indices[valid]} have similar IBI means: {selected_means[valid]}"
                )
            else:
                # Select the component with the highest IBI mean among valid ones
                op = np.argmax if key == "maternal" else np.argmin
                chosen_idx = selected_indices[valid][op(selected_means[valid])]
            result[key]["components"] = best_components[:, chosen_idx]
            return result

        else:
            logger.warning(
                f"Fallback: No valid components found within realistic IBI range."
            )
            if log_dict is not None:
                log_dict[f"note"] = (
                    log_dict.get("note", "")
                    + f"No valid {key} components found within realistic IBI range."
                )

    # Use the best components from the run with the lowest ibi_std
    logger.info(f"{key.capitalize()} - Best IBI Std: {best_ibi_std}")
    result[key]["components"] = best_components[:, np.argmin(best_ibi_std)]
    return result


def decompose_and_LMS(
    result,
    fs=1000,
    n_components=2,
    n_trials=1,
    mu=0.001,
    plot=True,
    log_dict=None,
    verbose=True,
    nlms=False,
    consider_mean=False,
    seed=None,
):

    def _apply_lms_filter(d, x, mu, plot=False, n=9):
        if plot:
            fig, axs = plt.subplots(3, 3, figsize=(10, 5), sharex=True)

        out = np.zeros_like(d, dtype=np.float64)
        for i in range(d.shape[1]):
            x = pa.input_from_history(x, n=n)[:, :, 0]
            # pad zeros
            x = np.concatenate([x, np.zeros((n - 1, x.shape[1]))], axis=0)
            if nlms:
                f = pa.filters.FilterNLMS(n=n, mu=mu, w="zeros")
            else:
                f = pa.filters.FilterLMS(n=n, mu=mu, w="zeros")
            y, e, w = f.run(d[:, i], x)
            out[:, i] = e.astype(np.float64)

            if plot:
                # Left: Plot target signal and error
                axs[i, 0].plot(d[:, i], label=f"Target {i}")
                axs[i, 0].plot(e, label=f"Error {i}")
                axs[i, 0].set_title(f"Target & Error {i}")
                # axs[i, 0].legend(loc="upper right")
                axs[i, 0].grid()
                axs[i, 0].set_xlim([3000, 4000])

                # Right: Plot filter output and input x
                axs[i, 1].plot(y, label=f"Output {i}")
                axs[i, 1].set_title(f"Filter Output {i}")
                axs[i, 1].legend(loc="upper right")
                axs[i, 1].grid()

                axs[i, 2].plot(x[:, 0], label=f"Input x {i}")
                axs[i, 2].set_title(f"Input x {i}")
                axs[i, 2].legend(loc="upper right")
                axs[i, 2].grid()
        return out

    if isinstance(mu, float):
        mu = [mu, mu]

    # get maternal reference by ICA
    result = _ica(
        result,
        "maternal",
        fs=fs,
        n_components=n_components,
        n_trials=n_trials,
        plot=plot,
        log_dict=log_dict,
        consider_mean=consider_mean,
        seed=seed,
    )

    result["fetal"]["dipole_moments"] = _apply_lms_filter(
        d=result["fetal"]["dipole_moments"].copy(),
        x=result["maternal"]["components"][:, None].copy(),
        mu=mu[0],
        plot=plot,
    )

    result = _ica(
        result,
        "fetal",
        fs=fs,
        in_key="dipole_moments",
        n_components=n_components,
        n_trials=n_trials,
        seed=seed,
        plot=plot,
        log_dict=log_dict,
        consider_mean=consider_mean,
    )

    # result["maternal"]["dipole_moments"] = _apply_lms_filter(
    #     d=result["maternal"]["dipole_moments"].copy(), x=result["fetal"]["dipole_moments"].copy(), mu=mu[1], plot=plot
    # )

    result["maternal"]["dipole_moments"] = _apply_lms_filter(
        d=result["maternal"]["dipole_moments"].copy(),
        x=result["fetal"]["components"][:, None].copy(),
        mu=mu[1],
        plot=plot,
    )

    return result


def decompose_subsequently(
    result,
    method="ica",
    fs=1000,
    verbose=True,
    plot=True,
    log_dict=None,
    n_trials=1,
    ica_components=3,
    consider_mean=True,
    seed=None,
):
    """
    Decomposes the dipole moments subsequently using the specified method.

    Parameters:
        result (dict): A dictionary containing the dipole moments for "fetal" and "maternal" keys.
                       Each key should map to another dictionary with a "dipole_moments" key containing
                       the data to be decomposed.
        method (str, optional): The decomposition method to use. Options are:
                                - "ica" (Independent Component Analysis)
                                - "pca" (Principal Component Analysis)
                                Default is "ica".
        fs (int, optional): The sampling frequency in Hz. Default is 1000.
        verbose (bool, optional): If True, prints detailed information about the decomposition process.
                                  Default is True.
        plot (bool, optional): If True, plots the components and results of the decomposition.
                               Default is True.
        log_dict (dict, optional): A dictionary to log information during the decomposition process.
                                Default is None.
        n_trials (int, optional): The number of trials for decomposition. Default is 1.
        ica_components (int, optional): The number of components to extract using ICA. Default is 3.


    Returns:
        dict: The updated result dictionary with an additional "components" key for each entry,
              containing the selected or reduced components.

    Notes:
        - For the "ica" method, the function selects the component with the highest correlation
          to the other dipole moments and the most peaks detected.
        - A warning is printed if the component with the most peaks does not match the one with
          the highest correlation.
        - For the "pca" method, the function reduces the data to a single principal component.
    """
    if method == "ica":
        for key in result.keys():
            result = _ica(
                result,
                key,
                fs=fs,
                n_components=ica_components,
                n_trials=n_trials,
                plot=plot,
                log_dict=log_dict,
                consider_mean=consider_mean,
                seed=seed,
            )

    elif method == "pca":
        # Perform PCA / dim reduction
        for key in result.keys():
            pca = PCA(n_components=1)
            result[key]["components"] = pca.fit_transform(result[key]["dipole_moments"])[
                :, 0
            ]
    else:
        raise NotImplementedError(f"Method {method} not implemented")

    if plot:
        plot_pca_components(result)
    return result


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_averaged_beats(heartbeats_dict, suffix="", savename=None):
    for component in heartbeats_dict.keys():

        fig, ax = plt.subplots(figsize=(7.11, 2.5))
        for key in heartbeats_dict[component].keys():
            # skip if df is empty
            if (
                key not in heartbeats_dict[component]
                or f"mean_beat{suffix}" not in heartbeats_dict[component][key]
            ):
                continue

            df = heartbeats_dict[component][key][f"mean_beat{suffix}"].copy()
            df.index = df.index * 1e3
            df_std = heartbeats_dict[component][key][f"std_beat{suffix}"].copy()
            df_std.index = df_std.index * 1e3

            logger.debug(df.shape)
            # # flip if inverted
            # if heartbeats_dict[component][key]["is_inverted"]:
            #     df = -df
            #     df_std = -df_std

            xmin, xmax = df.index.min(), df.index.max()
            ax.hlines(0, xmin, xmax, color="k", linestyle="--")
            ax.plot(df, label=rf"${key.lower()}$")
            ax.fill_between(
                df.index,
                df - df_std,
                df + df_std,
                alpha=0.2,
            )
        # plot mean ica
        # ax.plot(df.index, heartbeats_dict[component]["M_x"]["mean_ica_component"], label="Mean ICA", color="C3")
        # magnitude

        # Use the index of the first signal as the reference index
        reference_index = heartbeats_dict[component]["M_x"][f"mean_beat{suffix}"].index

        # Reindex all mean_beat_corrected DataFrames to the reference index
        aligned_means = pd.concat(
            [
                v[f"mean_beat{suffix}"].reindex(reference_index, method="nearest")
                for k, v in heartbeats_dict[component].items()
            ],
            axis=1,
        ).fillna(0)

        # Calculate the magnitude
        magnitude = np.linalg.norm(aligned_means.values, axis=1)
        ax.plot(
            aligned_means.index * 1000,
            magnitude,
            label="Magnitude",
            color="k",
            linestyle=":",
        )
        # ax.set_title(component)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel(r"Mean Dipole Moment [nAm$^2$]")
        ax.legend(ncol=4, loc="lower left")
        ax.grid()
        ax.grid(
            which="minor", linestyle=":", linewidth=0.5
        )  # Add gridlines between major ticks
        ax.minorticks_on()  # Enable minor ticks without adding labels
        if savename is not None:
            savename_ = savename.replace(".pdf", f"_{component}.pdf")
            plt.savefig(savename_, dpi=fig.dpi, bbox_inches="tight", pad_inches=0.01)
            plt.close()


def plot_pca_components(result):
    plt.figure(figsize=(7.11, 2))
    for k, v in result.items():
        v_ = v["components"] / np.linalg.norm(v["components"], axis=0, keepdims=True)
        plt.plot(v_, label=k)
    plt.legend()
    plt.title("Reduced Components")
    # plt.show()


def plot_component_analysis(result, fs, savename=None):
    time_ = np.arange(len(result["fetal"]["components"])) / fs
    fig, axes = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(7.11, 4),
        gridspec_kw={"height_ratios": [2, 2, 1.5], "hspace": 0.3},
        dpi=500,
    )

    for i, k in enumerate(result.keys()):

        outlier = result[k]["outlier"]
        if outlier.any():
            axes[i].fill_between(
                time_[result[k]["peaks"]],
                result[k]["components"].min(),
                result[k]["components"].max(),
                where=outlier,
                alpha=0.25,
                zorder=0,
                interpolate=False,
                label="Outlier",
                color="grey",
            )
            axes[i].legend(loc="upper right")

        axes[i].plot(time_, result[k]["components"], label="PCA", color="C0")
        axes[i].plot(
            time_[result[k]["peaks"]],
            result[k]["components"][result[k]["peaks"]],
            "rx",
            markersize=5,
            label="R-peaks",
        )
        axes[i].set_ylabel("Amplitude")
        axes[i].set_title(f"{k.capitalize()} Component")
        axes[i].grid(True)

        axes[2].plot(time_, result[k]["hr"], label=k.capitalize())
        axes[2].set_title("Heart Rate")
        axes[2].set_ylabel("Heart Rate [bpm]")
        axes[2].grid(True)
        axes[2].set_xlabel("Time [s]")
        axes[2].legend(loc="upper right")
        # set ylim based on min max hr over both k
        axes[2].set_ylim(
            [
                np.min([result["fetal"]["hr"].min(), result["maternal"]["hr"].min()])
                - 10,
                np.max([result["fetal"]["hr"].max(), result["maternal"]["hr"].max()])
                + 10,
            ]
        )

    if savename is not None:
        plt.savefig(savename, dpi=fig.dpi, pad_inches=0.01, bbox_inches="tight")
    # plt.tight_layout()
    if savename is None:
        plt.show()
    else:
        plt.close()


def plot_segmented_dipole_magnitudes(beat_dict, suffix="", plot_3d=True, savename=None):

    for key in beat_dict.keys():

        mean_sig = np.stack(
            [beat_dict[key][k]["mean_beat"] for k in beat_dict[key].keys()], axis=0
        ).T

        if mean_sig.shape[0] == 0:
            logger.warning(f"Empty mean signal for {key}")
            continue

        if plot_3d:
            # Create a separate figure for the 3D plot
            fig_3d = plt.figure(figsize=(3.5, 2), dpi=500)
            ax_3d = fig_3d.add_subplot(111, projection="3d")

            # reference coordinates legend
            image = plt.imread(plot_utils.COORDINATE_IMAGE)
            image = np.rot90(image, k=0)
            ax_ = plt.axes(
                [0.0, 0.75, 0.25, 0.25], frameon=True, zorder=999
            )  # Change the numbers in this array to position your image [left, bottom, width, height])
            ax_.imshow(image)
            ax_.axis("off")

            ibi = np.diff(beat_dict[key]["M_x"]["peaks"])
            left = int(len(mean_sig) / 2 - ibi.mean() * 0.5)
            right = int(len(mean_sig) / 2 + ibi.mean() * (1 - 0.5))

            ax_3d.plot(
                mean_sig[left:right, 0],
                mean_sig[left:right, 2],
                mean_sig[left:right, 1],
                color="tab:red",
            )
            ax_3d.set_box_aspect([1, 1, 1])
            plot_utils.set_axes_equal(ax_3d)

            ax_3d.plot(
                mean_sig[left:right, 0],
                mean_sig[left:right, 1],
                zs=ax_3d.get_ylim()[1] * 1.05,
                zdir="y",
                color="gray",
                linestyle="-",
                alpha=0.3,
                axlim_clip=False,
            )
            ax_3d.plot(
                mean_sig[left:right, 0],
                mean_sig[left:right, 2],
                zs=ax_3d.get_zlim()[0] * 0.95,
                zdir="z",
                color="gray",
                linestyle="-",
                alpha=0.3,
                axlim_clip=False,
            )
            ax_3d.plot(
                mean_sig[left:right, 2],
                mean_sig[left:right, 1],
                zs=ax_3d.get_xlim()[1] * 1.05,
                zdir="x",
                color="gray",
                linestyle="-",
                alpha=0.3,
                axlim_clip=False,
            )
            ax_3d.scatter(0, 0, 0, c="k", marker="x", label="Origin")

            ax_3d.set_ylabel(r"Z", labelpad=-2)
            ax_3d.set_xlabel(r"X", labelpad=-2)
            ax_3d.set_zlabel(r"Y", labelpad=-2)
            ax_3d.tick_params(pad=0)
            ax_3d.view_init(azim=210, elev=30)
            ax_3d.zaxis.labelpad = -2
            # fig_3d.tight_layout()
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0.15)
            if savename is not None:
                savename3d = savename.replace(".pdf", f"_3d_{key}.pdf")
                plt.savefig(savename3d, dpi=fig_3d.dpi, pad_inches=0.01)
                plt.close()

        # Create the 2D plots
        xlim = [-np.inf, np.inf]
        fig, axs = plt.subplots(
            1, 3, figsize=(7.11, 2), sharex=True, sharey=True, dpi=500
        )
        for i, k in enumerate(beat_dict[key].keys()):
            ax = axs[i]

            mean_heartbeat = beat_dict[key][k][f"mean_beat{suffix}"]
            std_heartbeat = beat_dict[key][k][f"std_beat{suffix}"]
            df = beat_dict[key][k][f"beats{suffix}"]
            df_pivoted = df.pivot(index="Time", columns="Label", values="Signal")

            xmin, xmax = (
                mean_heartbeat.index.min() * 1e3,
                mean_heartbeat.index.max() * 1e3,
            )
            ax.hlines(0, xmin, xmax, color="k", linestyle="--")
            ax.axvline(x=0, color="grey", linestyle="--")

            ax.plot(
                mean_heartbeat.index * 1e3,
                mean_heartbeat.values,
                color="tab:red",
                linewidth=1,
                label="Mean",
                zorder=3,
            )
            ax.fill_between(
                mean_heartbeat.index * 1e3,
                mean_heartbeat.values - std_heartbeat.values,
                mean_heartbeat.values + std_heartbeat.values,
                color="tab:red",
                alpha=0.2,
                label="Std",
                zorder=2,
            )
            alpha = min(1 / np.log2(np.log2(1 + df_pivoted.shape[1])), 1)
            ax.plot(
                df_pivoted.index * 1e3,
                df_pivoted.values,
                color="grey",
                linewidth=alpha,
                alpha=alpha,
                zorder=1,
            )
            ax.set_title(rf"${k}$")
            ax.set_xlabel("Time [ms]")

            # if i != 0:
            #     ax.set_yticks([])

            if mean_heartbeat.index[-1] < xlim[1]:
                xlim[1] = mean_heartbeat.index[-1]
            if mean_heartbeat.index[0] > xlim[0]:
                xlim[0] = mean_heartbeat.index[0]
            ax.grid()
            ax.set_xlim(xlim[0] * 1e3, xlim[1] * 1e3)
        axs[0].set_ylabel(rf"Dipole Moment [nAm$^2$]")
        axs[1].legend(
            loc="upper center", bbox_to_anchor=(0.5, 1.35), fancybox=False, ncol=2
        )
        plt.subplots_adjust(wspace=0.05)
        if savename is not None:
            savename2d = savename.replace(".pdf", f"_{key}.pdf")
            plt.savefig(savename2d, dpi=fig.dpi, bbox_inches="tight", pad_inches=0.01)
            plt.close()


def _plot_segmented_dipole_magnitudes(m_hat_i, peaks, fs):
    m_hat_i = 1e3 * m_hat_i.copy()

    # fig, axs = plt.subplots(
    #     1,
    #     5,
    #     figsize=(7.11, 2),
    #     sharex=True,
    #     sharey=False,
    #     dpi=500,
    #     gridspec_kw={"width_ratios": [2, 0.05, 1, 1, 1], "height_ratios": [2], "wspace": .1},
    # )

    fig = plt.figure(figsize=(7.11, 2), dpi=500)
    gs = fig.add_gridspec(1, 5, width_ratios=[2.5, 0.1, 1, 1, 1], wspace=0.1)

    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])
    ax4 = fig.add_subplot(gs[0, 4])

    axs = [ax0, ax1, ax2, ax3, ax4]
    axs[1].set_visible(False)
    labels = [r"$_x$", r"$_y$", r"$_z$"]

    mean_sig = []
    clean_sig = []
    for i in range(m_hat_i.shape[1]):
        ax = axs[i + 2]

        try:
            clean_m = nk.ecg_clean(m_hat_i[:, i], sampling_rate=fs, method="vg")
            clean_sig.append(clean_m)
            _, is_inverted = nk.ecg_invert(clean_m, sampling_rate=fs, show=False)
            # corrected_peaks = correct_peaks(clean_m, peaks.copy(), fs, window=0.02, inverted=is_inverted)
            heartbeats = ecg_segment(
                clean_m,
                rpeaks=peaks,
                sampling_rate=fs,
                show=False,
                ratio_pre=0.35,
                interval=1,
            )

            df = nk.epochs_to_df(heartbeats)
            # Average heartbeat
            mean_heartbeat = df.groupby("Time")["Signal"].mean()
            mean_sig.append(mean_heartbeat)
            df_pivoted = df.pivot(index="Time", columns="Label", values="Signal")

            ax.axvline(x=0, color="grey", linestyle="--")
            # Plot average heartbeat
            ax.plot(
                mean_heartbeat.index,
                mean_heartbeat,
                color="tab:red",
                linewidth=1,
                label="Average",
                zorder=3,
            )
            # Alpha of individual beats decreases with more heartbeats
            alpha = 1 / np.log2(np.log2(1 + df_pivoted.shape[1]))
            # Plot all heartbeats
            ax.plot(df_pivoted, color="grey", linewidth=alpha, alpha=alpha, zorder=2)
        except Exception as e:
            logger.error(f"Error processing signal {i}: {e}")

        ax.set_title(rf"M{labels[i]}")
        ax.set_xlabel("Time [s]")

        if i != 0:
            ax.set_yticks([])

    axs[2].set_ylabel(rf"Dipole Moment [nAm$^2$]")
    axs[-1].legend(loc="upper right")

    # Find the largest starting and smallest ending index across all mean_heartbeat DataFrames
    start_indices = [mean.index[0] for mean in mean_sig]
    end_indices = [mean.index[-1] for mean in mean_sig]
    max_start = max(start_indices)
    min_end = min(end_indices)

    # Create a common index range with evenly spaced values
    common_index = np.linspace(
        max_start, min_end, num=min(len(mean.index) for mean in mean_sig)
    )

    # Interpolate all mean_heartbeat DataFrames to the common index
    truncated_mean_sig = [
        mean.reindex(common_index, method="nearest").interpolate() for mean in mean_sig
    ]

    # Stack all truncated mean_heartbeat DataFrames into a single array
    mean_sig = np.stack(truncated_mean_sig, axis=1)

    # PLOT CARDIAC LOOP
    # axs[0].remove()
    # ax = fig.add_subplot(1, 4, 1, projection="3d")
    ax = axs[0]
    # reference coordinates legend
    image = plt.imread(plot_utils.COORDINATE_IMAGE)
    image = np.rot90(image, k=0)
    ax_ = plt.axes(
        [0.00, 0.9, 0.2, 0.2], frameon=True, zorder=999
    )  # Change the numbers in this array to position your image [left, bottom, width, height])
    ax_.imshow(image)
    ax_.axis("off")

    # 3D plot on the left, spanning the top three rows
    ax.plot(
        mean_sig[:, 0],
        mean_sig[:, 2],
        mean_sig[:, 1],
        color="tab:red",
    )
    # # Plot all heartbeats
    # ax.plot(clean_sig[0], clean_sig[2], clean_sig[1], color="k", linewidth=alpha, alpha=alpha, zorder=2, linestyle=":")

    ax.set_box_aspect([1, 1, 1])
    plot_utils.set_axes_equal(ax)

    # Plot projections on the side planes
    ax.plot(
        mean_sig[:, 0],
        mean_sig[:, 1],
        zs=ax.get_ylim()[1] * 1.05,
        zdir="y",
        color="gray",
        linestyle="-",
        alpha=0.3,
        axlim_clip=False,
    )
    ax.plot(
        mean_sig[:, 0],
        mean_sig[:, 2],
        zs=ax.get_zlim()[0] * 0.95,
        zdir="z",
        color="gray",
        linestyle="-",
        alpha=0.3,
        axlim_clip=False,
    )
    ax.plot(
        mean_sig[:, 2],
        mean_sig[:, 1],
        zs=ax.get_xlim()[1] * 1.05,
        zdir="x",
        color="gray",
        linestyle="-",
        alpha=0.3,
        axlim_clip=False,
    )
    # Plot origin
    ax.scatter(0, 0, 0, c="k", marker="x", label="Origin")

    ax.set_ylabel(r"Z", labelpad=-2)
    ax.set_xlabel(r"X", labelpad=-2)
    ax.set_zlabel(r"Y", labelpad=-2)
    ax.tick_params(pad=0)
    #
    ax.view_init(azim=210, elev=30)
    ax.zaxis.labelpad = -2
    gs.update(left=0.0, bottom=0.0, top=0.95, right=0.99, wspace=0.05, hspace=0)

    # fig.tight_layout()
    # if savename is not None:
    #     plt.savefig(savename, dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
    #     plt.show()


