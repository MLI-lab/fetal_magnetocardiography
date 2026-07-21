"""
Spatial filtering methods for fMCG signal separation.

This module provides various spatial filtering techniques for separating
physiological signals (e.g., maternal and fetal cardiac signals) using
template-based projection operators.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from fmcg.spatial_filtering.construct_template import construct_template


def ensure_row(vector):
    """
    Ensure vector is a row vector (2D with shape (1, n) or (k, n)).

    Parameters
    ----------
    vector : ndarray
        Input vector or matrix

    Returns
    -------
    ndarray
        Row vector (view, not copy)
    """
    vector = np.atleast_2d(vector)
    if vector.shape[1] == 1:
        return vector.T
    return vector


def minimum_norm_projector(topography, reg_power=1e-2, nullspace=False):
    """
    Minimum norm projection operator.

    Projects data onto (nullspace=True) or into (nullspace=False) the subspace
    spanned by the topography using minimum norm regularization.

    Parameters
    ----------
    topography : ndarray, shape (n_timepoints, n_channels) or (n_channels,)
        Spatial template or single topography vector
    reg_power : float, optional
        Regularization parameter as fraction of signal power (default: 0.01)
    nullspace : bool, optional
        If True, project onto nullspace; if False, project into subspace (default: False)

    Returns
    -------
    P : ndarray, shape (n_channels, n_channels)
        Projection operator matrix

    Reference
    ---------
    Wilson, J.D., and Haueisen, J. (2017). Separation of Physiological Signals
    Using Minimum Norm Projection Operators. IEEE Transactions on Biomedical
    Engineering, 64(4), 904-16. https://doi.org/10.1109/TBME.2016.2582643
    """
    topography = ensure_row(topography)

    # Compute covariance-like matrix: (n_channels, k) @ (k, n_channels) = (n_channels, n_channels)
    T = topography.T @ topography
    power = np.trace(T)
    P = T @ np.linalg.pinv(T + power * reg_power * np.eye(topography.shape[1]))

    if nullspace:
        P = np.eye(P.shape[0]) - P

    return P


def nullspace_projector(topography, **kwargs):
    """
    Nullspace projection operator (orthogonal projection).

    Projects data onto the nullspace (orthogonal complement) of the topography.
    Equivalent to minimum_norm_projector with nullspace=True.

    Parameters
    ----------
    topography : ndarray, shape (n_timepoints, n_channels) or (n_channels,)
        Spatial template to project out
    **kwargs : optional
        Additional arguments passed to minimum_norm_projector (e.g., reg_power)

    Returns
    -------
    P : ndarray, shape (n_channels, n_channels)
        Nullspace projection operator
    """
    return minimum_norm_projector(topography, **kwargs, nullspace=True)


def covariance_projector(topography, data, reg=0.0):
    """
    Covariance-based projection (LMMSE/Wiener filter).

    Linear Minimum Mean-Square Error filter that uses the covariance structure
    of both the template and the data for optimal signal estimation.

    Parameters
    ----------
    topography : ndarray, shape (n_timepoints, n_channels)
        Signal template
    data : ndarray, shape (n_samples, n_channels)
        Data for computing signal covariance
    reg : float, optional
        Tikhonov regularization on the data covariance (added as
        ``reg * trace(C)/n_channels * I``) before inversion. Required when the data is
        rank-deficient (e.g. an already-reconstructed near-dipolar field), otherwise the
        near-null directions blow up. Default 0.0 (no regularization, original behaviour).

    Returns
    -------
    W : ndarray, shape (n_channels, n_channels)
        LMMSE filter matrix

    References
    ----------
    Wilson, J.D., and Haueisen, J. (2017). IEEE Trans Biomed Eng, 64(4), 904-16.

    Chen, M., Van Veen, B.D., and Wakai, R.T. (2006). Linear Minimum Mean-Square
    Error Filtering for Evoked Responses: Application to Fetal MEG.
    IEEE Trans Biomed Eng, 53(5), 959-63. https://doi.org/10.1109/TBME.2006.872822
    """
    # LMMSE Filter: C_T / C_S
    signal_cov = np.cov(data.T)
    if reg:
        signal_cov = signal_cov + reg * np.trace(signal_cov) / signal_cov.shape[0] * np.eye(signal_cov.shape[0])
    U, S, _ = np.linalg.svd(signal_cov, hermitian=True)
    W = np.cov(topography.T) @ U @ np.diag(1 / S) @ U.T

    return W


def ssp(topography, k=None, explained_variance=None, verbose=False, nullspace=True):
    """
    Signal Space Projection (SSP) operator.

    Projects data onto or out of the k-dimensional subspace defined by the
    principal components of the template covariance.

    Parameters
    ----------
    topography : ndarray, shape (n_timepoints, n_channels)
        Spatial template
    k : int, optional
        Number of principal components to use. Mutually exclusive with explained_variance.
    explained_variance : float, optional
        Fraction of variance to explain (e.g., 0.95). Mutually exclusive with k.
    verbose : bool, optional
        Print variance statistics (default: False)
    nullspace : bool, optional
        If True, project onto nullspace; if False, project into subspace (default: True)

    Returns
    -------
    P : ndarray, shape (n_channels, n_channels)
        SSP projection operator

    Raises
    ------
    ValueError
        If both or neither of k and explained_variance are provided
    """
    if k is not None and explained_variance is not None:
        raise ValueError("Please provide either `k` or `explained_variance`, not both.")
    if k is None and explained_variance is None:
        raise ValueError("Please provide either `k` or `explained_variance`.")

    # Compute template covariance and SVD
    template_cov = np.cov(topography.T)
    U, S, _ = np.linalg.svd(template_cov, hermitian=True)
    explained_variance_by_component = S / np.sum(S)

    if verbose:
        with np.printoptions(precision=2, floatmode="fixed", suppress=True):
            print(
                f"Explained Variance by each component: {explained_variance_by_component*100}%"
            )
            print(
                f"Cumulative Explained Variance: {np.cumsum(explained_variance_by_component)*100}%"
            )
            print(f"Rank of template covariance: {np.linalg.matrix_rank(template_cov)}")

    # Determine k from explained variance if needed
    if k is None:
        k = (
            np.searchsorted(
                np.cumsum(explained_variance_by_component), explained_variance
            )
            + 1
        )
        if verbose:
            print(
                f"Selected number of components: {k}, explaining "
                f"{np.cumsum(explained_variance_by_component)[k-1]*100:.2f}% of variance"
            )

    # Build projector
    P = U[:, :k] @ U[:, :k].T
    if nullspace:
        P = np.eye(P.shape[0]) - P

    return P


def subspace_projector(topography, **kwargs):
    """
    Subspace projector (signal enhancement) similar to SSP but projecting onto the subspace instead of the nullspace.

    Projects data into the k-dimensional subspace of the principal components.
    Use for enhancing signals. Equivalent to ssp with nullspace=False.

    Parameters
    ----------
    topography : ndarray, shape (n_timepoints, n_channels)
        Spatial template to project into
    **kwargs : optional
        Additional arguments passed to ssp (k, explained_variance, verbose)

    Returns
    -------
    P : ndarray, shape (n_channels, n_channels)
        Subspace projection operator
    """
    return ssp(topography, **kwargs, nullspace=False)


def op(topography, threshold, plot=False):
    """
    Orthogonal Projection (OP) operator.

    Iteratively finds time points with maximum spatial norm and projects them out
    until all remaining vectors fall below the threshold. This adaptively selects
    the strongest topographies from the template.

    Parameters
    ----------
    topography : ndarray, shape (n_timepoints, n_channels)
        Spatial template
    threshold : float
        Stopping threshold for maximum topography norm
    plot : bool, optional
        Whether to plot each iteration (default: False)

    Returns
    -------
    P : ndarray, shape (n_channels, n_channels)
        Nullspace projection operator

    Reference
    ---------
    Vrba, J., et al. (2004). Fetal MEG redistribution by projection operators.
    IEEE Trans Biomed Eng, 51(7), 1207-1218. https://doi.org/10.1109/TBME.2004.827265
    """
    T = topography.copy()

    # Iteratively find max norm topographies and project out
    a_list = []
    for i in range(topography.shape[0]):
        t_max = np.linalg.norm(T, axis=-1).argmax(axis=0)
        a = T[t_max]
        a_list.append(a)

        if plot:
            print(
                f"Max norm of topography at time {t_max} with norm {np.linalg.norm(a)}"
            )
            plt.plot(T)
            plt.plot(np.linalg.norm(T, axis=1), "k:", label="Norm")
            plt.vlines(t_max, T.min(), T.max(), color="r", linestyle="--")
            plt.title(f"Iteration {i+1} (max norm {np.linalg.norm(a):.2f})")
            plt.legend()
            plt.show()

        # Project out the selected topography
        T = T @ nullspace_projector(a).T

        if np.linalg.norm(a) < threshold:
            print(
                f"Stopping at iteration {i+1} as max norm {np.linalg.norm(a):.2f} "
                f"is below threshold {threshold}"
            )
            break

    # Stack selected topographies and compute complete nullspace projector
    A = np.stack(a_list, axis=0)
    P = nullspace_projector(A)
    return P


######################################## VISUALIZATION ########################################


def plot_similarity_with_energy_context(
    topography_m,
    topography_f,
    k_m=None,
    k_f=None,
    metric="angle",
    names=["Maternal", "Fetal"],
):
    """
    Plot subspace similarity between two templates with energy context.

    Creates a heatmap showing alignment between principal components of two
    templates, with marginal plots showing energy distribution and minimum
    alignment angles.

    Parameters
    ----------
    topography_m : ndarray, shape (n_timepoints, n_channels)
        First template (e.g., maternal)
    topography_f : ndarray, shape (n_timepoints, n_channels)
        Second template (e.g., fetal)
    k_m : int, optional
        Number of components from first template (default: all)
    k_f : int, optional
        Number of components from second template (default: all)
    metric : str, optional
        Display metric: 'angle' (degrees) or 'alignment' (cosine) (default: 'angle')
    names : list of str, optional
        Names for the two templates (default: ['Maternal', 'Fetal'])

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    S : ndarray, shape (k_m, k_f)
        Alignment matrix (absolute value of cosine angles)
    """
    # Get eigenvectors from covariance matrices
    Cm = np.cov(topography_m.T)
    Cf = np.cov(topography_f.T)

    # Eigendecomposition (descending order by eigenvalue)
    eigenvalues_m, eigenvectors_m = np.linalg.eigh(Cm)
    eigenvalues_f, eigenvectors_f = np.linalg.eigh(Cf)

    idx_m = np.argsort(eigenvalues_m)[::-1]
    idx_f = np.argsort(eigenvalues_f)[::-1]

    Um = eigenvectors_m[:, idx_m]
    Uf = eigenvectors_f[:, idx_f]
    Lambda_m = eigenvalues_m[idx_m]
    Lambda_f = eigenvalues_f[idx_f]

    # Determine k values
    if k_m is None:
        k_m = Um.shape[1]
    if k_f is None:
        k_f = Uf.shape[1]

    Um_k = Um[:, :k_m]
    Uf_k = Uf[:, :k_f]
    Lambda_m_k = Lambda_m[:k_m]
    Lambda_f_k = Lambda_f[:k_f]

    # Calculate similarity (alignment) and angles
    S = np.abs(Um_k.T @ Uf_k)
    angles = np.rad2deg(np.arccos(np.clip(S, 0, 1)))

    # Select metric for display
    if metric == "angle":
        data = angles
        vmin, vmax = 0, 90
        cbar_label = "Angle θ (degrees)"
        fmt = "{:.0f}°"
    else:
        data = S
        vmin, vmax = 0, 1
        cbar_label = "Alignment (cos θ)"
        fmt = "{:.2f}"

    # Min angles per component against the other subspace
    min_angles_m = np.min(angles[:k_m, :k_f], axis=1)
    min_angles_f = np.min(angles[:k_m, :k_f], axis=0)

    # Energy percentages
    energy_m = (Lambda_m_k / np.sum(Lambda_m_k)) * 100
    energy_f = (Lambda_f_k / np.sum(Lambda_f_k)) * 100

    # Create figure with grid layout
    fig = plt.figure(figsize=(4.5, 4.5))
    gs = GridSpec(
        2, 3, width_ratios=[5, 1.5, 0.2], height_ratios=[1.5, 5], wspace=0.1, hspace=0.1
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_cbar = fig.add_subplot(gs[1, 2])

    # Main heatmap
    im = ax_main.imshow(data, cmap="plasma", aspect="equal", vmin=vmin, vmax=vmax)

    # Add text annotations if not too many components
    if k_m <= 15 and k_f <= 15:
        thresh = vmin + (vmax - vmin) / 2.5
        for i in range(k_m):
            for j in range(k_f):
                val = data[i, j]
                dark_bg = val > thresh if metric != "angle" else val < thresh
                ax_main.text(
                    j,
                    i,
                    fmt.format(val),
                    ha="center",
                    va="center",
                    color="white" if dark_bg else "black",
                    fontsize=7,
                )

    ax_main.set_xticks(np.arange(k_f))
    ax_main.set_yticks(np.arange(k_m))
    ax_main.set_xticklabels([str(j + 1) for j in range(k_f)])
    ax_main.set_yticklabels([str(i + 1) for i in range(k_m)])
    ax_main.set_xlabel(f"{names[1]} Component", fontsize=10)
    ax_main.set_ylabel(f"{names[0]} Component", fontsize=10)
    ax_main.grid(which="minor", color="white", linestyle="-", linewidth=1)
    for spine in ax_main.spines.values():
        spine.set_visible(False)

    # Top marginal: Energy bars + Min angle line
    bars_top = ax_top.bar(np.arange(k_f), energy_f, color="lightgrey", width=0.7)
    ax_top.bar_label(bars_top, fmt="%.0f%%", padding=1, fontsize=6)
    ax_top_angle = ax_top.twinx()
    ax_top_angle.plot(
        np.arange(k_f),
        min_angles_f,
        color="crimson",
        marker="o",
        markersize=3,
        linewidth=1,
    )
    ax_top_angle.set_ylim(0, 90)
    ax_top_angle.set_yticks([0, 30, 60, 90])
    ax_top_angle.set_ylabel("Min θ (°)", color="crimson", fontsize=8)
    ax_top_angle.tick_params(axis="y", labelcolor="crimson", labelsize=7)
    ax_top.set_ylabel("Energy %", fontsize=8)
    ax_top.tick_params(labelbottom=False, bottom=False, labelsize=7)

    # Right marginal: Energy bars + Min angle line
    bars_right = ax_right.barh(np.arange(k_m), energy_m, color="lightgrey", height=0.7)
    ax_right.bar_label(bars_right, fmt="%.0f%%", padding=1, fontsize=6, rotation=-90)
    ax_right_angle = ax_right.twiny()
    ax_right_angle.plot(
        min_angles_m,
        np.arange(k_m),
        color="crimson",
        marker="o",
        markersize=3,
        linewidth=1,
    )
    ax_right_angle.set_xlim(0, 90)
    ax_right_angle.set_xticks([0, 30, 60, 90])
    ax_right_angle.set_xlabel("Min θ (°)", color="crimson", fontsize=8)
    ax_right_angle.tick_params(axis="x", labelcolor="crimson", labelsize=7)
    ax_right.set_xlabel("Energy %", fontsize=8)
    ax_right.tick_params(labelleft=False, left=False, labelsize=7)

    # Colorbar
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label(cbar_label, fontsize=9, labelpad=10, rotation=270, va="bottom")
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_visible(False)

    return fig, S


######################################## SPATIAL FILTER CLASS ########################################


class SpatialFilter:
    """
    Unified spatial filtering interface for signal separation.

    Provides a high-level API for applying various spatial filtering methods,
    handling template construction, and orchestrating single-stage or two-stage
    filtering pipelines.

    Filtering Methods
    -----------------
    Single-stage suppression:
        - 'nullspace': Orthogonal projection onto nullspace
        - 'op': Adaptive orthogonal projection with threshold
        - 'ssp': Signal space projection onto nullspace

    Single-stage enhancement:
        - 'lmmse': Linear minimum mean-square error filter
        - 'minnorm': Minimum norm projector (into subspace)
        - 'subspace_projector': Signal space projection into subspace

    Two-stage (suppress then enhance):
        - 'nullspace_lmmse': Nullspace cancellation followed by LMMSE
        - 'nullspace_minnorm': Nullspace cancellation followed by minimum norm
        - 'op_minnorm': Adaptive OP cancellation followed by minimum norm (OPMN)

    Two-stage (enhance + enhance)
        - 'minnorm_lmmse': Minimum norm enhancement followed by LMMSE

    Parameters
    ----------
    method : str
        Filtering method name (see Filtering Methods above)
    fs : float
        Sampling frequency in Hz
    suppress : dict, optional
        Configuration for suppression template. Required keys:
        - 'sources': ndarray of shape (n_samples, n_components)
        - 'comps': list of int, component indices for HR detection
    enhance : dict, optional
        Configuration for enhancement template (same structure as suppress)
    window_size : float, optional
        Window size as fraction of cardiac cycle (default: 1)
    bpm_tol : float, optional
        BPM tolerance for beat selection (default: 5)
    t_start : float, optional
        Start time in seconds for template construction (default: None)
    t_end : float, optional
        End time in seconds for template construction (default: None)
    subtract_mean : bool, optional
        Subtract temporal mean from template (default: True)
    axis_mask : ndarray of bool, optional
        Boolean mask indicating valid channels for data reduction

    Examples
    --------
    Single-stage suppression (nullspace projection):

    >>> sf = SpatialFilter(
    ...     method='nullspace',
    ...     fs=200,
    ...     suppress={'sources': maternal_sources, 'comps': [3, 5]},
    ...     window_size=0.6,
    ...     bpm_tol=5
    ... )
    >>> filtered = sf.apply(data, reg_power=0.01)

    Two-stage filtering (suppress then enhance):

    >>> sf = SpatialFilter(
    ...     method='nullspace_minnorm',
    ...     fs=200,
    ...     suppress={'sources': maternal_sources, 'comps': [3, 5]},
    ...     enhance={'sources': fetal_sources, 'comps': [1, 2]},
    ...     window_size=0.25
    ... )
    >>> filtered = sf.apply(data, reg_power=0.01)
    >>> W = sf.get_filter(data, reg_power=0.01)

    Signal Space Projection (suppression):

    >>> sf = SpatialFilter(
    ...     method='ssp',
    ...     fs=200,
    ...     suppress={'sources': maternal_sources, 'comps': [3, 5]}
    ... )
    >>> filtered = sf.apply(data, k=3)  # Project out top 3 components
    >>> # Or use explained variance
    >>> filtered = sf.apply(data, explained_variance=0.95)

    Signal Space Projection (enhancement):

    >>> sf = SpatialFilter(
    ...     method='subspace_projector',
    ...     fs=200,
    ...     enhance={'sources': fetal_sources, 'comps': [1, 2]}
    ... )
    >>> filtered = sf.apply(data, k=2)  # Keep top 2 components

    Two-stage enhancement (minimum norm then LMMSE):

    >>> sf = SpatialFilter(
    ...     method='minnorm_lmmse',
    ...     fs=200,
    ...     enhance={'sources': fetal_sources, 'comps': [1, 2]},
    ...     window_size=0.25
    ... )
    >>> filtered = sf.apply(data, reg_power=0.01)
    """

    # Valid method names
    SINGLE_STAGE_SUPPRESS = ["nullspace", "op", "ssp"]
    SINGLE_STAGE_ENHANCE = ["lmmse", "minnorm", "subspace_projector"]
    TWO_STAGE = ["nullspace_lmmse", "nullspace_minnorm", "op_minnorm", "minnorm_lmmse"]

    def __init__(
        self,
        method,
        fs,
        suppress=None,
        enhance=None,
        window_size=1,
        bpm_tol=5,
        t_start=None,
        t_end=None,
        subtract_mean=True,
        axis_mask=None,
    ):
        self.method = method
        self.fs = fs
        self.suppress = suppress
        self.enhance = enhance
        self.window_size = window_size
        self.bpm_tol = bpm_tol
        self.t_start = t_start
        self.t_end = t_end
        self.subtract_mean = subtract_mean
        self.axis_mask = axis_mask

        # Validate method
        valid_methods = (
            self.SINGLE_STAGE_SUPPRESS + self.SINGLE_STAGE_ENHANCE + self.TWO_STAGE
        )
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of: {', '.join(valid_methods)}"
            )

        # Validate configuration based on method requirements
        self._validate_config()

    def _validate_config(self):
        """Validate that required templates are provided for the chosen method."""
        if self.method in self.SINGLE_STAGE_SUPPRESS:
            if self.suppress is None:
                raise ValueError(
                    f"Method '{self.method}' requires 'suppress' configuration with "
                    "'sources' and 'comps' keys"
                )
        elif self.method in self.SINGLE_STAGE_ENHANCE:
            if self.enhance is None:
                raise ValueError(
                    f"Method '{self.method}' requires 'enhance' configuration with "
                    "'sources' and 'comps' keys"
                )
        elif self.method in self.TWO_STAGE:
            if self.method == "minnorm_lmmse":
                # Enhance + Enhance method: only needs enhance config
                if self.enhance is None:
                    raise ValueError(
                        f"Method '{self.method}' requires 'enhance' configuration with "
                        "'sources' and 'comps' keys"
                    )
            else:
                # Suppress + Enhance methods: need both configs
                if self.suppress is None or self.enhance is None:
                    raise ValueError(
                        f"Method '{self.method}' requires both 'suppress' and 'enhance' "
                        "configurations with 'sources' and 'comps' keys"
                    )

    def _reduce_data(self, data):
        """
        Reduce data using axis_mask if needed.

        Parameters
        ----------
        data : ndarray
            Input data

        Returns
        -------
        ndarray
            Reduced data or original if no mask
        """
        if self.axis_mask is None:
            return data

        # Check if data has full channel dimension and needs reduction
        if (
            data.ndim == 3
            and data.shape[1] == self.axis_mask.shape[0]
            and data.shape[2] == self.axis_mask.shape[1]
        ):
            return data[:, self.axis_mask]

        return data

    def _construct_template(self, data, config):
        """
        Construct template from sources using cardiac beat averaging.

        Parameters
        ----------
        data : ndarray, shape (n_samples, n_channels)
            Data to construct template from
        config : dict
            Template configuration with 'sources' and 'comps' keys

        Returns
        -------
        ndarray, shape (n_timepoints, n_channels)
            Constructed spatial template
        """
        return construct_template(
            data=data,
            sources=config["sources"],
            fs=self.fs,
            comps=config["comps"],
            window_size=self.window_size,
            bpm_tol=self.bpm_tol,
            plot=False,
            axis_mask=None,
            subtract_mean=self.subtract_mean,
            t_start=self.t_start,
            t_end=self.t_end,
        )

    def _build_filter(self, data_reduced, **filter_config):
        """
        Build filter matrix based on method.

        Parameters
        ----------
        data_reduced : ndarray, shape (n_samples, n_channels)
            Reduced input data
        **filter_config : keyword arguments
            Method-specific filter parameters

        Returns
        -------
        W : ndarray, shape (n_channels, n_channels)
            Filter weight matrix
        """
        # ========== Single-stage suppression methods ==========
        if self.method in self.SINGLE_STAGE_SUPPRESS:
            template = self._construct_template(data_reduced, self.suppress)

            if self.method == "nullspace":
                return nullspace_projector(template, **filter_config)
            elif self.method == "op":
                # For OP, convert relative threshold_power to absolute threshold
                rms = np.sqrt(np.mean(data_reduced**2))
                threshold_power = filter_config.pop("threshold_power", 1.0)
                filter_config["threshold"] = threshold_power * rms
                return op(template, **filter_config)
            elif self.method == "ssp":
                return ssp(template, **filter_config)

        # ========== Single-stage enhancement methods ==========
        elif self.method in self.SINGLE_STAGE_ENHANCE:
            template = self._construct_template(data_reduced, self.enhance)

            if self.method == "lmmse":
                return covariance_projector(template, data_reduced)
            elif self.method == "minnorm":
                return minimum_norm_projector(
                    template, **filter_config, nullspace=False
                )
            elif self.method == "subspace_projector":
                return subspace_projector(template, **filter_config)

        # ========== Two-stage methods ==========
        elif self.method in self.TWO_STAGE:

            # Suppress + Enhance methods
            if self.method in ["nullspace_lmmse", "nullspace_minnorm", "op_minnorm"]:
                # Step 1: Suppress undesired signal
                template_suppress = self._construct_template(
                    data_reduced, self.suppress
                )

                if self.method in ["nullspace_lmmse", "nullspace_minnorm"]:
                    W_suppress = nullspace_projector(template_suppress, **filter_config)
                elif self.method == "op_minnorm":
                    # For OP, convert relative threshold_power to absolute threshold
                    rms = np.sqrt(np.mean(data_reduced**2))
                    threshold_power = filter_config.pop("threshold_power", 1.0)
                    op_config = {
                        "threshold": threshold_power * rms,
                        "plot": filter_config.get("plot", False),
                    }
                    W_suppress = op(template_suppress, **op_config)

                data_suppressed = data_reduced @ W_suppress.T

                # Step 2: Enhance desired signal from suppressed data
                template_enhance = self._construct_template(
                    data_suppressed, self.enhance
                )

                if self.method == "nullspace_lmmse":
                    W_enhance = covariance_projector(template_enhance, data_suppressed)
                elif self.method in ["nullspace_minnorm", "op_minnorm"]:
                    W_enhance = minimum_norm_projector(
                        template_enhance, **filter_config, nullspace=False
                    )

                # Combine filters: W such that data @ W.T = data @ W_suppress.T @ W_enhance.T
                # Therefore: W = W_enhance @ W_suppress
                return W_enhance @ W_suppress

            # Enhance + Enhance methods
            elif self.method == "minnorm_lmmse":
                # Step 1: First enhancement with minimum norm
                template_enhance1 = self._construct_template(data_reduced, self.enhance)
                W_minnorm = minimum_norm_projector(
                    template_enhance1, **filter_config, nullspace=False
                )
                data_enhanced = data_reduced @ W_minnorm.T

                # Step 2: Second enhancement with LMMSE
                template_enhance2 = self._construct_template(
                    data_enhanced, self.enhance
                )
                W_lmmse = covariance_projector(template_enhance2, data_enhanced)

                # Combine filters: W such that data @ W.T = data @ W_minnorm.T @ W_lmmse.T
                # Therefore: W = W_lmmse @ W_minnorm
                return W_lmmse @ W_minnorm

        else:
            raise ValueError(f"Unknown method '{self.method}'")

    def apply(self, data, **filter_config):
        """
        Apply spatial filter to data and return filtered signal.

        Parameters
        ----------
        data : ndarray, shape (n_samples, n_channels)
            Input data to filter
        **filter_config : keyword arguments
            Method-specific parameters:

            For 'ssp' or 'subspace_projector':
                - k : int - Number of components, OR
                - explained_variance : float - Fraction of variance (e.g., 0.95)
                - verbose : bool - Print variance info (default: False)

            For 'op' or 'op_minnorm':
                - threshold_power : float - Stopping threshold as fraction of RMS (default: 1.0)
                - plot : bool - Plot iterations (default: False)

            For 'nullspace' or minimum norm methods:
                - reg_power : float - Regularization parameter (default: 0)

        Returns
        -------
        filtered_data : ndarray, shape (n_samples, n_channels)
            Spatially filtered data
        """
        data_reduced = self._reduce_data(data)
        W = self._build_filter(data_reduced, **filter_config)
        return data @ W.T

    def get_filter(self, data, **filter_config):
        """
        Compute and return filter matrix without applying to data.

        Parameters
        ----------
        data : ndarray, shape (n_samples, n_channels)
            Input data (used for template construction)
        **filter_config : keyword arguments
            Method-specific parameters (same as apply())

        Returns
        -------
        W : ndarray, shape (n_channels, n_channels)
            Filter weight matrix
        """
        data_reduced = self._reduce_data(data)
        return self._build_filter(data_reduced, **filter_config)


class LocalSpatialFilter:
    """
    Local (time-windowed) spatial filter — a thin wrapper around ``SpatialFilter``.

    ``SpatialFilter`` is *global*: it estimates a single filter matrix from a
    template/covariance over the entire recording and applies it everywhere. This
    assumes the source topographies (maternal/fetal fields) are stationary. When
    they drift (fetal movement, position changes, heart-rate variation), a single
    global filter is a compromise that fits no part of the record well.

    ``LocalSpatialFilter`` instead slides a window over the recording, rebuilds a
    fresh ``SpatialFilter`` on each window's own data (and correspondingly sliced
    ``sources``), applies it locally, and stitches the windows back together. The
    resulting filter tracks the drifting topography over time. All filtering
    methods, template-construction options, and ``apply()`` keyword arguments are
    exactly those of ``SpatialFilter`` (see that class for the method list) and are
    forwarded verbatim.

    Parameters
    ----------
    method : str
        Filtering method name (see ``SpatialFilter`` for valid methods).
    fs : float
        Sampling frequency in Hz.
    window_sec : float
        Window length in seconds. Must be large enough to contain several cardiac
        cycles, since each window independently runs heart-rate detection and beat
        averaging for template construction (e.g. >= ~10 s).
    hop_sec : float, optional
        Step between successive window starts in seconds (default: ``window_sec / 2``,
        i.e. 50% overlap). Ignored when ``stitch='block'`` (forced to ``window_sec``).
    stitch : {'crossfade', 'block'}, optional
        How overlapping windows are combined (default: 'crossfade'):
        - 'crossfade': windows overlap and are blended with a Hann taper, normalised
          by the accumulated weights, so the time-varying filter has no block
          discontinuities.
        - 'block': non-overlapping windows filtered independently and concatenated
          (most concise; introduces discontinuities at block edges).
    suppress, enhance : dict, optional
        Template configurations, same structure as ``SpatialFilter`` (each with
        'sources' and 'comps'). The full-length 'sources' array is sliced to each
        window automatically.
    window_size, bpm_tol, t_start, t_end, subtract_mean, axis_mask
        Forwarded to each per-window ``SpatialFilter`` (same meaning as there),
        except ``t_start``/``t_end``, which here restrict the *time span that is
        filtered* (seconds); samples outside the span are passed through unchanged.

    Examples
    --------
    >>> lsf = LocalSpatialFilter(
    ...     method="lmmse", fs=fs, window_sec=30, hop_sec=15,
    ...     enhance={"sources": sources, "comps": fetals},
    ... )
    >>> filtered = lsf.apply(field_maps[:, axis_mask], reg_power=0.01)
    """

    def __init__(
        self,
        method,
        fs,
        window_sec,
        hop_sec=None,
        stitch="crossfade",
        suppress=None,
        enhance=None,
        window_size=1,
        bpm_tol=5,
        t_start=None,
        t_end=None,
        subtract_mean=True,
        axis_mask=None,
    ):
        if window_sec <= 0:
            raise ValueError(f"window_sec must be positive, got {window_sec}")
        if stitch not in ("crossfade", "block"):
            raise ValueError(
                f"Invalid stitch '{stitch}'. Must be 'crossfade' or 'block'."
            )

        self.method = method
        self.fs = fs
        self.window_sec = window_sec
        self.hop_sec = hop_sec if hop_sec is not None else window_sec / 2
        self.stitch = stitch
        self.suppress = suppress
        self.enhance = enhance
        self.window_size = window_size
        self.bpm_tol = bpm_tol
        self.t_start = t_start
        self.t_end = t_end
        self.subtract_mean = subtract_mean
        self.axis_mask = axis_mask

        self.win = int(window_sec * fs)
        self.hop = self.win if stitch == "block" else int(self.hop_sec * fs)
        if self.hop <= 0:
            raise ValueError(f"hop_sec must be positive, got {self.hop_sec}")

        # Fail fast on invalid method/config combinations, consistent with
        # SpatialFilter (validation is identical since args are forwarded).
        self._make_filter(self.suppress, self.enhance)

    def _make_filter(self, suppress, enhance):
        """Instantiate a ``SpatialFilter`` sharing this wrapper's configuration."""
        return SpatialFilter(
            method=self.method,
            fs=self.fs,
            suppress=suppress,
            enhance=enhance,
            window_size=self.window_size,
            bpm_tol=self.bpm_tol,
            subtract_mean=self.subtract_mean,
            axis_mask=self.axis_mask,
        )

    def _slice_cfg(self, cfg, s, e):
        """Slice a template config's ``sources`` to the window ``[s:e]``."""
        if cfg is None:
            return None
        c = dict(cfg)
        c["sources"] = cfg["sources"][s:e]
        return c

    def _taper(self, length):
        """Window weights for one segment (Hann for crossfade, flat for block)."""
        if self.stitch == "block":
            return np.ones(length)
        # +2 / [1:-1] keeps endpoints strictly positive so edge samples never get
        # zero weight (no divide-by-zero, no zeroed record ends).
        return np.hanning(length + 2)[1:-1]

    def apply(self, data, **filter_config):
        """
        Apply the local spatial filter to data and return the filtered signal.

        Parameters
        ----------
        data : ndarray, shape (n_samples, n_channels)
            Input data to filter.
        **filter_config : keyword arguments
            Method-specific parameters, forwarded to each per-window
            ``SpatialFilter.apply`` (e.g. ``reg_power``, ``k``,
            ``explained_variance``, ``threshold_power``).

        Returns
        -------
        filtered_data : ndarray, shape (n_samples, n_channels)
            Locally filtered data.
        """
        n = data.shape[0]

        # Restrict filtering to [t_start, t_end] (seconds); pass through the rest.
        span_start = 0 if self.t_start is None else max(0, int(self.t_start * self.fs))
        span_end = n if self.t_end is None else min(n, int(self.t_end * self.fs))

        out = np.array(data, dtype=float)  # copy; untouched samples pass through
        out[span_start:span_end] = 0.0
        wsum = np.zeros(n)

        for s in range(span_start, span_end, self.hop):
            e = min(s + self.win, span_end)
            sf = self._make_filter(
                self._slice_cfg(self.suppress, s, e),
                self._slice_cfg(self.enhance, s, e),
            )
            y = sf.apply(data[s:e], **filter_config)
            taper = self._taper(e - s)
            out[s:e] += taper[:, None] * y
            wsum[s:e] += taper
            if e == span_end:
                break

        active = wsum > 0
        out[active] /= wsum[active, None]
        return out
