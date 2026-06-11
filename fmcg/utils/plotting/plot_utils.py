import logging
import os
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import neurokit2 as nk
import torch
import matplotlib.gridspec as gridspec


from . import plt_config

logger = logging.getLogger(__name__)

COORDINATE_IMAGE = os.path.join(os.path.dirname(__file__), "coordinates.png")


def plot_magnetic_moments(m_hat, time=None, fs=None, savename=None, xlim=[50, 55], ylabel=r"Magnetic Moment [$\mathrm{\mu}$Am$^2$]", preprocess=False, preprocess_method="vg"):
    if time is None:
        if fs is None:
            raise ValueError("Either time or fs must be provided.")
        time = np.arange(m_hat.shape[0]) / fs

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7.11, 3))

    # Plot the first component of m_hat
    # m_hat[:, 0] may be (T,3) -> three axes
    y_fetal = m_hat[:, 0]
    if preprocess:
        try:
            import neurokit2 as nk
        except Exception:
            raise ImportError("neurokit2 is required for preprocessing but not installed")

        # estimate sampling rate if needed
        if fs is None:
            if time is None:
                raise ValueError("fs must be provided or time must be set to estimate sampling rate for preprocessing")
            fs_est = int(round(1.0 / np.median(np.diff(time))))
        else:
            fs_est = int(fs)

        # apply cleaning per axis if multi-dimensional
        if y_fetal.ndim == 1:
            y_fetal = nk.ecg_clean(y_fetal, sampling_rate=fs_est, method=preprocess_method)
        else:
            y_clean = np.zeros_like(y_fetal)
            for j in range(y_fetal.shape[1]):
                y_clean[:, j] = nk.ecg_clean(y_fetal[:, j], sampling_rate=fs_est, method=preprocess_method)
            y_fetal = y_clean

    ax[0].plot(time, y_fetal)
    ax[0].set_title("Fetal Magnetic Moment")
    from matplotlib.ticker import MaxNLocator, AutoMinorLocator

    ax[0].grid(True, which='minor', linestyle=':', linewidth=0.7)
    ax[0].minorticks_on()
    ax[0].grid(True)
    ax[0].yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    ax[0].yaxis.set_minor_locator(AutoMinorLocator(2))
    ax[0].legend(["x", "y", "z"], loc="upper right")

    # Plot the second component of m_hat
    # m_hat[:, 1] may be (T,3) -> three axes
    y_maternal = m_hat[:, 1]
    if preprocess:
        # reuse fs_est if available
        try:
            import neurokit2 as nk
        except Exception:
            raise ImportError("neurokit2 is required for preprocessing but not installed")

        if fs is None:
            if time is None:
                raise ValueError("fs must be provided or time must be set to estimate sampling rate for preprocessing")
            fs_est = int(round(1.0 / np.median(np.diff(time))))
        else:
            fs_est = int(fs)

        if y_maternal.ndim == 1:
            y_maternal = nk.ecg_clean(y_maternal, sampling_rate=fs_est, method=preprocess_method)
        else:
            y_clean = np.zeros_like(y_maternal)
            for j in range(y_maternal.shape[1]):
                y_clean[:, j] = nk.ecg_clean(y_maternal[:, j], sampling_rate=fs_est, method=preprocess_method)
            y_maternal = y_clean

    ax[1].plot(time, y_maternal)
    ax[1].set_title("Maternal Magnetic Moment")
    ax[1].grid(True)
    ax[1].minorticks_on()
    ax[1].grid(True, which='minor', linestyle=':', linewidth=0.7)
    ax[1].yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    ax[1].yaxis.set_minor_locator(AutoMinorLocator(2))
    # ax[1].set_ylabel(r"Magnetic Moment [uAm$^2$]")
    ax[1].legend(["x", "y", "z"], loc="upper right")

    # Set x-axis minor ticks for both subplots
    ax[1].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[0].xaxis.set_minor_locator(AutoMinorLocator(4))

    # Set the x-axis label for the shared axis
    ax[1].set_xlabel("Time [s]")
    ax[1].set_xlim(xlim)
    fig.supylabel(ylabel)

    if savename:
        plt.savefig(
            savename,
            bbox_inches="tight",
            pad_inches=0.01, dpi=fig.dpi,
        )
        plt.close(fig)
    else:
        plt.show()


def plot_compare_magnetic_moments(
    m_hat_list,
    titles=None,
    time=None,
    fs=None,
    savename=None,
    xlim=[50, 55],
    ylabel=r"Magnetic Moment [nAm$^2$]",
    preprocess=False,
    preprocess_method="vg",
    sharey=True,
):
    """
    Compare two `m_hat` estimates side-by-side in a 2x2 grid.

    Parameters
    ----------
    m_hat_list : list or tuple
        Two `m_hat` arrays (each shaped like the original `plot_magnetic_moments` input).
    titles : list or tuple, optional
        Two titles for the left/right columns. If None, defaults to ["A", "B"].
    Other parameters mirror `plot_magnetic_moments`.
    """
    if not isinstance(m_hat_list, (list, tuple)) or len(m_hat_list) != 2:
        raise ValueError("m_hat_list must be a list/tuple of two m_hat arrays")

    if titles is None:
        titles = ["A", "B"]

    if time is None:
        if fs is None:
            raise ValueError("Either time or fs must be provided.")
        time = np.arange(m_hat_list[0].shape[0]) / fs

    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(7.11, 2.5), dpi=500)
    from matplotlib.ticker import MaxNLocator, AutoMinorLocator

    for col, (m_hat, title) in enumerate(zip(m_hat_list, titles)):
        # fetal
        y_fetal = m_hat[:, 0]
        # maternal
        y_maternal = m_hat[:, 1]

        if preprocess:
            try:
                import neurokit2 as nk
            except Exception:
                raise ImportError("neurokit2 is required for preprocessing but not installed")

            if fs is None:
                if time is None:
                    raise ValueError(
                        "fs must be provided or time must be set to estimate sampling rate for preprocessing"
                    )
                fs_est = int(round(1.0 / np.median(np.diff(time))))
            else:
                fs_est = int(fs)

            # fetal clean
            if y_fetal.ndim == 1:
                y_fetal = nk.ecg_clean(y_fetal, sampling_rate=fs_est, method=preprocess_method)
            else:
                y_clean = np.zeros_like(y_fetal)
                for j in range(y_fetal.shape[1]):
                    y_clean[:, j] = nk.ecg_clean(y_fetal[:, j], sampling_rate=fs_est, method=preprocess_method)
                y_fetal = y_clean

            # maternal clean
            if y_maternal.ndim == 1:
                y_maternal = nk.ecg_clean(y_maternal, sampling_rate=fs_est, method=preprocess_method)
            else:
                y_clean = np.zeros_like(y_maternal)
                for j in range(y_maternal.shape[1]):
                    y_clean[:, j] = nk.ecg_clean(y_maternal[:, j], sampling_rate=fs_est, method=preprocess_method)
                y_maternal = y_clean

        axf = axes[0, col]
        axm = axes[1, col]

        # scale to nA (1 uA = 1000 nA)
        y_fetal_plot = y_fetal * 1e3
        y_maternal_plot = y_maternal * 1e3

        axf.plot(time, y_fetal_plot)
        # column title on the top subplot
        axf.set_title(title)
        axf.grid(True, which="minor", linestyle=":", linewidth=0.7)
        axf.minorticks_on()
        axf.grid(True)
        axf.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        axf.yaxis.set_minor_locator(AutoMinorLocator(2))
        #axf.legend(["x", "y", "z"], loc="upper right")
        # add boxed label inside the plot (like other plot text usage)
        bbox_props = dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="black", linewidth=.5)
        axf.text(0.03, 0.85, "Fetal", ha="left", va="center", transform=axf.transAxes, bbox=bbox_props, zorder=100, fontsize=9)

        axm.plot(time, y_maternal_plot)
        # no column title here; use the top subplot for the column title
        axm.grid(True, which="minor", linestyle=":", linewidth=0.7)
        axm.minorticks_on()
        axm.grid(True)
        axm.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        axm.yaxis.set_minor_locator(AutoMinorLocator(2))
        #axm.legend(["x", "y", "z"], loc="upper right")
        bbox_props = dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="black", linewidth=.5)
        axm.text(0.03, 0.85, "Maternal", ha="left", va="center", transform=axm.transAxes, bbox=bbox_props, zorder=100, fontsize=9)

        # x minor ticks for both
        axm.xaxis.set_minor_locator(AutoMinorLocator(4))
        axf.xaxis.set_minor_locator(AutoMinorLocator(4))

        # ensure xlim is applied to the bottom row (maternal) for both columns
        axm.set_xlim(xlim)

    # Share y-limits per row (fetal row 0, maternal row 1)
    if sharey:
        # Row 0: fetal plots
        fetal_ylims = [axes[0, 0].get_ylim(), axes[0, 1].get_ylim()]
        fetal_ymin = min(ylim[0] for ylim in fetal_ylims)
        fetal_ymax = max(ylim[1] for ylim in fetal_ylims)
        axes[0, 0].set_ylim(fetal_ymin, fetal_ymax)
        axes[0, 1].set_ylim(fetal_ymin, fetal_ymax)
        
        # Row 1: maternal plots
        maternal_ylims = [axes[1, 0].get_ylim(), axes[1, 1].get_ylim()]
        maternal_ymin = min(ylim[0] for ylim in maternal_ylims)
        maternal_ymax = max(ylim[1] for ylim in maternal_ylims)
        axes[1, 0].set_ylim(maternal_ymin, maternal_ymax)
        axes[1, 1].set_ylim(maternal_ymin, maternal_ymax)

    # shared x label and y label
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 1].set_xlabel("Time [s]")
    fig.supylabel(ylabel)

    # create one shared legend for the three axes labels
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color=f"C{i}") for i in range(3)]
    fig.legend(legend_handles, ["x", "y", "z"], ncol=3, loc="upper center", bbox_to_anchor=(0.53, 1.05))

    plt.tight_layout()
    if savename:
        plt.savefig(savename, bbox_inches="tight", pad_inches=0.01, dpi=fig.dpi)
        plt.close(fig)
    else:
        plt.show()


def plot_magnetic_moments_separate_axes(m_hat, time=None, fs=None, savename=None, xlim=[50, 55], ylabel=r"Magnetic Moment [$\mathrm{\mu}$Am$^2$]", preprocess=False, preprocess_method="vg", plot_pca=False, separate_figures=False):
    """
    Plot magnetic moments with each axis (x, y, z) in separate subplots.
    
    Parameters
    ----------
    m_hat : ndarray
        Magnetic moments with shape (time, 2, 3) where 2 is [fetal, maternal] and 3 is [x, y, z]
    time : ndarray, optional
        Time array. If None, will be computed from fs
    fs : float, optional
        Sampling frequency. Required if time is None
    savename : str, optional
        Path to save the figure. If None, figure is displayed
    xlim : list, optional
        X-axis limits [start, end] in seconds
    ylabel : str, optional
        Y-axis label
    """
    if time is None:
        if fs is None:
            raise ValueError("Either time or fs must be provided.")
        time = np.arange(m_hat.shape[0]) / fs

    from matplotlib.ticker import MaxNLocator, AutoMinorLocator
    
    axis_names = ['x', 'y', 'z']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # helper to preprocess and (optionally) compute PCA1
    def _prepare_source(col_idx):
        Y = m_hat[:, col_idx, :].astype(float)
        # preprocess each axis if requested
        if preprocess:
            try:
                import neurokit2 as nk
            except Exception:
                raise ImportError("neurokit2 is required for preprocessing but not installed")

            if fs is None:
                if time is None:
                    raise ValueError("fs must be provided or time must be set to estimate sampling rate for preprocessing")
                fs_est = int(round(1.0 / np.median(np.diff(time))))
            else:
                fs_est = int(fs)

            Yc = np.zeros_like(Y)
            for j in range(Y.shape[1]):
                Yc[:, j] = nk.ecg_clean(Y[:, j], sampling_rate=fs_est, method=preprocess_method)
            Y = Yc

        pca1 = None
        if plot_pca:
            # mean-center and take first principal component via SVD
            Yc = Y - np.mean(Y, axis=0)
            try:
                U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
                pca1 = Yc @ Vt.T[:, 0]
            except Exception:
                pca1 = None

        return Y, pca1

    # If splitting into two separate figures
    if separate_figures:
        # fetal
        Y_f, pca_f = _prepare_source(0)
        nrows = 3
        fig_f, axes_f = plt.subplots(nrows, 1, sharex=True, figsize=(7.11, 1 * nrows))
        if nrows == 1:
            axes_f = [axes_f]
        for i in range(3):
            ax = axes_f[i]
            ax.plot(time, Y_f[:, i], color=colors[i], label=f"{axis_names[i]}")
            # overlay PCA (scaled to axis)
            if plot_pca and pca_f is not None:
                p = pca_f
                # scale PCA to match axis std
                p_scaled = (p - np.mean(p)) / (np.std(p) + 1e-12) * (np.std(Y_f[:, i]) + 0) + np.mean(Y_f[:, i])
                ax.plot(time, p_scaled, color='k', linestyle='--', label='PCA1')
            ax.grid(True, which='minor', linestyle=':', linewidth=0.7)
            ax.minorticks_on()
            ax.grid(True)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.xaxis.set_minor_locator(AutoMinorLocator(4))
            ax.set_xlim(xlim)
            ax.legend(loc='upper right')

        fig_f.supylabel("Fetal " + ylabel)
        axes_f[-1].set_xlabel('Time [s]')
        plt.tight_layout()
        if savename:
            base, ext = os.path.splitext(savename)
            sav_f = f"{base}_fetal{ext}"
            plt.savefig(sav_f, bbox_inches='tight', pad_inches=0.01, dpi=fig_f.dpi)
            plt.close(fig_f)
        else:
            plt.show()

        # maternal
        Y_m, pca_m = _prepare_source(1)
        fig_m, axes_m = plt.subplots(nrows, 1, sharex=True, figsize=(7.11, 1* nrows))
        if nrows == 1:
            axes_m = [axes_m]
        for i in range(3):
            ax = axes_m[i]
            ax.plot(time, Y_m[:, i], color=colors[i], label=f"{axis_names[i]}")
            if plot_pca and pca_m is not None:
                p = pca_m
                p_scaled = (p - np.mean(p)) / (np.std(p) + 1e-12) * (np.std(Y_m[:, i]) + 0) + np.mean(Y_m[:, i])
                ax.plot(time, p_scaled, color='k', linestyle='--', label='PCA1')
            ax.grid(True, which='minor', linestyle=':', linewidth=0.7)
            ax.minorticks_on()
            ax.grid(True)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.xaxis.set_minor_locator(AutoMinorLocator(4))
            ax.set_xlim(xlim)
            ax.legend(loc='upper right')

        fig_m.supylabel("Maternal " + ylabel)
        axes_m[-1].set_xlabel('Time [s]')
        plt.tight_layout()
        if savename:
            base, ext = os.path.splitext(savename)
            sav_m = f"{base}_maternal{ext}"
            plt.savefig(sav_m, bbox_inches='tight', pad_inches=0.01, dpi=fig_m.dpi)
            plt.close(fig_m)
        else:
            plt.show()
        return

    # default: single figure with fetal axes stacked then maternal axes
    # build combined arrays
    Y_f, pca_f = _prepare_source(0)
    Y_m, pca_m = _prepare_source(1)

    n_f = 3 + (1 if plot_pca else 0)
    n_m = 3 + (1 if plot_pca else 0)
    total_rows = n_f + n_m
    fig, axes = plt.subplots(total_rows, 1, sharex=True, figsize=(7.11, 2.5 * total_rows))
    if total_rows == 1:
        axes = [axes]

    plot_idx = 0
    # plot fetal
    for i in range(3):
        ax = axes[plot_idx]
        ax.plot(time, Y_f[:, i], color=colors[i])
        ax.set_title(f"Fetal {axis_names[i]}")
        ax.grid(True, which='minor', linestyle=':', linewidth=0.7)
        ax.minorticks_on()
        ax.grid(True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.set_xlim(xlim)
        plot_idx += 1
    if plot_pca and pca_f is not None:
        ax = axes[plot_idx]
        ax.plot(time, pca_f, color='k')
        ax.set_title('Fetal PCA1')
        ax.grid(True)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.set_xlim(xlim)
        plot_idx += 1

    # plot maternal
    for i in range(3):
        ax = axes[plot_idx]
        ax.plot(time, Y_m[:, i], color=colors[i])
        ax.set_title(f"Maternal {axis_names[i]}")
        ax.grid(True, which='minor', linestyle=':', linewidth=0.7)
        ax.minorticks_on()
        ax.grid(True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.set_xlim(xlim)
        plot_idx += 1
    if plot_pca and pca_m is not None:
        ax = axes[plot_idx]
        ax.plot(time, pca_m, color='k')
        ax.set_title('Maternal PCA1')
        ax.grid(True)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.set_xlim(xlim)

    # Set x-axis label for bottom row
    axes[-1].set_xlabel("Time [s]")
    
    # Set y-axis label
    fig.supylabel(ylabel)
    
    plt.tight_layout()
    
    if savename:
        plt.savefig(
            savename,
            bbox_inches="tight",
            pad_inches=0.01, dpi=fig.dpi,
        )
        plt.close(fig)
    else:
        plt.show()


def set_axes_equal(ax):
    """
    Taken from https://stackoverflow.com/questions/13685386/how-to-set-the-equal-aspect-ratio-for-all-axes-x-y-z

    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.55 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def draw_system(
    r_sensors,
    r_dipole,
    m,
    N=1000,
    aspect=[4, 4, 3],
    sensor_array_size=(200, 12),
    loop_scale=1,
    lim=[None, None, None],
    labels=None,
    savename=None,
    figsize=(7.11, 5),
    img_pos=[0.00, -0.1, 0.2, 0.2],
):
    fig = plt.figure(figsize=figsize)

    # # reference coordinates
    image = plt.imread(COORDINATE_IMAGE)
    # image = np.rot90(image, k=3)
    ax = plt.axes(
        img_pos, frameon=False, zorder=999
    )  # Change the numbers in this array to position your image [left, bottom, width, height])
    ax.imshow(image)
    ax.axis("off")

    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        r_sensors[:, 0],
        r_sensors[:, 2],
        r_sensors[:, 1],
        color="gray",
        s=sensor_array_size[0],
        marker="s",
        label="Sensors",
        edgecolor="black",
        # depthshade=False,
        zorder=1,
        alpha=0.8,
    )
    # Label each sensor according to its index
    for i, coord in enumerate(r_sensors):
        ax.text(
            coord[0],
            coord[2],
            coord[1],
            f"{i+1}",
            color="m",
            fontsize=sensor_array_size[1],
            weight="bold",
            va="center",
            ha="center",
            zorder=10,
        )

    # Plot dipole
    assert r_dipole.ndim == m.ndim
    if r_dipole.shape[0] > m.shape[0]:
        m = np.repeat(m, r_dipole.shape[0], axis=0)
        print("Warning: m shape does not match r_dipole shape. Repeating m.")
    if r_dipole.shape[0] < m.shape[0]:
        r_dipole = np.repeat(r_dipole, m.shape[0], axis=0)
        print("Warning: r_dipole shape does not match m shape. Repeating r_dipole.")

    if r_dipole.ndim == 2:
        r_dipole = r_dipole[np.newaxis, :, :]
        m = m[np.newaxis, :, :]

    r_dipole = np.transpose(r_dipole, (1, 0, 2))
    m = np.transpose(m, (1, 0, 2))
    colors = np.array(["tab:red", "tab:blue"])[: len(r_dipole)]

    for i, (dipole, moment, color) in enumerate(zip(r_dipole.copy(), m.copy(), colors)):
        if dipole.shape[0] == 1:
            ax.quiver(
                dipole[0, 0],
                dipole[0, 2],
                dipole[0, 1],
                moment[0, 0],
                moment[0, 2],
                moment[0, 1],
                color=color,
                length=0.02,
                normalize=True,
                label=f"{labels[i]} Dipole" if labels else "Dipole",
                linewidths=1.5,
                arrow_length_ratio=0.4,
            )
        else:
            moment *= loop_scale / np.linalg.norm(moment)
            ax.plot(
                dipole[:N, 0] + moment[:N, 0],
                dipole[:N, 2] + moment[:N, 2],
                dipole[:N, 1] + moment[:N, 1],
                color=color,
                linestyle="dotted",
                label=f"{labels[i]} Dipole" if labels else "Dipole",
                alpha=0.5,
            )

        # Origin marker
        ax.scatter(
            dipole[:N, 0],
            dipole[:N, 2],
            dipole[:N, 1],
            c=color,
            marker="x",
            label=f"{labels[i]} Dipole Position" if labels else "Dipole Position",
        )
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    ax.set_zlabel("Y [m]")
    ax.invert_zaxis()
    ax.view_init(azim=-165, elev=20)
    ax.legend(loc="upper center", ncol=3)
    ax.zaxis.labelpad = 0
    # set_axes_equal(ax)
    ax.set_box_aspect(aspect)
    ax.set_xlim(lim[0])
    ax.set_ylim(lim[2])
    ax.set_zlim(lim[1])
    plt.tight_layout()
    fig.subplots_adjust(0.00, 0.00, 0.99, 0.99, wspace=0.15, hspace=0.4)

    if savename is not None:
        plt.savefig(savename, dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    else:
        plt.show()


def plot_vcg(
    filtered_signals,
    ecg=None,
    fs=1000,
    fetal_downsample_factor=1,
    savename=None,
    sig_name="VCG",
    plt_HS_coor=True,
    labels=["x", "y", "z"],
):
    N = min(int(3 * fs), len(filtered_signals))
    time = np.arange(N) / fs

    if ecg is None:
        segments = {
            "P-wave": ([0], [N]),
        }
        rpeaks = None
    else:
        # Segment the signal according to the waves
        cleaned = nk.ecg_clean(
            ecg, sampling_rate=int(fs / fetal_downsample_factor), method="vg"
        )
        _, rpeaks = nk.ecg_peaks(
            cleaned, sampling_rate=int(fs / fetal_downsample_factor), method="vg"
        )
        s, waves = nk.ecg_delineate(
            cleaned,
            rpeaks,
            sampling_rate=int(fs / fetal_downsample_factor),
            method="dwt",
        )
        segments = {
            "P-wave": (waves["ECG_P_Onsets"], waves["ECG_P_Offsets"]),
            "Q-wave": (waves["ECG_P_Offsets"], waves["ECG_Q_Peaks"]),
            "R-wave": (waves["ECG_Q_Peaks"], waves["ECG_S_Peaks"]),
            "S-wave": (waves["ECG_S_Peaks"], waves["ECG_T_Onsets"]),
            "T-wave": (waves["ECG_T_Onsets"], waves["ECG_T_Offsets"]),
            "Baseline": (waves["ECG_T_Offsets"][:-1], waves["ECG_P_Onsets"][1:]),
        }
    colors = {
        "P-wave": "tab:blue",
        "Q-wave": "tab:orange",
        "R-wave": "tab:red",
        "S-wave": "tab:green",
        "T-wave": "tab:purple",
        "Baseline": "tab:grey",
    }

    fig = plt.figure(figsize=(7.11, 3.5), dpi=500)
    gs = gridspec.GridSpec(4, 2, width_ratios=[2, 3], height_ratios=[1, 1, 1, 1])

    # reference coordinates
    if plt_HS_coor:
        image = plt.imread(COORDINATE_IMAGE)
        image = np.rot90(image, k=3)
        ax = plt.axes(
            [0.00, 0.15, 0.2, 0.2], frameon=True, zorder=999
        )  # Change the numbers in this array to position your image [left, bottom, width, height])
        ax.imshow(image)
        ax.axis("off")

    # 3D plot on the left, spanning the top three rows
    ax = fig.add_subplot(gs[:3, 0], projection="3d")
    ax.plot(
        filtered_signals[:N, 0],
        filtered_signals[:N, 1],
        filtered_signals[:N, 2],
        color=colors["Baseline"],
    )
    # Plot projections on the side planes
    ax.plot(
        filtered_signals[:N, 0],
        filtered_signals[:N, 1],
        zs=ax.get_zlim()[0] * 0.95,
        zdir="z",
        color="gray",
        linestyle="-",
        alpha=0.3,
        axlim_clip=True,
    )
    ax.plot(
        filtered_signals[:N, 0],
        filtered_signals[:N, 2],
        zs=ax.get_ylim()[1] * 1.05,
        zdir="y",
        color="gray",
        linestyle="-",
        alpha=0.3,
        axlim_clip=True,
    )
    ax.plot(
        filtered_signals[:N, 1],
        filtered_signals[:N, 2],
        zs=ax.get_xlim()[1] * 1.05,
        zdir="x",
        color="gray",
        linestyle="-",
        alpha=0.3,
        axlim_clip=True,
    )

    for wave, (start_vals, end_vals) in segments.items():
        for s_idx, e_idx in zip(start_vals, end_vals):
            if (
                not np.isnan(s_idx)
                and not np.isnan(e_idx)
                and int(s_idx) < N
                and int(e_idx) < N
            ):
                ax.plot(
                    filtered_signals[int(s_idx) : int(e_idx), 0],
                    filtered_signals[int(s_idx) : int(e_idx), 1],
                    filtered_signals[int(s_idx) : int(e_idx), 2],
                    color=colors[wave],
                )

    # Origin marker
    if rpeaks is not None:
        ax.scatter(0, 0, 0, c="k", marker="x", label="Origin")
        r_peak_idx = rpeaks["ECG_R_Peaks"][1]
        r_peak_vals = filtered_signals[r_peak_idx, :]
        ax.quiver(
            0,
            0,
            0,
            r_peak_vals[0],
            r_peak_vals[1],
            r_peak_vals[2],
            color="k",
            arrow_length_ratio=0.1,
            label="R Peak Arrow",
            linewidths=1.5,
        )
        ax.text(
            r_peak_vals[0],
            r_peak_vals[1],
            r_peak_vals[2] + 0.1,
            "Cardiac Vector",
            color="k",
            fontsize=10,
            weight="bold",
        )

    ax.set_xlabel(labels[0], labelpad=-2)
    ax.set_ylabel(labels[1], labelpad=-2)
    ax.set_zlabel(labels[2], labelpad=-2)
    # ax.set_title("{sig_name} Loop", pad=0)
    ax.tick_params(pad=0)
    # plt_config.set_axes_equal(ax)
    ax.view_init(azim=210, elev=30)
    ax.zaxis.labelpad = -2

    # Create subplots on the right
    axs = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    for i, label in enumerate(
        [rf"{sig_name}$_x$", rf"{sig_name}$_y$", rf"{sig_name}$_z$"]
    ):
        axs[i].plot(
            time[: int(N)], filtered_signals[: int(N), i], color=colors["Baseline"]
        )
        for wave, (start_vals, end_vals) in segments.items():
            for s_idx, e_idx in zip(start_vals, end_vals):
                if (
                    not np.isnan(s_idx)
                    and not np.isnan(e_idx)
                    and int(s_idx) < N
                    and int(e_idx) < N
                ):
                    axs[i].plot(
                        time[int(s_idx) : int(e_idx)],
                        filtered_signals[int(s_idx) : int(e_idx), i],
                        color=colors[wave],
                        label=wave if s_idx == start_vals[0] else "",
                    )

        axs[i].set_title(label)
        axs[i].set_ylabel("Amplitude")
        axs[i].grid()
    axs[0].legend(
        ncol=6, loc="upper center", bbox_to_anchor=(0.05, 2.2), fancybox=False
    )
    axs[-1].set_xlabel("Time [s]")

    gs.update(left=0.16, bottom=0.01, top=0.8, right=0.99, wspace=0.2, hspace=0.8)
    if savename is not None:
        plt.savefig(savename, dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    else:
        plt.show()


def plot_vector_loop(
    m_true,
    r_true,
    m_pred,
    r_pred,
    time,
    i_dipole=0,
    savename=None,
    title=r"Magnetic Moment ($\mathrm{\mu}$Am$^2$)",
):
    if isinstance(m_true, torch.Tensor):
        m_true = m_true.detach().cpu().numpy()
    if isinstance(r_true, torch.Tensor):
        r_true = r_true.detach().cpu().numpy()
    if isinstance(m_pred, torch.Tensor):
        m_pred = m_pred.detach().cpu().numpy()
    if isinstance(r_pred, torch.Tensor):
        r_pred = r_pred.detach().cpu().numpy()

    N = len(time)
    if r_pred.ndim == 2:
        r_pred = r_pred[np.newaxis, :, :]
    if r_pred.shape[0] == 1:
        r_pred = np.repeat(r_pred, N, axis=0)
    if r_true.ndim == 2:
        r_true = r_true[np.newaxis, :, :]
    if r_true.shape[0] == 1:
        r_true = np.repeat(r_true, N, axis=0)

    if i_dipole == "all":
        range_dipoles = range(r_pred.shape[1])
    else:
        range_dipoles = [i_dipole]

    for i_dipole in range_dipoles:
        fig = plt.figure(figsize=(7.11, 3), dpi=500)
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 3], height_ratios=[1, 1])

        # 3D plot on the left, spanning the top three rows
        ax = fig.add_subplot(gs[:3, 0], projection="3d")

        ax.plot(
            r_pred[:N, i_dipole, 0] + m_pred[:N, i_dipole, 0],
            r_pred[:N, i_dipole, 1] + m_pred[:N, i_dipole, 1],
            r_pred[:N, i_dipole, 2] + m_pred[:N, i_dipole, 2],
            color="k",
            linestyle="dotted",
            label="Estimated Loop",
        )
        # Origin marker
        ax.scatter(
            r_pred[:, i_dipole, 0],
            r_pred[:, i_dipole, 1],
            r_pred[:, i_dipole, 2],
            c="tab:grey",
            marker="x",
            label="Origin",
            alpha=0.5,
        )
        ax.set_xlabel("X", labelpad=-5)
        ax.set_ylabel("Y", labelpad=-5)
        ax.set_zlabel("Z", labelpad=5)
        ax.set_title("Vector Loop", pad=0)
        ax.tick_params(pad=0)
        ax.view_init(azim=210, elev=30)
        ax.zaxis.labelpad = -2

        # Create subplots on the right
        axs = [fig.add_subplot(gs[i, 1]) for i in range(2)]

        # Plot the estimated and true magnitude values
        for i, label in enumerate([r"$x$", r"$y$", r"$z$"]):
            axs[0].plot(
                time,
                m_pred[: int(N), i_dipole, i],
                label=f"Estimated {label}",
            )
            axs[0].plot(
                time,
                m_true[: int(N), i_dipole, i],
                linestyle="dotted",
                color=axs[0].lines[-1].get_color(),
                label=f"True {label}",
            )
        axs[0].set_title(title)
        axs[0].grid()
        axs[0].legend(
            ncol=6, loc="upper center", bbox_to_anchor=(0.05, 1.6), fancybox=False
        )

        # Plot the estimated and true position values
        for i, label in enumerate([r"$x$", r"$y$", r"$z$"]):
            axs[1].plot(
                time,
                r_pred[: int(N), i_dipole, i],
                label=label,
            )
            axs[1].plot(
                time,
                r_true[: int(N), i_dipole, i],
                linestyle="dotted",
                color=axs[1].lines[-1].get_color(),
            )
        axs[1].set_title("Position [m]")
        axs[1].grid()
        axs[-1].set_xlabel("Time [s]")

        gs.update(left=0.15, bottom=0.01, top=0.80, right=0.99, wspace=0.25, hspace=0.4)
        if savename is not None:
            savename_ = savename.replace(".pdf", f"_dipole_{i_dipole}.pdf")
            plt.savefig(savename_, dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
        else:
            plt.show()


def plot_sensor_signals(
    field,
    time=None,
    xlim=[0, 2],
    ylim=None,
    labels=[r"$x$", r"$y$", r"$z$"],
    xlabel="Time [s]",
    ylabel="Field [pT]",
    title="Field per Sensor and Location",
    sensor_names=None,
    savename=None,
    sharex=True,
    sharey=True,
    artifacts_mask=None,
):

    fig, axs = plt.subplots(
        4, 4, figsize=(7.11, 5), sharex=sharex, sharey=sharey, dpi=500
    )
    for i in range(field.shape[-2]):
        ax = axs[i // 4, i % 4]
        values = field[:, i, :]

        if time is None:
            time = np.arange(values.shape[0])

        mask = (time >= xlim[0]) & (time <= min(xlim[1], time[-1]))
        times = np.repeat(time[mask][:, None], values.shape[-1], axis=1)
        lines = ax.plot(times, values[mask], label=labels)

        if sensor_names is not None:
            ax.set_title(f"Sensor {sensor_names[i]}")
        else:
            ax.set_title(f"Sensor {i+1}")
        ax.grid()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if artifacts_mask is not None:
            ax.fill_between(
                time,
                np.nanmin(values[mask]) if not np.isnan(values[mask]).all() else 0,
                np.nanmax(values[mask]) if not np.isnan(values[mask]).all() else 0,
                where=artifacts_mask,
                color="red",
                alpha=0.3,
                label="Artifacts",
                linewidth=0,
            )
            for line in lines:
                line.set_linewidth(0.5)

    if ylim is None and not np.isnan(values[mask]).all():
        min_val = np.nanmin(field[mask, :, :])
        max_val = np.nanmax(field[mask, :, :])
        ylim_ = [int(np.floor(min_val)), int(np.ceil(max_val))]
        ax.set_ylim(ylim_)

    fig.legend(
        lines,
        labels=labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.98),
        fancybox=False,
    )
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel, x=0)
    fig.subplots_adjust(0.06, 0.08, 0.99, 0.87, wspace=0.15, hspace=0.4)

    if savename:
        plt.savefig(savename, dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    else:
        plt.show()

def plot_single_channel(
    gt_field,
    fields,
    channel=0,
    time=None,
    xlim=(0, 2),
    ylim=None,
    xlabel="Time [s]",
    ylabel="Field [pT]",
    method_labels=None,
    gt_label="GT",
    savename=None,
):
    import numpy as np
    import matplotlib.pyplot as plt

    # Ensure list
    if not isinstance(fields, list):
        fields = [fields]

    if method_labels is None:
        method_labels = [f"Method {i+1}" for i in range(len(fields))]

    gt = np.asarray(gt_field)
    if gt.ndim == 1:
        gt = gt[:, None]

    T = gt.shape[0]

    proc_fields = []
    for f in fields:
        arr = np.asarray(f)
        if arr.ndim == 1:
            arr = arr[:, None]
        arr = arr[:T]
        proc_fields.append(arr)

    if time is None:
        time = np.arange(T)

    mask = (time >= xlim[0]) & (time <= xlim[1])
    t = time[mask]

    # Extract channel
    gt_ch = gt[:, channel]

    # Auto ylim
    if ylim is None:
        vals = [gt_ch[mask]] + [f[mask, channel] for f in proc_fields]
        vals = np.concatenate(vals)
        if vals.size and not np.isnan(vals).all():
            ylim = (np.floor(vals.min()), np.ceil(vals.max()))

    # Plot
    fig, ax = plt.subplots(figsize=(7.11, 1.75), dpi=500)

    ax.plot(t, gt_ch[mask], color="k", lw=1.2, label=gt_label)

    cmap = plt.get_cmap("tab10")
    ls = ["-", "--", "-.", ":"]
    for i, f in enumerate(proc_fields):
        if channel < f.shape[1]:
            ax.plot(t, f[mask, channel], color=cmap(i), lw=1.5, label=method_labels[i], linestyle=ls[i % len(ls)])

    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    # add minorgrid
    ax.minorticks_on()
    ax.grid(which="minor", linestyle=":", linewidth=0.7, alpha=0.7)

    if savename:
        plt.savefig(savename, dpi=fig.dpi, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
    else:
        plt.show()

    return fig



def plot_dict(
    sensor_dict,
    time=None,
    xlim=[50, 52],
    ylim=[-10, 10],
    fetal_peaks=None,
    maternal_peaks=None,
    labels=[r"$x$", r"$y$", r"$z$"],
    xlabel="Time [s]",
    ylabel="Field [pT]",
    title="Field Measurements per Sensor and Location",
    savename=None,
    sharex=True,
    sharey=True,
):
    # # plot
    # fig, axs = plt.subplots(4, 4, figsize=(15, 10), sharex=True, sharey=True)
    # for i, (name, values) in enumerate(sensor_dict.items()):
    #     ax = axs[i // 4, i % 4]
    #     if time is None:
    #         time = np.arange(values.shape[0])
    #     mask = (time >= xlim[0]) & (time <= xlim[1])
    #     times = np.repeat(time[mask][:, None], values.shape[1], axis=1)
    #     ax.plot(times, values[mask], label=labels)
    #     ax.set_title(name)
    #     ax.grid()
    #     ax.set_xlim(xlim)
    #     ax.set_ylim(ylim)
    #     if fetal_peaks is not None:
    #         ax.vlines(fetal_peaks, ymin=-25, ymax=25, color="r", linestyle="--", alpha=0.3)
    #     if maternal_peaks is not None:
    #         ax.vlines(maternal_peaks, ymin=-25, ymax=25, color="b", linestyle="--", alpha=0.3)
    #     ax.legend(loc="upper right")
    # fig.supxlabel(xlabel)
    # fig.supylabel(ylabel)
    # fig.suptitle(title)
    # plt.tight_layout()
    # plt.savefig("grid.png", dpi=100)
    # #plt.show()
    from matplotlib.ticker import MaxNLocator

    fig, axs = plt.subplots(
        4, 4, figsize=(7.11, 5), sharex=sharex, sharey=sharey, dpi=500
    )
    for i, (name, values) in enumerate(sensor_dict.items()):
        ax = axs[i // 4, i % 4]

        if time is None:
            time = np.arange(values.shape[0])
        mask = (time >= xlim[0]) & (time <= xlim[1])
        times = np.repeat(time[mask][:, None], values.shape[1], axis=1)
        lines = ax.plot(times, values[mask], label=labels)

        ax.set_title(f"Sensor {name}")
        ax.grid()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if ylim is None and not np.isnan(values[mask]).all():
            min_val = np.nanmin(values[mask])
            max_val = np.nanmax(values[mask])
            ylim_ = [int(np.floor(min_val)), int(np.ceil(max_val))]
            ax.set_ylim(ylim_)

        if fetal_peaks is not None:
            ax.vlines(
                fetal_peaks, ymin=-25, ymax=25, color="r", linestyle="--", alpha=0.3
            )
        if maternal_peaks is not None:
            ax.vlines(
                maternal_peaks, ymin=-25, ymax=25, color="b", linestyle="--", alpha=0.3
            )

    fig.legend(
        lines,
        labels=labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.98),
        fancybox=False,
    )
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel, x=0)
    fig.subplots_adjust(0.06, 0.08, 0.99, 0.87, wspace=0.15, hspace=0.4)

    if savename:
        plt.savefig(savename, dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    else:
        plt.show()


def plot_channels(
    signals,
    time=None,
    xlim=[50, 52],
    ylim=None,
    fetal_peaks=None,
    maternal_peaks=None,
    ncols=2,
    label="Channel",
):
    if time is None:
        time = np.arange(signals.shape[0])
    nrows = (signals.shape[1] + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, nrows * 2), sharex=True)
    axs = axs.flatten()
    for i in range(signals.shape[1]):
        axs[i].plot(time, signals[:, i])
        axs[i].set_title(f"{label} {i}")
        axs[i].set_xlim(xlim)
        axs[i].set_ylim(ylim)
        axs[i].grid()
        axs[i].set_xlabel("Time [s]")
        axs[i].set_ylabel("Amplitude")

        if fetal_peaks is not None:
            axs[i].vlines(
                fetal_peaks, ymin=-25, ymax=25, color="r", linestyle="--", alpha=0.3
            )
        if maternal_peaks is not None:
            axs[i].vlines(
                maternal_peaks, ymin=-25, ymax=25, color="b", linestyle="--", alpha=0.3
            )
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    plt.tight_layout()
    plt.show()


def plot_dipole_positions_3d(
    r_fitted,
    r_sensors,
    r_init=None,
    bounds=None,
    time_idx=0,
    figsize=(3.5, 2.5),
    dpi=300,
    colors=None,
    labels=None,
    show_coord_img=True,
    coord_img_bbox=None,
    savename=None,
):
    """
    Plot 3D visualization of fitted dipole positions, sensors, and bounds box.
    
    Parameters
    ----------
    r_fitted : numpy.ndarray or torch.Tensor
        Fitted dipole positions with shape (T, n_dipoles, 3) or (n_dipoles, 3)
    r_sensors : numpy.ndarray or torch.Tensor
        Sensor positions with shape (n_sensors, 3)
    r_init : numpy.ndarray or torch.Tensor, optional
        Initial dipole positions with shape (T, n_dipoles, 3) or (n_dipoles, 3)
    bounds : numpy.ndarray, optional
        Bounds for each dipole with shape (n_dipoles, 3, 2) where bounds[i, j] = (min, max)
        If None, uses default fetal/maternal bounds
    time_idx : int, default=0
        Time index to plot (if positions vary over time)
    figsize : tuple, default=(3.5, 2.5)
        Figure size in inches
    dpi : int, default=300
        Figure DPI
    colors : list, optional
        Colors for each dipole. If None, uses ['red', 'blue']
    labels : list, optional
        Labels for each dipole. If None, uses ['Fetal', 'Maternal']
    show_coord_img : bool, default=True
        Whether to show coordinate system image inset
    coord_img_bbox : list, optional
        Bounding box for coordinate image [left, bottom, width, height]
        If None, uses [0.7, .05, 0.28, 0.28]
    savename : str, optional
        Path to save figure. If None, displays figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    ax : matplotlib.axes.Axes3D
        The 3D axes
    """
    # Convert tensors to numpy
    if isinstance(r_fitted, torch.Tensor):
        r_fitted = r_fitted.detach().cpu().numpy()
    if isinstance(r_sensors, torch.Tensor):
        r_sensors = r_sensors.detach().cpu().numpy()
    if r_init is not None and isinstance(r_init, torch.Tensor):
        r_init = r_init.detach().cpu().numpy()
    
    # Handle dimensions
    if r_fitted.ndim == 3:
        r_fitted = r_fitted[time_idx]
    if r_init is not None and r_init.ndim == 3:
        r_init = r_init[time_idx]
    
    # Default colors and labels
    if colors is None:
        colors = ['red', 'blue']
    if labels is None:
        labels = ['Fetal', 'Maternal']
    
    # Default bounds (fetal and maternal)
    if bounds is None:
        bounds = np.array([
            [(-0.2, 0.2), (-0.2, -0.01), (-0.2, 0.2)],   # Fetal bounds
            [(-0.4, 0.4), (-0.4, -0.01), (0.2, 0.7)]      # Maternal bounds
        ]) * 100  # Convert to cm
    
    # Coordinate image bbox
    if coord_img_bbox is None:
        coord_img_bbox = [0.7, .05, 0.28, 0.28]
    
    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Add coordinate system image inset
    if show_coord_img:
        try:
            image = plt.imread(COORDINATE_IMAGE)
            ax_img = plt.axes(coord_img_bbox, frameon=True, zorder=999)
            ax_img.imshow(image)
            ax_img.axis("off")
        except:
            logger.warning("Could not load coordinate system image")
    
    # Create 3D axis
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract box bounds (using first dipole bounds for min, second for z-max)
    bx_min, bx_max = bounds[0, 0, 0], bounds[0, 0, 1]
    by_min, by_max = bounds[0, 1, 0], bounds[0, 1, 1]
    bz_min, bz_max = bounds[0, 2, 0], bounds[1, 2, 1]
    
    # Draw bounds box
    box_vertices = np.array([
        [bx_min, by_min, bz_min], [bx_max, by_min, bz_min],
        [bx_max, by_max, bz_min], [bx_min, by_max, bz_min],
        [bx_min, by_min, bz_max], [bx_max, by_min, bz_max],
        [bx_max, by_max, bz_max], [bx_min, by_max, bz_max]
    ])
    box_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
    for start, end in box_edges:
        ax.plot([box_vertices[start, 0], box_vertices[end, 0]],
                [box_vertices[start, 2], box_vertices[end, 2]],
                [box_vertices[start, 1], box_vertices[end, 1]],
                color='black', linewidth=1, alpha=0.6)
    
    # Plot initial positions if provided
    if r_init is not None:
        r_init_cm = r_init * 100  # Convert to cm
        for i in range(min(r_init.shape[0], len(colors))):
            ax.scatter(r_init_cm[i, 0], r_init_cm[i, 2], r_init_cm[i, 1],
                       c='green' if i == 0 else 'blue', marker='D', s=20, 
                       edgecolors='red', linewidths=0.5, 
                       label=f'{labels[i]} Init')
    
    # Plot sensors
    r_sens_cm = r_sensors * 100  # Convert to cm
    ax.scatter(r_sens_cm[:, 0], r_sens_cm[:, 2], r_sens_cm[:, 1],
               c='grey', marker='s', s=10, edgecolors='black', 
               linewidths=0.5, label='Sensors')
    
    # Plot fitted dipoles
    r_fitted_cm = r_fitted * 100  # Convert to cm
    for i in range(min(r_fitted.shape[0], len(colors))):
        ax.scatter(r_fitted_cm[i, 0], r_fitted_cm[i, 2], r_fitted_cm[i, 1],
                   c=colors[i], s=20, marker='x', label=labels[i], 
                   edgecolors='black', linewidths=1, zorder=10)
    
    # Set labels and limits
    ax.set_xlabel('x [cm]', labelpad=-2)
    ax.set_ylabel('z [cm]', labelpad=-3)
    ax.set_zlabel('y [cm]', labelpad=-7)
    
    # Set view angles and invert axes to match coordinate system
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.invert_zaxis()
    ax.view_init(elev=20, azim=20)
    
    # Adjust tick parameters
    ax.tick_params(axis='y', pad=1)
    ax.tick_params(axis='z', pad=-1)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=2)
    
    plt.subplots_adjust(left=0.07, top=1.2, right=.7, bottom=-0.05)
    plt.tight_layout()
    
    if savename is not None:
        plt.savefig(savename, dpi=fig.dpi, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
    else:
        plt.show()
    
    return fig, ax


def plot_ica_method(
    avg_waveform, std_waveform, fs, heartbeats_dict, savename=None, key="fetal"
):
    # for i in range(len(avg_waveform)):
    #     if avg_waveform[i][int(len(avg_waveform)/2)] < 0:
    #         avg_waveform[i] = -avg_waveform[i]
    #         std_waveform[i] = -std_waveform[i]

    mean_beat = np.mean(np.abs(avg_waveform), axis=0)

    middle = int(len(mean_beat) / 2)
    time = (np.arange(len(mean_beat)) - middle) / fs * 1000

    fig, ax = plt.subplots(figsize=(7.11, 2.5), dpi=500)
    ax.axvline(x=0, color="grey", linestyle="--")
    for i in range(len(avg_waveform)):
        ax.plot(
            time,
            avg_waveform[i],
            # alpha=0.75,
            linewidth=1 / np.log2(np.log2(1 + len(mean_beat))),  # .75,
            color="k",
            # linestyle="--",
        )
        # ax.fill_between(
        #     time,
        #     avg_waveform[i] - std_waveform[i],
        #     avg_waveform[i] + std_waveform[i],
        #     alpha=0.2,
        # )
    ax.hlines(0, time[0], time[-1], color="k", linestyle="--", linewidth=0.75)
    # ax.set_xlim(time[0], time[-1])
    df = heartbeats_dict[key]["M_x"]["mean_beat"].copy()
    df.index = df.index * 1e3
    xmin, xmax = df.index.min(), df.index.max()
    ax.set_xlim(xmin, xmax)
    # ax.set_title(component)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel(r"Reconstructed Field [pT]")
    # ax.legend(ncol=4, loc="lower left")
    ax.grid()
    ax.grid(
        which="minor", linestyle=":", linewidth=0.5
    )  # Add gridlines between major ticks
    ax.minorticks_on()  # Enable minor ticks without adding labels
    if savename is not None:
        plt.savefig(savename, dpi=fig.dpi, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
    else:
        plt.show()


def plot_sensor_signals_comparison(
    fields,
    time=None,
    xlim=[0, 2],
    ylim=None,
    labels=[r"$x$", r"$y$", r"$z$"],
    field_labels=None,
    xlabel="Time [s]",
    ylabel="Field [pT]",
    sensor_names=None,
    savename=None,
    sharey=True,
    artifacts_mask=None,
):
    """Creates comparison grid: 16 rows (sensors) x n_fields columns (methods)."""
    if not isinstance(fields, list):
        fields = [fields]
    
    if field_labels is None:
        field_labels = [f"Field {i+1}" for i in range(len(fields))]
    
    n_fields = len(fields)
    n_sensors = fields[0].shape[-2]
    
    # 16 rows (sensors) x n_fields columns (methods)
    fig, axs = plt.subplots(
        n_sensors, n_fields, 
        figsize=(1.5 * n_fields, 1 * n_sensors), 
        sharex=True, 
        sharey=sharey, 
        dpi=500
    )
    
    # Handle single field case
    if n_fields == 1:
        axs = axs[:, None]
    
    if time is None:
        time = np.arange(fields[0].shape[0])
    mask = (time >= xlim[0]) & (time <= min(xlim[1], time[-1]))
    
    # Calculate shared ylim across all fields
    if ylim is None and sharey:
        all_values = np.concatenate([f[mask, :, :] for f in fields])
        if not np.isnan(all_values).all():
            min_val = np.nanmin(all_values)
            max_val = np.nanmax(all_values)
            ylim = [int(np.floor(min_val)), int(np.ceil(max_val))]
    
    for field_idx, field in enumerate(fields):
        for sensor_idx in range(n_sensors):
            ax = axs[sensor_idx, field_idx]
            
            values = field[:, sensor_idx, :]
            times = np.repeat(time[mask][:, None], values.shape[-1], axis=1)
            lines = ax.plot(times, values[mask], label=labels, linewidth=0.8)
            
            # Column titles (method names)
            if sensor_idx == 0:
                ax.set_title(field_labels[field_idx], fontsize=9, fontweight='bold')
            
            # Row labels (sensor names)
            if field_idx == 0:
                if sensor_names is not None:
                    ax.set_ylabel(f"S{sensor_names[sensor_idx]}", fontsize=8)
                else:
                    ax.set_ylabel(f"S{sensor_idx+1}", fontsize=8)
            
            ax.grid(alpha=0.3)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.tick_params(labelsize=7)
            
            # Only show tick labels on edges
            # if sensor_idx < n_sensors - 1:
            #     ax.set_xticklabels([])
            #if field_idx > 0:
            #    ax.set_yticklabels([])
            
            if artifacts_mask is not None:
                ax.fill_between(
                    time,
                    ylim[0] if ylim else np.nanmin(values[mask]),
                    ylim[1] if ylim else np.nanmax(values[mask]),
                    where=artifacts_mask,
                    color="red",
                    alpha=0.2,
                    linewidth=0,
                )
    
    # Single legend at top
    fig.legend(
        lines,
        labels=labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.995),
        fancybox=False,
        frameon=False,
    )
    
    fig.supxlabel(xlabel, y=0.02)
    fig.supylabel(ylabel, x=0.02)
    fig.subplots_adjust(0.08, 0.05, 0.98, 0.97, wspace=0.05, hspace=0.15)
    
    if savename:
        plt.savefig(savename, dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    else:
        plt.show()

def plot_averaged_beats_comparison(
    averaged_beats,
    std_beats=None,
    method_labels=None,
    time=None,
    include_magnitude=True,
    xlabel="Time [ms]",
    ylabel=r"Mean Beat [pT]",
    savename=None,
    sharey=True,
    ncols=1,
    xlim=None,
):
    """
    Compare averaged beats across multiple methods.
    
    Parameters
    ----------
    averaged_beats : list of arrays
        List of (time, channels) arrays, one per method
    std_beats : list of arrays, optional
        List of (time, channels) arrays containing standard deviations
    method_labels : list of str, optional
        Labels for each method
    time : array, optional
        Time vector in seconds (will be converted to ms)
    channel_labels : list of str
        Labels for each channel (default: x, y, z)
    include_magnitude : bool
        Whether to plot magnitude
    ncols : int
        Number of columns for the plot layout
    """
    if not isinstance(averaged_beats, list):
        averaged_beats = [averaged_beats]
    
    if std_beats is not None:
        if not isinstance(std_beats, list):
            std_beats = [std_beats]
    
    if method_labels is None:
        method_labels = [f"Method {i+1}" for i in range(len(averaged_beats))]
    
    n_methods = len(averaged_beats)
    
    
    # Calculate rows needed
    nrows = int(np.ceil(n_methods / ncols))
    
    # Create subplots
    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(7.11, 2.5 * nrows),
        sharex=True,
        sharey=sharey,
        dpi=300
    )
    
    # Flatten axs for consistent indexing
    if nrows * ncols == 1:
        axs = np.array([axs])
    else:
        axs = np.array(axs).flatten()
    
    # Create time vector if not provided
    if time is None:
        time = np.arange(averaged_beats[0].shape[0]) / 1000  # Assume 1kHz sampling
    
    time_ms = time * 1e3  # Convert to ms
    
    for method_idx, avg_beat in enumerate(averaged_beats):
        ax = axs[method_idx]
        n_channels = avg_beat.shape[1]
        
        # Get std for this method if available
        current_std = None
        if std_beats is not None and method_idx < len(std_beats):
             current_std = std_beats[method_idx]

        # Plot each channel
        for ch_idx in range(n_channels):
            ax.plot(
                time_ms, 
                avg_beat[:, ch_idx], 
                linewidth=0.5
            )
            
            if current_std is not None:
                ax.fill_between(
                    time_ms,
                    avg_beat[:, ch_idx] - current_std[:, ch_idx],
                    avg_beat[:, ch_idx] + current_std[:, ch_idx],
                    alpha=0.2
                )
        
        # Plot magnitude if requested
        if include_magnitude:
            magnitude = np.linalg.norm(avg_beat, axis=1)
            ax.plot(
                time_ms,
                magnitude,
                label="Magnitude",
                color="k",
                linestyle="--",
                linewidth=0.5
            )
        
        # Zero line
        ax.axhline(0, color="k", linestyle="--", alpha=0.3, linewidth=0.8)
        
        # Formatting
        ax.set_ylabel(f"{method_labels[method_idx]}\n{ylabel}", fontsize=9)
        ax.legend(ncol=n_channels + (1 if include_magnitude else 0), 
                 loc="lower left", fontsize=8, frameon=False)
        ax.grid(True, alpha=0.3)
        ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.2)
        ax.minorticks_on()
    
    # Hide any unused subplots
    for i in range(n_methods, len(axs)):
        axs[i].axis("off")
        
    # X-label
    if ncols == 1:
        # For single column, put label on the bottom plot (which is index n_methods-1)
        axs[n_methods - 1].set_xlabel(xlabel, fontsize=10)
    else:
        # For grid, use a centered figure-level label
        fig.text(0.5, 0.04, xlabel, ha="center", fontsize=10)

    if xlim is not None:
        for ax in axs[:n_methods]:
            ax.set_xlim(xlim)
    
    fig.subplots_adjust(0.1, 0.08, 0.98, 0.96, hspace=0.15, wspace=0.25)
    
    if savename is not None:
        plt.savefig(savename, dpi=fig.dpi, bbox_inches="tight", pad_inches=0.01)
        plt.close()
    else:
        plt.show()