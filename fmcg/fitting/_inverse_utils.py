"""
Implementation of various regularization penalties for the inverse problem.
"""

import numpy as np
import torch

from ..utils.utils import MutualInformation


class CosineSchedulerWithWarmup:
    def __init__(self, T_warmup, T_max, vmax):
        self.T_warmup = T_warmup
        self.T_max = T_max
        self.vmax = vmax

    def __call__(self, epoch):
        if epoch < 0:
            return 0.0
        elif epoch < self.T_warmup:
            return (
                self.vmax
                * (1 - np.cos(np.pi * (epoch + 1) / max(self.T_warmup, 1)))
                / 2
            )
        elif epoch <= self.T_max:
            return (
                self.vmax
                * (
                    1
                    + np.cos(
                        np.pi
                        * (epoch - self.T_warmup + 1)
                        / max(self.T_max - self.T_warmup, 1)
                    )
                )
                / 2
            )
        else:
            return 0.0


def bilateral_tv(x, sigma_d, sigma_r, n=1, weight=1.0, p=1, dim=0, normalize=True):
    """
    Computes the bilateral total variation for a 1D signal x at a given order.

    Args:
        x (torch.Tensor): Tensor of shape (N,) or (T, num_dipoles, C)
        sigma_d (float): Spatial standard deviation.
        sigma_r (float): Intensity standard deviation.
        n (int): Order of the finite difference.
        weight (float|torch.Tensor): Weight for each time series in x.
        p (int): Norm order.
        dim (int): Dimension along which to compute the TV (default 0 = time).
        normalize (bool): If True, normalize by time^(1/p) and number of components.
                         This makes the regularization strength independent of signal length
                         and number of dipoles.
    """
    if isinstance(weight, (int, float, list)):
        weight = torch.tensor(weight).to(x.device)
    if weight.ndim < x.ndim - 1:
        weight = weight.unsqueeze(-1)

    diff = torch.diff(x, n, dim=dim)  # shape (N - order,)

    # The "spatial" distance is the order (since indices differ by order)
    spatial_weight = np.exp(-(n**2) / (2 * sigma_d**2))

    # The intensity-dependent weight is computed per sample:
    intensity_weight = torch.exp(-(diff**2) / (2 * sigma_r**2))

    btv = weight * torch.sum(
        spatial_weight * intensity_weight * diff.abs() ** p, dim=dim
    ) ** (1 / p)

    if normalize:
        # Normalize accounting for p-norm scaling
        num_diffs = diff.shape[dim]  # Number of temporal differences
        num_components = diff.numel() // num_diffs  # All non-time elements
        btv = btv / ((num_diffs ** (1.0 / p)) * num_components)

    return btv.sum()


def lagged_windowed_correlation(a, b, max_lag=0, step=1, W=1000):
    """
    Compute windowed correlation over different lags without truncating signals.
    Uses zero padding so that all lags are computed on the same-length data.
    """
    import torch.nn.functional as F

    # Transpose to (C, T)
    a = a.transpose(0, 1)
    b = b.transpose(0, 1)

    num_lags = 2 * max_lag + 1

    # Pad both signals so unfolding with lags won't truncate edges
    pad_amt = max_lag
    a_padded = F.pad(a, (pad_amt, pad_amt))  # (C, T + 2*max_lag)
    b_padded = F.pad(b, (pad_amt, pad_amt))  # same

    # Create lagged views of a
    a_unfolded = a_padded.unfold(1, num_lags, step)  # (C, T, num_lags)
    overlap_length = a_unfolded.shape[1]

    if W < 1:
        W = overlap_length
    if W > overlap_length:
        W = overlap_length

    # Align b so each lag is centered
    b_unfolded = b_padded[:, pad_amt : pad_amt + overlap_length]

    # --- WINDOWING ---
    a_unfolded = a_unfolded.transpose(1, 2)  # (C, num_lags, overlap_length)
    a_unfolded = a_unfolded.unfold(2, W, W // 2)  # (C, num_lags, num_windows, W)
    a_unfolded = a_unfolded.permute(2, 1, 3, 0)  # (num_windows, num_lags, W, C)

    b_windows = b_unfolded.unfold(1, W, W // 2)  # (C, num_windows, W)
    b_windows = b_windows.unsqueeze(1).expand(-1, num_lags, -1, -1)
    b_windows = b_windows.permute(2, 1, 3, 0)  # (num_windows, num_lags, W, C)

    # --- CORRELATION ---
    a_centered = a_unfolded - a_unfolded.mean(dim=2, keepdim=True)
    b_centered = b_windows - b_windows.mean(dim=2, keepdim=True)

    cov = torch.einsum("wltc,wltd->wlcd", a_centered, b_centered) / W
    std1 = torch.sqrt((a_centered.float() ** 2).mean(dim=2))
    std2 = torch.sqrt((b_centered.float() ** 2).mean(dim=2))

    correlations = cov / (std1.unsqueeze(-1) * std2.unsqueeze(-2) + 1e-12)
    correlations = correlations.square()
    return correlations.mean().sqrt()


def fft_windowed_correlation(x, y, W=None, hop=None, per_lag=False, lag_std=100):
    """
    Compute windowed, lagged correlation using FFT, aligned with classical windowed method.

    Args:
        x, y: torch tensors of shape (T, C) -- T time points, C channels
        W: window size. If None or >= T, use the whole signal as a single window
        hop: hop size (ignored if W >= T)

    Returns:
        scalar: sqrt of mean squared correlations over windows and channels
    """
    T, C = x.shape

    # Use full signal if W is None or too large
    if W is None or W >= T:
        W = T
        num_windows = 1
        x_windows = x.unsqueeze(0)  # (1, T, C)
        y_windows = y.unsqueeze(0)
    else:
        if hop is None:
            hop = W // 2
        # num_windows = 1 + (T - W) // hop
        x_windows = x.unfold(0, W, hop).permute(0, 2, 1)  # (num_windows, W, C)
        y_windows = y.unfold(0, W, hop).permute(0, 2, 1)  # same

    # Center each window
    x_centered = x_windows - x_windows.mean(dim=1, keepdim=True)
    y_centered = y_windows - y_windows.mean(dim=1, keepdim=True)

    # # windowing
    # hann = torch.signal.windows.hann(x_centered.shape[1], device=x.device).view(1,-1,1)
    # x_centered *= hann
    # y_centered *= hann

    # FFT along window dimension
    Xf = torch.fft.rfft(x_centered, n=W, dim=1)
    Yf = torch.fft.rfft(y_centered, n=W, dim=1)

    # cross spectrum
    Sxy_freq = Xf.unsqueeze(3) * Yf.conj().unsqueeze(2)  # (num_windows, N, Cx, Cy)

    r_xy = Sxy_freq.abs().square().sum(dim=1, keepdim=True)

    # # Standard deviations per window
    # std_x = x_centered.std(dim=1, keepdim=True).unsqueeze(2)
    # std_y = y_centered.std(dim=1, keepdim=True).unsqueeze(1)

    # return (r_xy / (W**2 * std_x * std_y + 1e-12)).mean().sqrt()  # (num_windows, 1, Cx, Cy)

    # Cross-correlation using IFFT (Wiener–Khinchin for cross-correlation)
    r_xy = torch.fft.irfft(Sxy_freq, n=W, dim=1).real  # (num_windows, W, Cx, Cy)

    if lag_std > 0:
        lag_weight = torch.signal.windows.gaussian(
            x_centered.shape[1], std=lag_std, device=x.device
        ).view(1, -1, 1, 1)
        lag_weight /= lag_weight.sum()  # Normalize Gaussian window
    else:
        lag_weight = 1
        lag_weight /= x_centered.shape[1]
    r_xy = torch.fft.fftshift(r_xy, dim=1) * lag_weight

    # Standard deviations per window
    std_x = x_centered.std(dim=1, keepdim=True).unsqueeze(2)
    std_y = y_centered.std(dim=1, keepdim=True).unsqueeze(1)

    # Normalized correlation
    corr = r_xy / (std_x * std_y + 1e-12)

    if per_lag:
        return corr

    return corr.square().mean(dim=1).mean().sqrt()


def selective_lagged_windowed_correlation(a, b, W=1000, lag_idx=[0]):
    import torch.nn.functional as F

    # Transpose to (C, T)
    a = a.transpose(0, 1)
    b = b.transpose(0, 1)

    if not isinstance(lag_idx, (torch.Tensor)):
        lag_idx = torch.tensor(lag_idx, device=a.device)
    num_lags = lag_idx.max() - lag_idx.min() + 1
    max_lag = lag_idx.abs().max()

    # Pad both signals so unfolding with lags won't truncate edges
    pad_amt = max_lag
    a_padded = F.pad(a, (pad_amt, pad_amt))  # (C, T + 2*max_lag)
    b_padded = F.pad(b, (pad_amt, pad_amt))  # same

    # Create lagged views of a
    a_unfolded = a_padded.unfold(1, num_lags, 1)  # (C, T, num_lags)
    a_unfolded = a_unfolded[:, :, lag_idx]  # Select lags
    overlap_length = a_unfolded.shape[1]

    if W < 1:
        W = overlap_length
    if W > overlap_length:
        W = overlap_length

    # Align b so each lag is centered
    b_unfolded = b_padded[:, pad_amt : pad_amt + overlap_length]

    # --- WINDOWING ---
    a_unfolded = a_unfolded.transpose(1, 2)  # (C, num_lags, overlap_length)
    a_unfolded = a_unfolded.unfold(2, W, W // 2)  # (C, num_lags, num_windows, W)
    a_unfolded = a_unfolded.permute(2, 1, 3, 0)  # (num_windows, num_lags, W, C)

    b_windows = b_unfolded.unfold(1, W, W // 2)  # (C, num_windows, W)
    b_windows = b_windows.unsqueeze(1).expand(-1, len(lag_idx), -1, -1)
    b_windows = b_windows.permute(2, 1, 3, 0)  # (num_windows, num_lags, W, C)

    # --- CORRELATION ---
    a_centered = a_unfolded - a_unfolded.mean(dim=2, keepdim=True)
    b_centered = b_windows - b_windows.mean(dim=2, keepdim=True)

    cov = torch.einsum("wltc,wltd->wlcd", a_centered, b_centered) / W
    std1 = torch.sqrt((a_centered.float() ** 2).mean(dim=2))
    std2 = torch.sqrt((b_centered.float() ** 2).mean(dim=2))

    correlations = cov / (std1.unsqueeze(-1) * std2.unsqueeze(-2) + 1e-12)
    correlations = correlations.square()
    return correlations.mean().sqrt()


def correlation_penalty(x, weight=1.0, lag=50, window=5000, step=1):
    """
    Computes correlation penalty, optionally summing over multiple time lags.

    Args:
        x: Input tensor of shape (T, 2, C) where T is time, 2 is number of channels, C is features
        weight: Weight for the penalty
        lag: the time samples to lag the second channel by. If None, no lag is applied.

    Returns:
        Penalty value (scalar tensor)
    """
    if isinstance(weight, (int, float, list)):
        weight = torch.tensor(weight).to(x.device)

    if lag is None:
        lag = 0

    # return weight * fft_windowed_correlation(x[:, 0, :], x[:, 1, :], W=window, hop=None, lag_std=lag)
    # return weight * selective_lagged_windowed_correlation(x[:, 0, :], x[:, 1, :], W=window, lag_idx=[-25,-20,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25])
    return weight * lagged_windowed_correlation(
        x[:, 0, :], x[:, 1, :], max_lag=lag, W=window, step=step
    )


def negentropy(m, weight=1.0, axis=0, num_samples=100_000):
    m = torch.linalg.norm(m, ord=2, dim=-1)
    # Normalize m to zero mean and unit variance along the specified axis
    m_mean = m.mean(dim=axis, keepdim=True)
    m_std = (
        m.std(dim=axis, keepdim=True) + torch.finfo(torch.float32).eps
    )  # avoid division by zero
    m_norm = (m - m_mean) / m_std

    # Compute the expectation E{log cosh(m_norm)} via sample averaging
    E_G_m = torch.mean(torch.log(torch.cosh(m_norm)), dim=axis)

    # Compute E{log cosh(ν)} for a standard Gaussian using Monte Carlo sampling
    # ν is sampled from N(0,1)
    nu_samples = torch.randn(num_samples, device=m.device)
    E_G_nu = torch.mean(torch.log(torch.cosh(nu_samples)))

    # Negentropy approximation for each "channel" (averaging over axis)
    negentropy = (E_G_m - E_G_nu) ** 2

    # Return the mean negentropy across all channels (or you could return the per-channel value)
    return torch.mean(-weight * negentropy)


def kurtosis(x, weight=1.0):
    x = torch.linalg.norm(x, ord=2, dim=-1)
    mean = torch.mean(x, axis=0)
    diffs = x - mean
    var = torch.mean(torch.pow(diffs, 2.0), axis=0)
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    # excess kurtosis
    kurtoses = torch.mean(torch.pow(zscores, 4.0), axis=0) - 3.0
    return -(weight * kurtoses).mean()


def mutual_information_penalty(x, weight=1.0, sigma=0.01, num_bins=256, normalize=True):
    if isinstance(weight, (int, float, list)):
        weight = torch.tensor(weight).to(x.device)

        # Mutual information
    mi = MutualInformation(sigma=sigma, num_bins=num_bins, normalize=normalize)
    mi = mi.to(x.device)
    a = torch.linalg.norm(x[:, 0, :], ord=2, dim=-1).unsqueeze(0).unsqueeze(-1)
    b = torch.linalg.norm(x[:, 1, :], ord=2, dim=-1).unsqueeze(0).unsqueeze(-1)
    # a,b = x[:,0].transpose(0,1).unsqueeze(-1), x[:,1].transpose(0,1).unsqueeze(-1)
    return weight * mi(a, b)


def tv(x, n=1, p=2, weight=1.0, dim=0, normalize=True):
    """
    Computes the total variation of a signal x at a given order.

    Args:
        x (torch.Tensor): Tensor of shape (N,)
        n (int): Order of the finite difference.
        p (int): Norm order.
        weight (float|torch.Tensor): Weight for each time series in x.
        dim (int): Dimension along which to compute the TV.
        normalize (bool): If True, normalize by time^(1/p) and number of components.
                         This makes the regularization strength independent of signal length
                         and number of parameters (dipoles/spatial dimensions).
    """

    if isinstance(weight, (int, float, list)):
        weight = torch.tensor(weight).to(x.device)
    if weight.ndim < x.ndim - 1:
        weight = weight.unsqueeze(-1)

    diff = torch.diff(x, n=n, dim=dim)

    # tv = weight *torch.nn.HuberLoss(reduction='none', delta=)(diff, torch.zeros_like(diff)).sum(dim=dim)/0.00001
    # tv = weight * torch.linalg.vector_norm(diff[diff.abs() < 0.0001], dim=dim, ord=p)
    tv = weight * torch.linalg.vector_norm(diff, dim=dim, ord=p)

    if normalize:
        # Normalize accounting for p-norm scaling
        num_diffs = diff.shape[dim]  # Number of temporal differences
        num_components = diff.numel() // num_diffs  # All non-time elements
        tv = tv / ((num_diffs ** (1.0 / p)) * num_components)

    return tv.sum()


def mse(x, y, p=2):
    """
    Computes the mean squared error between two signals x and y.

    Args:
        x (torch.Tensor): Tensor of shape (N,)
        y (torch.Tensor): Tensor of shape (N,)
        p (int): Norm order
    """
    # return torch.linalg.vector_norm(x - y) ** 2 / torch.linalg.vector_norm(y) ** 2
    return (
        0.5
        * (torch.linalg.vector_norm(x - y, ord=p, dim=(1)) ** p).mean()
        / (torch.linalg.vector_norm(y, ord=p, dim=(1)) ** p).mean()
    )
    # return .5 * (x-y).square().mean() / y.square().mean()


def huber(x, y, delta=1.0):
    return (
        torch.nn.functional.huber_loss(x, y, delta=delta, reduction="none")
        .mean(axis=0)
        .mean()
    )


def norm(x, p=2, weight=1.0, dim=(1), normalize=True):
    """
    Computes the p-norm of a signal x.

    Args:
        x (torch.Tensor): Tensor of shape (N,)
        p (int): Norm order
        weight (float|torch.Tensor): Weight for each time series in x.
        dim (int): Dimension along which to compute the norm.
        normalize (bool): If True, normalize by T^(1/p) and number of components.
                         This makes the regularization strength independent of signal length
                         and number of parameters.
    """
    if isinstance(weight, (int, float, list)):
        weight = torch.tensor(weight).to(x.device)
    if weight.ndim < x.ndim - 1:
        weight = weight.unsqueeze(-1)

    result = weight * torch.linalg.vector_norm(x, ord=p, dim=dim)

    if normalize:
        # Normalize accounting for p-norm scaling
        # Assume first dimension is time
        time_dim = 0
        T = x.shape[time_dim]
        num_components = x.numel() // T  # All non-time elements
        result = result / ((T ** (1.0 / p)) * num_components)

    return result.sum()


def lncosh(x, y, lamb=3.0):
    """
    Log-cosh penalty.
    Args:
        x (torch.Tensor): Tensor of shape (N,)
        y (torch.Tensor): Tensor of shape (N,)
        lamb (float): Sharpness parameter. Higher values approximate L1 more closely.
    """

    return (1/lamb *torch.log(torch.cosh(lamb*(x-y)))).mean() / (1/lamb *torch.log(torch.cosh(lamb*y))).mean()


def lhsaf(x, y, lamb=3):
    """
    Log-hyperbolic secantant penalty.
    Args:
        x (torch.Tensor): Tensor of shape (N,)
        y (torch.Tensor): Tensor of shape (N,)
        lamb (float): Sharpness parameter. Higher values approximate L1 more closely.
    """
    def sech(x):
        # numerically stable sech
        return 2.0 * torch.exp(-x.abs()) / (1.0 + torch.exp(-2.0 * x.abs()))
    return (-1/lamb * torch.log1p(sech(lamb*(x-y)))).mean() / (1/lamb * torch.log1p(sech(lamb*y))).mean()