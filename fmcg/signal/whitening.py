from fmcg.utils.utils import sig_power


import numpy as np


def get_whitening_matrix(sig, eps=1e-6, method="PCA"):
    """
    Compute the whitening matrix for a given signal using various methods.

    Whitening is a transformation that decorrelates the input signal and 
    scales it to have unit variance. This function supports multiple 
    whitening methods, including PCA, ZCA, ZCA-corr, and PCA-corr.

    Parameters:
        sig : ndarray
            Input signal array of shape (n_samples, n_features).
        eps : float, optional
            Small constant added to eigenvalues for numerical stability. Default is 1e-9.
        method : str, optional
            Whitening method to use. Options are:
            - "PCA": Principal Component Analysis whitening.
            - "ZCA": Zero-phase Component Analysis whitening.
            - "ZCA-corr": ZCA whitening with correlation matrix.
            - "PCA-corr": PCA whitening with correlation matrix.
            Default is "PCA".

    Returns:
        W : ndarray
            Whitening matrix of shape (n_features, n_features).

    Notes:
        - PCA whitening transforms the data into a space where the covariance 
        matrix is diagonal and scales the components to have unit variance.
        - ZCA whitening produces a transformation that is as close as possible 
        to the original data in terms of Euclidean distance.
        - ZCA-corr and PCA-corr use the correlation matrix instead of the 
        covariance matrix for whitening, which normalizes the data by its 
        standard deviation before applying the transformation.
    """
    R = np.cov(sig.T)
    U, S, _ = np.linalg.svd(R, hermitian=True)

    # fig = plt.figure(figsize=(3.05, 3), dpi=500)
    # plt.plot(S)
    # plt.xlabel("Eigenvalue index")
    # plt.ylabel("Noise Covariance Eigenvalues")
    # plt.yscale("log")
    # plt.grid()
    # plt.show()

    if method == "PCA":
        W = np.diag(1.0 / np.sqrt(S + eps)) @ U.T
    elif method == "ZCA":
        W = U @ np.diag(1 / np.sqrt(S + eps)) @ U.T
    elif method == "ZCA2":
        W = U @ np.diag(1 / (S + eps)) @ U.T
    elif method == "SSP":
        k = 11
        W = np.eye(U.shape[0]) - (U[:, :k] @ U[:, :k].T)
    elif method == "ZCA-corr":
        V_ = np.diag(1.0 / (np.std(R, axis=0) + eps))
        P = np.corrcoef(R, rowvar=False)
        UP, SP, _ = np.linalg.svd(P, hermitian=True)
        W = UP @ np.diag(1.0 / np.sqrt(SP + eps)) @ UP.T @ V_
    elif method == "PCA-corr":
        V_ = np.diag(1.0 / (np.std(R, axis=0) + eps))
        P = np.corrcoef(R, rowvar=False)
        UP, SP, _ = np.linalg.svd(P, hermitian=True)
        W = np.diag(1.0 / np.sqrt(SP + eps)) @ UP.T @ V_
    else:
        raise ValueError(f"Unknown whitening method: {method}")
    return W


def apply_whitening(field_maps, noise_maps, axis_mask=None, eps=1e-9, method="PCA", return_whitening_matrix=False, rescale=False):
    """
    Apply whitening to the field maps using the noise maps.

    Parameters:
        field_maps (np.ndarray): The field maps to be whitened.
        noise_maps (np.ndarray): The noise maps used for whitening.
        axis_mask (np.ndarray, optional): Mask to select specific axes for whitening. Default is None.
        eps (float, optional): Small value to avoid division by zero. Default is 1e-9.
        method (str, optional): Whitening method. Options are "PCA", "ZCA", "ZCA-corr", "PCA-corr". Default is "PCA".
        return_whitening_matrix (bool, optional): If True, return the whitening matrix. Default is False.
        rescale (bool, optional): If True, rescale the whitened field maps to match the energy of the original field maps. Default is False.

    Returns:
        np.ndarray: Whitened field maps.
    """
    field_maps_ = field_maps.copy()
    noise_maps_ = noise_maps.copy()

    # Reshape field and noise maps to 2D arrays
    if axis_mask is not None:
        field_maps_ = field_maps_[:, axis_mask==1]
        noise_maps_ = noise_maps_[:, axis_mask==1]
    else:
        field_maps_ = field_maps_.reshape(field_maps_.shape[0], -1)
        noise_maps_ = noise_maps_.reshape(noise_maps_.shape[0], -1)

    # subtract mean
    field_maps_ -= np.nanmean(field_maps_, axis=0)
    noise_maps_ -= np.nanmean(noise_maps_, axis=0)

    W = get_whitening_matrix(noise_maps_, eps=eps, method=method)

    # Apply whitening
    if axis_mask is not None:
        field_maps_w = np.ones_like(field_maps) * np.nan
        field_maps_w[:, axis_mask==1] = field_maps_ @ W.T
    else:
        field_maps_w = field_maps_ @ W.T
        field_maps_w = field_maps_w.reshape(field_maps.shape)

    if rescale:
        a = sig_power(field_maps, win=1000) / sig_power(field_maps_w, win=1000)

        #print(f"Rescale factor: {a}")
        field_maps_w *= a
        W *= a

    if return_whitening_matrix:
        return field_maps_w, W

    return field_maps_w

def unwhiten(whitened_field_maps, whitening_matrix, axis_mask=None):
    """
    Unwhiten the field maps using the provided whitening matrix.

    Parameters:
        whitened_field_maps (np.ndarray): The whitened field maps to be unwhitened.
        whitening_matrix (np.ndarray): The whitening matrix used for unwhitening.
        axis_mask (np.ndarray, optional): Mask to select specific axes for unwhitening. Default is None.

    Returns:
        np.ndarray: Unwhitened field maps.
    """
    
    if axis_mask is not None and whitened_field_maps.ndim == 3:
        return np.linalg.solve(whitening_matrix, whitened_field_maps[:, axis_mask==1].T).T
    else:
        return np.linalg.solve(whitening_matrix, whitened_field_maps.T).T
