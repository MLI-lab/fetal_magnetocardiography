"""
Vector Cardiogram (VCG) Transformation Module

This module provides bidirectional transformations between 12-lead ECG and 3-lead orthogonal VCG:

Forward Transformations (ECG → VCG):
    - Kors transformation (default): Standard method, widely validated
    - Inverse Dower transformation: Alternative method using pseudo-inverse

Inverse Transformations (VCG → ECG):
    - Dower reconstruction (default): Standard method for ECG reconstruction
    - Inverse Kors reconstruction: Uses pseudo-inverse of Kors matrix

Key Functions:
    - ecg_to_vcg(): Convert ECG to VCG using Kors or Inverse Dower method
    - vcg_to_ecg(): Reconstruct ECG from VCG using Dower or Inverse Kors method

References:
    - Kors JA, et al. Reconstruction of the Frank vectorcardiogram from standard
      electrocardiographic leads. Eur Heart J. 1990;11(12):1083-1092.
    - Dower GE, et al. On deriving the electrocardiogram from vectorcardiographic leads.
      Clin Cardiol. 1980;3(2):87-95.
"""

import numpy as np
import torch
from typing import Union, Optional, List, Tuple


# ============================================================================
# FORWARD TRANSFORMATIONS: ECG (8 leads) → VCG (3 leads)
# ============================================================================

# Kors transformation matrix
# Converts 8 standard ECG leads (I, II, V1, V2, V3, V4, V5, V6) to 3 orthogonal VCG leads (X, Y, Z)
# Reference: Kors JA, et al. Reconstruction of the Frank vectorcardiogram from standard
# electrocardiographic leads: diagnostic comparison of different methods.
# Eur Heart J. 1990;11(12):1083-1092.
KORS_MATRIX = np.array([
    [0.38, -0.07, -0.13, 0.05, -0.01, 0.14, 0.06, 0.54],      # X axis
    [-0.07, 0.93, 0.06, -0.02, -0.05, 0.06, -0.17, 0.13],     # Y axis
    [0.11, -0.23, -0.43, -0.06, -0.14, -0.20, -0.11, 0.31]    # Z axis
], dtype=np.float32)

# Inverse Dower transformation matrix
# This is the pseudo-inverse of the Dower matrix (used for VCG → ECG reconstruction)
# Lead order: ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
# Reference: Dower GE, et al. On deriving the electrocardiogram from vectorcardiographic leads.
# Clin Cardiol. 1980;3(2):87-95.
INVERSE_DOWER_MATRIX = np.array([
    [ 0.156, -0.010, -0.172, -0.074,  0.122,  0.231,  0.239,  0.194],  # X axis
    [-0.227,  0.887,  0.057, -0.019, -0.106, -0.022,  0.041,  0.048],  # Y axis
    [ 0.022,  0.102, -0.229, -0.310, -0.246, -0.063,  0.055,  0.108]   # Z axis
], dtype=np.float32)

# ============================================================================
# INVERSE TRANSFORMATIONS: VCG (3 leads) → ECG (8 leads)
# ============================================================================

# Dower reconstruction matrix
# Converts 3 orthogonal VCG leads (X, Y, Z) back to 8 standard ECG leads
# Lead order: ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
DOWER_MATRIX = np.array([
    [ 0.632, -0.235,  0.059],  # I
    [ 0.235,  1.066, -0.132],  # II
    [-0.515,  0.157, -0.917],  # V1
    [ 0.044,  0.164, -1.387],  # V2
    [ 0.882,  0.098, -1.277],  # V3
    [ 1.213,  0.127, -0.601],  # V4
    [ 1.125,  0.127, -0.086],  # V5
    [ 0.831,  0.076,  0.230]   # V6
], dtype=np.float32)


# Inverse Kors matrix (pseudo-inverse)
# Reconstructs 8 ECG leads from 3 VCG leads using pseudo-inverse of Kors matrix
# Lead order: ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
# Computed as: pinv(KORS_MATRIX)
INVERSE_KORS_MATRIX = np.linalg.pinv(KORS_MATRIX).astype(np.float32)

# Standard lead names in ECG recordings
# LUDB and ISP datasets use these 8 leads for VCG transformation
KORS_INPUT_LEADS = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Output VCG lead names
VCG_OUTPUT_LEADS = ['X', 'Y', 'Z']


def select_leads(signal: np.ndarray, lead_names: List[str], target_leads: List[str]) -> np.ndarray:
    """
    Select specific leads from a multi-lead signal (case-insensitive matching).

    Args:
        signal: ECG signal array [num_samples, num_channels]
        lead_names: List of lead names corresponding to channels
        target_leads: List of target lead names to extract

    Returns:
        Selected signal array [num_samples, len(target_leads)]

    Raises:
        ValueError: If any target lead is not found in lead_names
    """
    # Create a case-insensitive mapping of lead name to channel index
    # WFDB often stores lead names in lowercase, so we normalize for comparison
    lead_to_idx = {lead.upper(): idx for idx, lead in enumerate(lead_names)}

    # Normalize target leads to uppercase for matching
    target_leads_upper = [lead.upper() for lead in target_leads]

    # Check that all target leads are available
    missing_leads = [lead for lead in target_leads_upper if lead not in lead_to_idx]
    if missing_leads:
        raise ValueError(
            f"Missing leads: {missing_leads}. "
            f"Available leads: {[l.upper() for l in lead_names]}. "
            f"ECG-to-VCG transformation requires leads: {KORS_INPUT_LEADS}"
        )

    # Extract columns for target leads in the specified order
    selected_indices = [lead_to_idx[lead] for lead in target_leads_upper]
    return signal[:, selected_indices]


def ecg_to_vcg(signal: Union[np.ndarray, torch.Tensor],
               lead_names: Optional[List[str]] = None,
               method: str = 'kors') -> Union[np.ndarray, torch.Tensor]:
    """
    Transform 12-lead ECG to 3-lead orthogonal VCG using transformation matrix.

    IMPORTANT: Ensure that ECG signal is NOT normalized/scaled before VCG transformation,
    as the transformation matrix coefficients are calibrated for raw ECG amplitudes.
    Normalization will distort the orthogonal relationships in VCG space.

    Args:
        signal: ECG signal array [num_samples, num_channels] or torch tensor
        lead_names: List of lead names corresponding to channels.
                   If None, assumes channels are in order [I, II, III, aVR, aVL, aVF, V1-V6]
                   or just [I, II, V1-V6] for 8-channel data.
        method: Transformation method, either 'kors' or 'inverse_dower' (default: 'kors').
               - 'kors': Standard Kors transformation (recommended, most widely used)
               - 'inverse_dower': Alternative using pseudo-inverse of Dower matrix

    Returns:
        VCG signal array [num_samples, 3] with leads [X, Y, Z]
        Maintains input type (numpy array or torch tensor)

    Raises:
        ValueError: If method is invalid or required leads are missing

    Example:
        >>> # With explicit lead names using Kors method (default)
        >>> vcg = ecg_to_vcg(ecg_signal, lead_names=['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
        >>> print(vcg.shape)  # (num_samples, 3)

        >>> # Using inverse Dower method
        >>> vcg = ecg_to_vcg(ecg_signal, lead_names=lead_names, method='inverse_dower')

        >>> # Assuming standard 12-lead order
        >>> vcg = ecg_to_vcg(ecg_12lead)  # Automatically extracts [I, II, V1-V6]
    """
    # Convert torch tensor to numpy for processing
    is_torch = isinstance(signal, torch.Tensor)
    if is_torch:
        device = signal.device
        dtype = signal.dtype
        signal_np = signal.cpu().numpy()
    else:
        signal_np = signal
        dtype = signal.dtype

    # Infer lead names if not provided
    if lead_names is None:
        num_channels = signal_np.shape[1]
        if num_channels == 8:
            # Assume 8 leads in standard order
            lead_names = KORS_INPUT_LEADS
        elif num_channels == 12:
            # Standard 12-lead ECG: [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
            lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        else:
            raise ValueError(
                f"Cannot infer lead order for {num_channels} channels. "
                f"Expected 8 or 12 channels, or provide explicit lead_names."
            )

    # Select the 8 leads required for transformation
    selected_signal = select_leads(signal_np, lead_names, KORS_INPUT_LEADS)

    # Select transformation matrix based on method
    if method == 'kors':
        transform_matrix = KORS_MATRIX
    elif method == 'inverse_dower':
        transform_matrix = INVERSE_DOWER_MATRIX
    else:
        raise ValueError(
            f"Invalid method '{method}'. Must be 'kors' or 'inverse_dower'."
        )

    # Apply transformation matrix
    # selected_signal shape: [num_samples, 8]
    # transform_matrix shape: [3, 8]
    # Result shape: [num_samples, 3]
    vcg_signal = selected_signal @ transform_matrix.T

    # Convert back to original type and device if needed
    if is_torch:
        vcg_signal = torch.from_numpy(vcg_signal).to(dtype=dtype, device=device)
    else:
        vcg_signal = vcg_signal.astype(dtype)

    return vcg_signal


def vcg_to_ecg(signal: Union[np.ndarray, torch.Tensor],
               method: str = 'dower') -> Union[np.ndarray, torch.Tensor]:
    """
    Transform 3-lead orthogonal VCG back to 8-lead ECG using inverse transformation.

    This function reconstructs standard ECG leads from VCG leads. Two methods are available:
    - 'dower': Uses the Dower matrix (recommended for reconstruction)
    - 'kors': Uses pseudo-inverse of Kors matrix

    Args:
        signal: VCG signal array [num_samples, 3] with leads [X, Y, Z]
        method: Reconstruction method, either 'dower' or 'kors' (default: 'dower')

    Returns:
        ECG signal array [num_samples, 8] with leads ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        Maintains input type (numpy array or torch tensor)

    Raises:
        ValueError: If signal doesn't have 3 channels or method is invalid

    Example:
        >>> # Reconstruct ECG from VCG
        >>> vcg_signal = np.random.randn(1000, 3)  # X, Y, Z leads
        >>> ecg_signal = vcg_to_ecg(vcg_signal, method='dower')
        >>> print(ecg_signal.shape)  # (1000, 8)
        >>>
        >>> # Get lead names for reconstructed ECG
        >>> lead_names = KORS_INPUT_LEADS  # ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    Notes:
        - The Dower method is generally preferred for ECG reconstruction
        - The reconstructed ECG will approximate the original but may not be identical
        - Lead order in output: ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    """
    # Convert torch tensor to numpy for processing
    is_torch = isinstance(signal, torch.Tensor)
    if is_torch:
        device = signal.device
        dtype = signal.dtype
        signal_np = signal.cpu().numpy()
    else:
        signal_np = signal
        dtype = signal.dtype

    # Validate input shape
    if signal_np.shape[1] != 3:
        raise ValueError(
            f"VCG signal must have 3 channels (X, Y, Z), got {signal_np.shape[1]} channels. "
            f"Expected shape: [num_samples, 3]"
        )

    # Select transformation matrix based on method
    if method == 'dower':
        transform_matrix = DOWER_MATRIX
    elif method == 'kors':
        transform_matrix = INVERSE_KORS_MATRIX
    else:
        raise ValueError(
            f"Invalid method '{method}'. Must be 'dower' or 'kors'."
        )

    # Apply inverse transformation matrix
    # signal_np shape: [num_samples, 3]
    # transform_matrix shape: [8, 3]
    # Result shape: [num_samples, 8]
    ecg_signal = signal_np @ transform_matrix.T

    # Convert back to original type and device if needed
    if is_torch:
        ecg_signal = torch.from_numpy(ecg_signal).to(dtype=dtype, device=device)
    else:
        ecg_signal = ecg_signal.astype(dtype)

    return ecg_signal


def create_vcg_lead_names(original_lead_names: List[str]) -> List[str]:
    """
    Create VCG lead names (always returns ['X', 'Y', 'Z']).

    This is a convenience function to maintain consistency with lead naming.
    After VCG transformation, the output always has 3 orthogonal leads: X, Y, Z.

    Args:
        original_lead_names: Original ECG lead names (used for context only)

    Returns:
        List of VCG lead names: ['X', 'Y', 'Z']
    """
    return VCG_OUTPUT_LEADS.copy()


def validate_vcg_transformation(ecg_signal: np.ndarray,
                                vcg_signal: np.ndarray) -> dict:
    """
    Validate VCG transformation by checking output properties.

    Args:
        ecg_signal: Original ECG signal [num_samples, num_ecg_channels]
        vcg_signal: Transformed VCG signal [num_samples, 3]

    Returns:
        Dict with validation statistics:
        - n_samples: Number of time samples
        - n_leads: Number of output leads (should be 3)
        - amplitude_ranges: (min, max) for each VCG lead
        - kors_rank: Rank of Kors matrix (should be 3)
    """
    if vcg_signal.shape[1] != 3:
        raise ValueError(f"VCG signal should have 3 leads, got {vcg_signal.shape[1]}")

    stats = {
        'n_samples': vcg_signal.shape[0],
        'n_leads': vcg_signal.shape[1],
        'amplitude_ranges': tuple(
            (vcg_signal[:, i].min(), vcg_signal[:, i].max())
            for i in range(3)
        ),
        'kors_rank': np.linalg.matrix_rank(KORS_MATRIX)
    }

    return stats
