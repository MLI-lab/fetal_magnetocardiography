#!/usr/bin/env python3
"""
Simple WaveletBasis demonstration for sparse modeling.
"""

import torch
import numpy as np
from fmcg.fitting.basis import WaveletBasis, BSpline

# choose device consistently
device = torch.device("cpu")

def create_test_signal(n_time=256):
    """Create a test signal on the selected device."""
    t = torch.arange(n_time, dtype=torch.float32, device=device)
    signal = torch.sin(2 * np.pi * t / 32) + 0.1 * torch.randn(n_time, device=device)
    # Add a spike for sparsity testing
    signal[n_time // 2 : n_time // 2 + 5] += 2.0
    return signal

def compare_wavelets():
    """Compare wavelets for sparse modeling."""
    n_time = 128
    signal = create_test_signal(n_time).to(device)
    
    print("Wavelet Comparison:")
    print("-" * 30)
    
    wavelets = ['haar', 'db4', 'coif2']
    threshold = 0.1
    
    for wavelet in wavelets:
        wb = WaveletBasis(n_time=n_time, wavelet=wavelet, device=str(device))
        coeffs = wb.fit_coefficients(signal)
        
        # Apply sparsity
        sparse_coeffs = wb.soft_threshold(coeffs, threshold)
        sparse_signal = wb.forward(sparse_coeffs)
        
        # Calculate metrics
        n_nonzero = torch.sum(torch.abs(sparse_coeffs) > 1e-6).item()
        sparsity = n_nonzero / len(sparse_coeffs)
        mse = torch.mean((signal - sparse_signal)**2).item()
        
        print(f"{wavelet:6}: {sparsity:.1%} non-zero, MSE: {mse:.4f}")

def test_interface():
    """Test interface compatibility."""
    n_time = 64
    data = torch.randn(n_time, 2, device=device)
    
    print("\nInterface Test:")
    print("-" * 20)
    
    bases = [
        ('Wavelet', WaveletBasis(n_time=n_time, wavelet='db4', device=str(device))),
        ('B-spline', BSpline(n_knots=16, n_time=n_time, device=str(device)))
    ]
    
    for name, basis in bases:
        coeffs = basis.fit_coefficients(data)
        reconstructed = basis.forward(coeffs)
        error = torch.mean((data - reconstructed)**2).item()
        print(f"{name}: {coeffs.shape} -> MSE: {error:.6f}")

if __name__ == "__main__":
    print("WaveletBasis Demo")
    print("=" * 20)
    
    compare_wavelets()
    test_interface()
    
    print("\nDemo completed!")