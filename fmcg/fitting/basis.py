import numpy as np
import torch
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    pywt = None


class BSpline:
    """
    A class to represent B-spline basis functions. Creates a matrix of B-spline basis functions
    evaluated at specified time points.

    Attributes:
    -----------
    degree : int
        The degree of the spline.
    device : str
        The device to store the tensors ('cpu' or 'cuda').
    n_knots : int
        The number of knots.
    n_time : int
        The number of time points.
    basis : Tensor
        The B-spline basis functions evaluated at the time points.

    """

    def __init__(self, n_knots, n_time, degree=3, clamped=True, device="cpu"):
        self.degree = degree
        self.device = device
        self.n_knots = n_knots
        self.n_time = n_time

        knots = torch.linspace(0, n_time, n_knots, device=device)

        if clamped:
            knots = torch.cat(
                (
                    torch.zeros(degree, device=device),
                    knots,
                    torch.ones(degree, device=device),
                )
            )
        time = torch.arange(n_time, device=device)
        self.basis = self._bspline_basis(time, degree=degree, knots=knots).to(device)

    def forward(self, coefficients):
        """
        Perform a forward pass by multiplying the basis matrix with the given coefficients.

        Args:
            coefficients (torch.Tensor): A tensor of coefficients to be multiplied with the basis matrix.

        Returns:
            torch.Tensor: The result of the matrix multiplication between the basis matrix and the coefficients.
        """
        return torch.matmul(self.basis, coefficients)

    def fit_coefficients(self, data):
        """
        Fits the coefficients for the given data using the basis matrix.

        Parameters:
        data (torch.Tensor): The data to fit the coefficients to.

        Returns:
        torch.Tensor: The solution of the least squares problem, representing the fitted coefficients.
        """
        return torch.linalg.lstsq(self.basis, data).solution

    @staticmethod
    def _bspline_basis(t, degree, knots):
        """
        Evaluate B-spline basis functions at the given time points.

        Args:
            t: (Tensor, shape T) of time points.
            degree: The degree of the spline.
            knots: (Tensor|Array, shape n) of knot positions, where n is the number of knots.

        Returns:
            B: (Tensor, shape T x n) of basis functions evaluated at the time points.
        """
        T = t.shape[0]
        num_knots = knots.shape[0]
        n = num_knots - degree - 1  # number of basis functions

        # Initialize basis functions for degree 0
        # B[i](t) = 1 if knots[i] <= t < knots[i+1], else 0.
        B = []
        for i in range(n):
            # For the last knot interval, include the right endpoint
            if i == n - 1:
                cond = (t >= knots[i]) & (t <= knots[i + 1])
            else:
                cond = (t >= knots[i]) & (t < knots[i + 1])
            B.append(cond.float())
        # Stack into a [T, n] tensor.
        B = torch.stack(B, dim=1)

        # Recursively compute higher-degree basis functions
        for d in range(1, degree + 1):
            B_new = []
            for i in range(n):
                # Compute the first term.
                denom1 = knots[i + d] - knots[i]
                if denom1 == 0:
                    term1 = torch.zeros_like(t)
                else:
                    term1 = ((t - knots[i]) / denom1) * B[:, i]

                # Compute the second term.
                if i + 1 < n:
                    denom2 = knots[i + d + 1] - knots[i + 1]
                    if denom2 == 0:
                        term2 = torch.zeros_like(t)
                    else:
                        term2 = ((knots[i + d + 1] - t) / denom2) * B[:, i + 1]
                else:
                    term2 = torch.zeros_like(t)
                B_new.append(term1 + term2)
            B = torch.stack(B_new, dim=1)
        return B



class IdentityBasis:
    """
    A class used to represent an Identity Basis, i.e., constants over time.
    This is a simple basis function that returns the coefficients as they are.
    It is useful for cases where no transformation is needed.
    The coefficients are simply expanded over the number of time steps.

    Attributes
    ----------
    n_time : int
        The number of time steps.
    device : str, optional
        The device to be used (default is "cpu").

    """

    def __init__(self, n_time, device="cpu"):
        self.device = device
        self.n_time = n_time

    def forward(self, coefficients):
        return coefficients.expand(self.n_time, -1)

    def fit_coefficients(self, data):
        return data[0]


class TanhBasis:
    """
    A class that implements a basis transformation using the hyperbolic tangent (tanh) function.

    Attributes:
        device (str): The device on which tensors will be allocated (e.g., "cpu" or "cuda").
        n_time (int, optional): Number of time steps to tile the output, if exhibiting not time dimension.


    """

    def __init__(self, bnds, n_time=None, device="cpu"):
        self.device = device
        self.n_time = n_time

        bnds_ = torch.tensor(bnds.copy(), device=device, dtype=torch.float32).reshape(
            bnds.shape[0] * bnds.shape[1], 2
        )
        self.b1 = (bnds_[:, 1] - bnds_[:, 0]) / 2
        self.b0 = (bnds_[:, 0] + bnds_[:, 1]) / 2

    def forward(self, coefficients):
        x = self.b1 * torch.tanh(coefficients) + self.b0
        if self.n_time is not None:
            return x.tile(self.n_time, 1, 1)
        return x

    def fit_coefficients(self, data):
        if data.ndim == 2 and self.n_time is not None:
            data = data[0]
        return torch.tanh(data - self.b0) / self.b1


class SigmoidBasis:
    """
    A class that implements a sigmoid basis transformation.

    Attributes:
        device (str): The device on which the computations will be performed (e.g., "cpu" or "cuda").
        n_time (int, optional): Number of time steps to tile the output, if exhibiting not time dimension.


    """

    def __init__(self, bnds, n_time=None, device="cpu"):
        self.device = device
        self.n_time = n_time

        if isinstance(bnds, list):
            bnds = np.array(bnds)
        bnds_ = torch.tensor(bnds.copy(), device=device, dtype=torch.float32).reshape(
            bnds.shape[0] * bnds.shape[1], 2
        )
        self.b1 = bnds_[:, 1] - bnds_[:, 0]
        self.b0 = bnds_[:, 0]

    def forward(self, coefficients):
        x = self.b1 / (1 + torch.exp(-coefficients)) + self.b0
        if self.n_time is not None:
            return x.tile(self.n_time, 1, 1)
        return x

    def fit_coefficients(self, data):
        if data.ndim == 2 and self.n_time is not None:
            data = data[0]
        return -torch.log((self.b1 / (data - self.b0)) - 1)


class ExponentialBasis:
    """
    A class that implements an exponential basis transformation.

    Attributes:
        basis (str): The basis type, either "e" for natural exponential or "10" for base 10.
        n_time (int, optional): Number of time steps to tile the output, if exhibiting not time dimension.
        device (str): The device on which the computations will be performed (e.g., "cpu" or "cuda").
    """

    def __init__(self, basis="e", n_time=None, device="cpu"):
        self.device = device
        self.n_time = n_time

        if basis == "e":
            self.a = torch.exp(torch.tensor(1.0, device=device, dtype=torch.float32))
        elif basis == "10":
            self.a = torch.tensor(10.0, device=device, dtype=torch.float32)
        else:
            raise ValueError(
                "Unsupported basis type. Use 'e' for natural exponential or '10' for base 10."
            )

        self.exponent = 1.0

    def forward(self, coefficients):
        exponent = coefficients[0]
        return coefficients[1:] * (self.a.pow(exponent))

    def fit_coefficients(self, data):
        exponent = torch.log(
            data.abs().mean(dim=0, keepdim=True) + torch.finfo(torch.float32).eps
        ) / torch.log(
            self.a
        )  # .to(dtype=torch.float32)

        return torch.concatenate([exponent, data / (self.a.pow(exponent))], axis=0)


class WaveletBasis:
    """
    Memory-efficient wavelet basis with gradient support. Uses direct wavelet 
    reconstruction instead of storing massive basis matrices.
    
    Args:
        n_time (int): Number of time points
        wavelet (str): Wavelet type ('haar', 'db4', 'db8', 'coif2', etc.)
        mode (str): Boundary mode ('symmetric', 'periodization')  
        device (str): Device ('cpu' or 'cuda')
        level (int): Decomposition level (None for max level)
    """
    
    def __init__(self, n_time, wavelet='db4', mode='symmetric', device="cuda", level=None, **kwargs):
        if not PYWT_AVAILABLE:
            raise ImportError("PyWavelets required: pip install PyWavelets")
        
        self.wavelet = wavelet
        self.mode = mode
        self.device = device
        self.n_time = n_time
        self.level = level
        
        try:
            self.wavelet_obj = pywt.Wavelet(wavelet)
        except ValueError:
            raise ValueError(f"Unknown wavelet '{wavelet}'")
        
        # Calculate coefficient structure once
        test_signal = np.zeros(n_time)
        coeffs = pywt.wavedec(test_signal, wavelet, mode=mode, level=level)
        self.coeff_lengths = [len(c) for c in coeffs]
        self.total_coeffs = sum(self.coeff_lengths)
        self.coeff_slices = [(sum(self.coeff_lengths[:i]), sum(self.coeff_lengths[:i+1])) 
                            for i in range(len(self.coeff_lengths))]
        
        print(f"WaveletBasis: Signal {n_time}→{self.total_coeffs} coeffs, "
              f"{self.total_coeffs * 4 / 1024**2:.1f}MB memory")
        self.basis = None  # Never create basis matrix

    def forward(self, coefficients):
        """Forward pass: reconstruct signal from coefficients with gradient support."""
        return WaveletBasis._ApplyReconstruction.apply(
            coefficients, self.wavelet, self.mode, self.level, 
            self.coeff_slices, self.n_time, self.device
        )
    
    def __call__(self, coefficients):
        """Make the instance callable."""
        return self.forward(coefficients)

    class _ApplyReconstruction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, coefficients, wavelet, mode, level, coeff_slices, n_time, device):
            """Forward pass: reconstruct signal from coefficients."""
            # Save for backward pass
            ctx.wavelet, ctx.mode, ctx.level = wavelet, mode, level
            ctx.coeff_slices, ctx.n_time, ctx.device = coeff_slices, n_time, device
            ctx.save_for_backward(coefficients)
            
            return WaveletBasis._reconstruct(coefficients, wavelet, mode, coeff_slices, n_time, device)
        
        @staticmethod
        def backward(ctx, grad_output):
            """Backward pass: compute gradients w.r.t. coefficients."""
            grad_coeffs = WaveletBasis._decompose(grad_output, ctx.wavelet, ctx.mode, ctx.level,
                                                 ctx.coeff_slices, ctx.device)
            return grad_coeffs, None, None, None, None, None, None
    
    @staticmethod
    def _reconstruct(coefficients, wavelet, mode, coeff_slices, n_time, device):
        """Core reconstruction logic."""
        if coefficients.dim() == 1:
            return WaveletBasis._reconstruct_single(coefficients, wavelet, mode, coeff_slices, n_time, device)
        elif coefficients.dim() == 2:
            results = [WaveletBasis._reconstruct_single(coefficients[:, ch], wavelet, mode, 
                                                       coeff_slices, n_time, device)
                      for ch in range(coefficients.shape[1])]
            return torch.stack(results, dim=1)
        else:
            raise ValueError(f"Coefficients must be 1D or 2D, got shape {coefficients.shape}")
    
    @staticmethod
    def _reconstruct_single(coeffs_flat, wavelet, mode, coeff_slices, n_time, device):
        """Reconstruct single signal from coefficients."""
        coeffs_np = coeffs_flat.detach().cpu().numpy()
        expected_length = coeff_slices[-1][1]
        
        # Ensure correct coefficient length
        if len(coeffs_np) != expected_length:
            if len(coeffs_np) > expected_length:
                coeffs_np = coeffs_np[:expected_length]
            else:
                padded = np.zeros(expected_length)
                padded[:len(coeffs_np)] = coeffs_np
                coeffs_np = padded
        
        # Reshape to PyWavelets format and reconstruct
        coeffs = [coeffs_np[start:end] for start, end in coeff_slices]
        reconstructed = pywt.waverec(coeffs, wavelet, mode=mode)
        
        # Ensure correct output length
        if len(reconstructed) >= n_time:
            result = reconstructed[:n_time]
        else:
            result = np.zeros(n_time)
            result[:len(reconstructed)] = reconstructed
            
        return torch.tensor(result, dtype=torch.float32, device=device)
    
    @staticmethod
    def _decompose(signal, wavelet, mode, level, coeff_slices, device):
        """Decompose signal for gradient computation.""" 
        if signal.dim() == 1:
            return WaveletBasis._decompose_single(signal, wavelet, mode, level, coeff_slices, device)
        elif signal.dim() == 2:
            grad_coeffs_list = [WaveletBasis._decompose_single(signal[:, ch], wavelet, mode, level,
                                                              coeff_slices, device)
                               for ch in range(signal.shape[1])]
            return torch.stack(grad_coeffs_list, dim=1)
        else:
            raise ValueError(f"Signal must be 1D or 2D, got shape {signal.shape}")
    
    @staticmethod 
    def _decompose_single(signal, wavelet, mode, level, coeff_slices, device):
        """Decompose single signal into coefficients."""
        signal_np = signal.detach().cpu().numpy()
        coeffs = pywt.wavedec(signal_np, wavelet, mode=mode, level=level)
        coeffs_flat = np.concatenate([c.flatten() for c in coeffs])
        
        # Ensure correct gradient length
        expected_length = coeff_slices[-1][1]
        if len(coeffs_flat) != expected_length:
            if len(coeffs_flat) > expected_length:
                coeffs_flat = coeffs_flat[:expected_length]
            else:
                padded = np.zeros(expected_length)
                padded[:len(coeffs_flat)] = coeffs_flat
                coeffs_flat = padded
                
        return torch.tensor(coeffs_flat, dtype=torch.float32, device=device)
    

    
    def fit_coefficients(self, data):
        """Compute wavelet coefficients from data."""
        data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
        
        if data_np.ndim == 1:
            coeffs = pywt.wavedec(data_np, self.wavelet, mode=self.mode, level=self.level)
            coeffs_flat = np.concatenate([c.flatten() for c in coeffs])
        elif data_np.ndim == 2:
            all_coeffs = []
            for ch in range(data_np.shape[1]):
                coeffs = pywt.wavedec(data_np[:, ch], self.wavelet, mode=self.mode, level=self.level)
                all_coeffs.append(np.concatenate([c.flatten() for c in coeffs]))
            coeffs_flat = np.column_stack(all_coeffs)
        else:
            raise ValueError(f"Data must be 1D or 2D, got shape {data_np.shape}")
            
        return torch.tensor(coeffs_flat, dtype=torch.float32, device=self.device)

    def soft_threshold(self, coefficients, threshold):
        """Apply soft thresholding for sparse modeling."""
        return torch.sign(coefficients) * torch.clamp(torch.abs(coefficients) - threshold, min=0)
    
    def estimate_memory_usage(self):
        """Estimate actual memory usage (coefficients only)."""
        return {
            'coefficients': self.total_coeffs,
            'mb': self.total_coeffs * 4 / (1024**2),
            'gb': self.total_coeffs * 4 / (1024**3),
            'note': 'Memory-efficient: coefficients only, no basis matrix'
        }
    
    @classmethod  
    def create_memory_optimal(cls, *args, **kwargs):
        """Create optimized instance (all instances are memory-optimal now)."""
        return cls(*args, **kwargs)


