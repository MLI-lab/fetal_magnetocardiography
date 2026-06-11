import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import warnings

class ConstrainedICA:
    """
    Constrained Independent Component Analysis implementation.
    
    Supports various constraints:
    - Non-negativity constraints
    - Sparsity constraints (L1 regularization)
    - Reference signal constraints
    - Smoothness constraints (L2 regularization on differences)
    """
    
    def __init__(self, n_components=None, constraint_type='none', 
                 lambda_reg=0.1, reference_signals=None, max_iter=1000, 
                 tol=1e-6, random_state=None):
        """
        Initialize Constrained ICA.
        
        Parameters:
        -----------
        n_components : int, optional
            Number of components to extract
        constraint_type : str
            Type of constraint: 'none', 'non_negative', 'sparse', 'reference', 'smooth'
        lambda_reg : float
            Regularization parameter for constraints
        reference_signals : array-like, optional
            Reference signals for reference constraint
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        random_state : int, optional
            Random seed for reproducibility
        """
        self.n_components = n_components
        self.constraint_type = constraint_type
        self.lambda_reg = lambda_reg
        self.reference_signals = reference_signals
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # Initialize attributes
        self.mixing_matrix_ = None
        self.unmixing_matrix_ = None
        self.sources_ = None
        self.mean_ = None
        self.whitening_matrix_ = None
        self.n_iter_ = 0
        
    def _center_and_whiten(self, X):
        """Center and whiten the data."""
        # Center the data
        self.mean_ = np.mean(X, axis=1, keepdims=True)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        cov = np.cov(X_centered)
        
        # Add regularization to prevent numerical issues
        cov += 1e-12 * np.eye(cov.shape[0])
        
        # Eigenvalue decomposition for whitening
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Keep only eigenvalues above threshold and ensure they're positive
        threshold = 1e-12
        non_zero_idx = eigenvals > threshold
        eigenvals = eigenvals[non_zero_idx]
        eigenvecs = eigenvecs[:, non_zero_idx]
        
        # Ensure we have at least one eigenvalue
        if len(eigenvals) == 0:
            eigenvals = np.array([1e-10])
            eigenvecs = np.eye(cov.shape[0])[:, :1]
        
        # Whitening matrix with numerical safeguard
        eigenvals_safe = np.maximum(eigenvals, 1e-12)
        self.whitening_matrix_ = np.dot(eigenvecs, np.diag(1.0 / np.sqrt(eigenvals_safe)))
        
        # Whiten the data
        X_whitened = np.dot(self.whitening_matrix_.T, X_centered)
        
        # Check for inf/nan values and replace with zeros if necessary
        X_whitened = np.nan_to_num(X_whitened, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X_whitened
    
    def _g_function(self, u, deriv_order=0):
        """Nonlinear function for ICA (tanh)."""
        return np.log(np.cosh(u))
        if deriv_order == 0:
            return np.tanh(u)
        elif deriv_order == 1:
            return 1 - np.tanh(u)**2
        else:
            raise ValueError("Only derivatives up to order 1 are supported")
    
    def _apply_constraints(self, W, sources):
        """Apply constraints to the unmixing matrix and sources."""
        if self.constraint_type == 'non_negative':
            # Apply non-negativity constraint using rectification
            sources = np.maximum(sources, 0)
            
        elif self.constraint_type == 'sparse':
            # Apply sparsity constraint using soft thresholding
            threshold = self.lambda_reg
            sources = np.sign(sources) * np.maximum(np.abs(sources) - threshold, 0)
            
        elif self.constraint_type == 'reference' and self.reference_signals is not None:
            # Apply reference constraint by minimizing distance to reference signals
            ref_signals = np.array(self.reference_signals)
            if ref_signals.shape[0] == sources.shape[0]:
                # Compute correlation and reorder sources to match references
                correlations = np.abs(np.corrcoef(sources, ref_signals)[:sources.shape[0], sources.shape[0]:])
                # Hungarian algorithm could be used here for optimal assignment
                # For simplicity, we use greedy assignment
                assignment = []
                used_refs = []
                for i in range(sources.shape[0]):
                    best_ref = -1
                    best_corr = -1
                    for j in range(ref_signals.shape[0]):
                        if j not in used_refs and correlations[i, j] > best_corr:
                            best_corr = correlations[i, j]
                            best_ref = j
                    if best_ref != -1:
                        assignment.append(best_ref)
                        used_refs.append(best_ref)
                    else:
                        assignment.append(0)  # fallback
                
                # Reorder sources to match references
                new_order = np.argsort(assignment)
                sources = sources[new_order]
                W = W[new_order]
        
        return W, sources
    
    def _objective_function(self, w, X_white, component_idx):
        """Objective function for constrained ICA optimization."""
        w = w.reshape(-1, 1)
        w_norm = np.linalg.norm(w)
        
        # Prevent division by zero
        if w_norm < 1e-12:
            return 1e6  # Return large penalty for zero vector
            
        w = w / w_norm  # Normalize
        
        # Extract source
        source = np.dot(w.T, X_white).flatten()
        
        # Replace inf/nan values
        source = np.nan_to_num(source, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Standard ICA objective (negative entropy approximation)
        g_vals = self._g_function(source)
        g_vals = np.nan_to_num(g_vals, nan=0.0, posinf=1.0, neginf=-1.0)
        objective = -np.mean(g_vals)
        
        # Add constraint terms
        if self.constraint_type == 'sparse':
            objective += self.lambda_reg * np.mean(np.abs(source))
        elif self.constraint_type == 'smooth':
            # Smoothness constraint (penalize large differences)
            diff = np.diff(source, n=1)
            print(f"Component {objective}: diff={np.mean(diff**2)}")
            objective += self.lambda_reg * np.mean(np.abs(diff))
        elif self.constraint_type == 'reference' and self.reference_signals is not None:
            # Reference constraint
            if component_idx < len(self.reference_signals):
                ref_signal = self.reference_signals[component_idx]
                if len(ref_signal) == len(source):
                    # Ensure signals have non-zero variance
                    source_var = np.var(source)
                    ref_var = np.var(ref_signal)
                    if source_var > 1e-12 and ref_var > 1e-12:
                        try:
                            correlation = np.corrcoef(source, ref_signal)[0, 1]
                            correlation = np.nan_to_num(correlation, nan=0.0)
                            objective -= self.lambda_reg * np.abs(correlation)
                        except:
                            pass  # Skip correlation if it fails
        
        # Ensure objective is finite
        objective = np.nan_to_num(objective, nan=1e6, posinf=1e6, neginf=-1e6)
        
        return objective
    
    def fit(self, X):
        """
        Fit the Constrained ICA model.
        
        Parameters:
        -----------
        X : array-like, shape (n_features, n_samples)
            Training data
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X should be a 2D array")
        
        n_features, n_samples = X.shape
        
        if self.n_components is None:
            self.n_components = n_features
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Center and whiten the data
        X_white = self._center_and_whiten(X)
        
        # Initialize unmixing matrix
        W = np.random.randn(self.n_components, X_white.shape[0])
        
        # Gram-Schmidt orthogonalization
        for i in range(self.n_components):
            for j in range(i):
                W[i] -= np.dot(W[i], W[j]) * W[j]
            W[i] /= np.linalg.norm(W[i])
        
        # Iterative optimization
        for iteration in range(self.max_iter):
            W_old = W.copy()
            
            # Update each component
            for i in range(self.n_components):
                # Optimize single component
                result = minimize(
                    self._objective_function,
                    W[i],
                    args=(X_white, i),
                    method='BFGS',
                    options={'maxiter': 500}
                )
                
                if result.success:
                    W[i] = result.x
                    w_norm = np.linalg.norm(W[i])
                    if w_norm > 1e-12:
                        W[i] /= w_norm
                    else:
                        # Reinitialize if norm is too small
                        W[i] = np.random.randn(W[i].shape[0])
                        W[i] /= np.linalg.norm(W[i])
            
            # Gram-Schmidt orthogonalization with numerical safeguards
            for i in range(self.n_components):
                for j in range(i):
                    dot_product = np.dot(W[i], W[j])
                    W[i] -= dot_product * W[j]
                
                w_norm = np.linalg.norm(W[i])
                if w_norm > 1e-12:
                    W[i] /= w_norm
                else:
                    # Reinitialize if norm is too small
                    W[i] = np.random.randn(W[i].shape[0])
                    W[i] /= np.linalg.norm(W[i])
            
            # Check convergence with numerical safeguards
            try:
                conv_matrix = np.dot(W, W_old.T)
                conv_matrix = np.nan_to_num(conv_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
                diag_values = np.diag(conv_matrix)
                conv_criterion = np.max(np.abs(np.abs(diag_values) - 1))
                if conv_criterion < self.tol:
                    break
            except:
                # If convergence check fails, continue
                pass
        
        self.n_iter_ = iteration + 1
        self.unmixing_matrix_ = W
        
        # Extract sources
        sources = np.dot(W, X_white)
        
        # Apply constraints
        self.unmixing_matrix_, sources = self._apply_constraints(W, sources)
        
        # Store results
        self.sources_ = sources
        self.mixing_matrix_ = np.linalg.pinv(self.unmixing_matrix_)
        
        return self
    
    def transform(self, X):
        """
        Transform data using the fitted model.
        
        Parameters:
        -----------
        X : array-like, shape (n_features, n_samples)
            Data to transform
            
        Returns:
        --------
        sources : array-like, shape (n_components, n_samples)
            Extracted sources
        """
        if self.unmixing_matrix_ is None:
            raise ValueError("Model must be fitted before transform")
        
        X = np.asarray(X)
        X_centered = X - self.mean_
        X_white = np.dot(self.whitening_matrix_.T, X_centered)
        sources = np.dot(self.unmixing_matrix_, X_white)
        
        # Apply constraints
        #_, sources = self._apply_constraints(self.unmixing_matrix_, sources)
        
        return sources
    
    def fit_transform(self, X):
        """Fit the model and transform the data."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, sources):
        """
        Transform sources back to original space.
        
        Parameters:
        -----------
        sources : array-like, shape (n_components, n_samples)
            Sources to transform back
            
        Returns:
        --------
        X_reconstructed : array-like, shape (n_features, n_samples)
            Reconstructed data
        """
        if self.mixing_matrix_ is None:
            raise ValueError("Model must be fitted before inverse_transform")
        
        sources = np.asarray(sources)
        X_white_reconstructed = np.dot(self.mixing_matrix_, sources)
        X_reconstructed = np.dot(self.whitening_matrix_, X_white_reconstructed) + self.mean_
        
        return X_reconstructed


def performance_metrics(original_sources, estimated_sources):
    """Calculate performance metrics"""
    # Amari error (permutation-invariant measure)
    def amari_error(W, A):
        P = np.dot(W, A)
        n = P.shape[0]
        
        # Row-wise normalization with safeguard
        P_abs = np.abs(P)
        row_max = np.max(P_abs, axis=1, keepdims=True)
        row_max = np.maximum(row_max, 1e-12)  # Prevent division by zero
        P_row = P_abs / row_max
        
        # Column-wise normalization with safeguard
        col_max = np.max(P_abs, axis=0, keepdims=True)
        col_max = np.maximum(col_max, 1e-12)  # Prevent division by zero
        P_col = P_abs / col_max
        
        amari = (np.sum(P_row) - n) + (np.sum(P_col) - n)
        return amari / (2 * n * (n - 1))
    
    # Signal-to-Interference Ratio (SIR)
    def sir_score(original, estimated):
        # Find best permutation and scaling
        n_sources = original.shape[0]
        best_sir = -np.inf
        
        from itertools import permutations
        for perm in permutations(range(n_sources)):
            current_sir = 0
            for i in range(n_sources):
                # Scale estimated to match original with safeguards
                denominator = np.dot(estimated[perm[i]], estimated[perm[i]])
                if denominator < 1e-12:
                    continue  # Skip this permutation if denominator is too small
                    
                scale = np.dot(original[i], estimated[perm[i]]) / denominator
                scale = np.nan_to_num(scale, nan=0.0, posinf=1.0, neginf=-1.0)
                scaled_est = scale * estimated[perm[i]]
                
                # Calculate SIR for this source with safeguards
                signal_power = np.var(original[i])
                noise_power = np.var(original[i] - scaled_est)
                
                # Ensure positive powers
                signal_power = max(signal_power, 1e-12)
                noise_power = max(noise_power, 1e-12)
                
                sir = 10 * np.log10(signal_power / noise_power)
                sir = np.nan_to_num(sir, nan=-100.0, posinf=100.0, neginf=-100.0)
                current_sir += sir
                
            if np.isfinite(current_sir):
                best_sir = max(best_sir, current_sir / n_sources)
        
        # Return reasonable default if all permutations failed
        if not np.isfinite(best_sir):
            best_sir = -100.0
            
        return best_sir
    
    # Safe correlation calculation
    def safe_correlation(sig1, sig2):
        try:
            # Check if signals have sufficient variance
            if np.var(sig1) < 1e-12 or np.var(sig2) < 1e-12:
                return 0.0
            corr_matrix = np.corrcoef(sig1, sig2)
            corr = corr_matrix[0, 1]
            return np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        except:
            return 0.0
    
    try:
        sir = sir_score(original_sources, estimated_sources)
        
        # Calculate average maximum correlation
        correlations = []
        for i in range(len(original_sources)):
            max_corr = 0.0
            for j in range(len(estimated_sources)):
                corr = abs(safe_correlation(original_sources[i], estimated_sources[j]))
                max_corr = max(max_corr, corr)
            correlations.append(max_corr)
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        return {
            'sir': sir,
            'correlation': avg_correlation
        }
    except Exception as e:
        # Return default values if calculation fails
        return {
            'sir': -100.0,
            'correlation': 0.0
        }


def comprehensive_visual_comparison(algorithms=None):
    """Comprehensive comparison of FastICA vs Constrained ICA with visualizations"""
    print("=" * 60)
    print("COMPREHENSIVE ICA COMPARISON WITH VISUALIZATION")
    print("=" * 60)

    # Test scenarios
    scenarios = {
        'Standard Sources': {
            'description': 'Well-separated non-Gaussian sources',
            'sources_func': lambda t: np.array([
                np.sin(2 * t) + 0.1 * np.sin(10 * t),  # Sinusoidal
                np.sign(np.sin(3 * t)),  # Square wave
                np.random.laplace(size=len(t))  # Laplacian
            ])
        },
        'Non-negative Sources': {
            'description': 'Sources constrained to be non-negative',
            'sources_func': lambda t: np.array([
                np.maximum(np.sin(2 * t), 0),  # Rectified sinusoid
                np.maximum(np.random.laplace(size=len(t)), 0),  # Rectified Laplacian
                np.abs(np.random.randn(len(t)))  # Absolute Gaussian
            ])
        },
        'Sparse Sources': {
            'description': 'Sparse sources with many zeros',
            'sources_func': lambda t: np.array([
                np.random.laplace(size=len(t)) * (np.random.rand(len(t)) > 0.7),  # Sparse Laplacian
                np.sin(2 * t) * (np.random.rand(len(t)) > 0.8),  # Sparse sinusoid
                (np.random.randn(len(t)) > 1.5).astype(float)  # Very sparse
            ])
        }
    }

    # Mixing matrix
    A = np.array([[1, 1, 1],
                  [0.5, 2, 1.0], 
                  [1.5, 1.0, 2.0]])

    results = {}

    for scenario_idx, (scenario_name, scenario) in enumerate(scenarios.items()):
        print(f"\n{scenario_name}: {scenario['description']}")
        print("-" * 50)

        # Generate data
        np.random.seed(42)
        t = np.linspace(0, 8, 1000)
        sources = scenario['sources_func'](t)
        X = np.dot(A, sources) + 0.05 * np.random.randn(*np.dot(A, sources).shape)

        # Test algorithms
        if algorithms is None:
            algorithms = {
                'FastICA': FastICA(n_components=3, random_state=42, max_iter=1000),
                'Constrained ICA (none)': ConstrainedICA(n_components=3, constraint_type='none'),
                'Constrained ICA (non-negative)': ConstrainedICA(n_components=3, constraint_type='non_negative'),
                'Constrained ICA (sparse)': ConstrainedICA(n_components=3, constraint_type='sparse', lambda_reg=0.1),
                'Constrained ICA (smooth)': ConstrainedICA(n_components=3, constraint_type='smooth', lambda_reg=0.1)
            }

        scenario_results = {}
        estimated_sources_dict = {}

        for alg_name, algorithm in algorithms.items():
            try:
                # Time the algorithm
                start_time = time.time()

                if 'FastICA' in alg_name:
                    estimated = algorithm.fit_transform(X.T.copy()).T
                else:
                    estimated = algorithm.fit_transform(X.copy())

                execution_time = time.time() - start_time

                # Check for inf/nan in estimated sources
                estimated = np.nan_to_num(estimated, nan=0.0, posinf=1e6, neginf=-1e6)

                # Calculate performance metrics
                metrics = performance_metrics(sources.copy(), estimated)

                # Additional metrics with safeguards
                try:
                    kurtosis_values = [abs(kurtosis(s)) for s in estimated]
                    kurtosis_values = [np.nan_to_num(k, nan=0.0, posinf=10.0, neginf=0.0) for k in kurtosis_values]
                    source_kurtosis = np.mean(kurtosis_values)
                except:
                    source_kurtosis = 0.0

                scenario_results[alg_name] = {
                    'execution_time': execution_time,
                    'sir': metrics['sir'],
                    'correlation': metrics['correlation'],
                    'kurtosis': source_kurtosis
                }
                estimated_sources_dict[alg_name] = estimated

                print(f"{alg_name:25} | Time: {execution_time:.3f}s | SIR: {metrics['sir']:6.2f}dB | Corr: {metrics['correlation']:.3f}")

            except Exception as e:
                print(f"{alg_name:25} | FAILED: {str(e)[:30]}...")
                scenario_results[alg_name] = {
                    'execution_time': 0.0,
                    'sir': -100.0,
                    'correlation': 0.0,
                    'kurtosis': 0.0
                }
                estimated_sources_dict[alg_name] = None

        results[scenario_name] = scenario_results

        # Visualization for this scenario
        fig, axes = plt.subplots(len(algorithms)+2, 3, figsize=(15, 16))
        fig.suptitle(f"{scenario_name}: {scenario['description']}", fontsize=16)

        # Plot original sources
        for col_idx in range(3):
            axes[0, col_idx].plot(t, sources[col_idx])
            axes[0, col_idx].set_title(f'Original Source {col_idx+1}')
            axes[0, col_idx].grid(True)

        # Plot mixed signals
        for col_idx in range(3):
            axes[1, col_idx].plot(t, X[col_idx])
            axes[1, col_idx].set_title(f'Mixed Signal {col_idx+1}')
            axes[1, col_idx].grid(True)

        # Plot estimated sources for each algorithm in the algorithms dict
        alg_plot_list = list(algorithms.keys())
        for row_idx, alg_key in enumerate(alg_plot_list, start=2):
            est = estimated_sources_dict.get(alg_key)
            for col_idx in range(3):
                if est is not None:
                    axes[row_idx, col_idx].plot(t, est[col_idx])
                    axes[row_idx, col_idx].set_title(f'{alg_key} {col_idx+1}')
                    axes[row_idx, col_idx].grid(True)
                else:
                    axes[row_idx, col_idx].set_title(f'{alg_key} FAILED')
                    axes[row_idx, col_idx].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()

    return results
