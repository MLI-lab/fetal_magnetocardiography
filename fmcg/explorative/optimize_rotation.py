import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def create_rotation_matrix(angles: np.ndarray) -> np.ndarray:
    """
    Create a 3D rotation matrix from Euler angles (X->Y->Z order).
    
    Args:
        angles: Array of shape (3,) containing [rx, ry, rz] in degrees
        
    Returns:
        3x3 rotation matrix
    """
    rx, ry, rz = np.radians(angles)
    
    # Pre-compute trigonometric values
    cos_rx, sin_rx = np.cos(rx), np.sin(rx)
    cos_ry, sin_ry = np.cos(ry), np.sin(ry)
    cos_rz, sin_rz = np.cos(rz), np.sin(rz)
    
    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, cos_rx, -sin_rx],
                   [0, sin_rx, cos_rx]])
    
    Ry = np.array([[cos_ry, 0, sin_ry],
                   [0, 1, 0],
                   [-sin_ry, 0, cos_ry]])
    
    Rz = np.array([[cos_rz, -sin_rz, 0],
                   [sin_rz, cos_rz, 0],
                   [0, 0, 1]])
    
    # Combined rotation: Rz @ Ry @ Rx
    return Rz @ Ry @ Rx

def apply_rotation(data: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    Apply rotation to 3D time series data.
    
    Args:
        data: Array of shape (N, 3) where N is number of time points
        angles: Array of shape (3,) containing [rx, ry, rz] in degrees
        
    Returns:
        Rotated data of same shape
    """
    R = create_rotation_matrix(angles)
    return data @ R.T

def obj_func(angles: np.ndarray, data: np.ndarray) -> float:
    """
    Objective function to maximize the mean of all signal components.
    
    Args:
        angles: Array of shape (3,) containing [rx, ry, rz] in degrees
        data: Array of shape (N, 3) where N is number of time points
        
    Returns:
        Negative mean (for minimization)
    """
    rotated_data = apply_rotation(data, angles)
    # Return negative mean for minimization
    return np.mean(rotated_data,axis=0).mean()
    return -np.mean(rotated_data, axis=0).mean()

def optimize_rotation_parameters(
    data: np.ndarray,
    method: str = "L-BFGS-B",
    n_restarts: int = 5,
    verbose: bool = True
) -> Dict:
    """
    Optimize rotation parameters to maximize signal characteristics.
    
    Args:
        data: Array of shape (N, 3) where N is number of time points
        method: Optimization method
        n_restarts: Number of random restarts
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing optimization results
    """
    # Bounds for angles in degrees
    bounds = [(-180, 180), (-180, 180), (-180, 180)]
    
    best_result = None
    best_value = np.inf
    
    for i in range(n_restarts):
        # Random initial guess
        x0 = np.random.uniform(-180, 180, 3)
        
        try:
            result = minimize(
                obj_func,
                x0,
                args=(data,),
                method=method,
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.fun < best_value:
                best_value = result.fun
                best_result = result
                
        except Exception as e:
            if verbose:
                print(f"Restart {i+1} failed: {e}")
            continue
    
    if best_result is None:
        raise RuntimeError("All optimization attempts failed")
    
    # Calculate final metrics
    optimal_angles = best_result.x
    rotated_data = apply_rotation(data, optimal_angles)
    
    results = {
        'optimal_angles': optimal_angles,
        'optimization_result': best_result,
        'rotated_data': rotated_data,
        'original_mean': np.mean(data),
        'rotated_mean': np.mean(rotated_data),
        'original_positive_fraction': np.mean(data > 0),
        'rotated_positive_fraction': np.mean(rotated_data > 0),
        'improvement_mean': np.mean(rotated_data) - np.mean(data),
        'improvement_positive': np.mean(rotated_data > 0) - np.mean(data > 0)
    }
    
    if verbose:
        print(f"Optimal angles: [{optimal_angles[0]:.2f}°, {optimal_angles[1]:.2f}°, {optimal_angles[2]:.2f}°]")
        print(f"Original mean: {results['original_mean']:.6f}")
        print(f"Rotated mean: {results['rotated_mean']:.6f}")
        print(f"Mean improvement: {results['improvement_mean']:.6f}")
        print(f"Original positive fraction: {results['original_positive_fraction']:.3f}")
        print(f"Rotated positive fraction: {results['rotated_positive_fraction']:.3f}")
        print(f"Positive fraction improvement: {results['improvement_positive']:.3f}")
    
    return results

def plot_optimization_comparison(
    original_data: np.ndarray,
    rotated_data: np.ndarray,
    time_axis: Optional[np.ndarray] = None,
    title_suffix: str = ""
) -> None:
    """
    Plot comparison between original and optimized rotated data.
    
    Args:
        original_data: Original data array of shape (N, 3)
        rotated_data: Rotated data array of shape (N, 3)
        time_axis: Time axis for plotting (optional)
        title_suffix: Additional text for plot title
    """
    if time_axis is None:
        time_axis = np.arange(len(original_data))
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Original data
    axes[0].plot(time_axis, original_data[:, 0], 'r-', label='X', alpha=0.8)
    axes[0].plot(time_axis, original_data[:, 1], 'g-', label='Y', alpha=0.8)
    axes[0].plot(time_axis, original_data[:, 2], 'b-', label='Z', alpha=0.8)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0].set_title(f'Original Data{title_suffix}')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Rotated data
    axes[1].plot(time_axis, rotated_data[:, 0], 'r-', label='X', alpha=0.8)
    axes[1].plot(time_axis, rotated_data[:, 1], 'g-', label='Y', alpha=0.8)
    axes[1].plot(time_axis, rotated_data[:, 2], 'b-', label='Z', alpha=0.8)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_title(f'Optimized Rotated Data{title_suffix}')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def run_optimization_analysis(heartbeats_dict: Dict, component: str = "fetal", plot_results: bool = False) -> Dict:
    """
    Run optimization analysis on heartbeat data.
    
    Args:
        heartbeats_dict: Dictionary containing heartbeat data
        component: Component to analyze ("fetal" or "maternal")
        plot_results: Whether to plot comparison for each objective
        
    Returns:
        Dictionary containing optimization results for all objectives
    """
    # Extract data
    data = np.stack([heartbeats_dict[component][key]["mean_beat"] for key in heartbeats_dict[component].keys()]).T
    
    print(f"Analyzing {component} component")
    print(f"Data shape: {data.shape}")
    
    # Try different objectives
    result = optimize_rotation_parameters(data, verbose=True)
    
    # Plot comparison only if requested
    if plot_results:
        plot_optimization_comparison(
            data,
            result['rotated_data'],
            title_suffix=f" - ({component})"
        )
            


    return result

def get_best_rotation_parameters(results: Dict) -> Tuple[str, np.ndarray]:
    """
    Get the best rotation parameters from optimization results.
    
    Args:
        results: Dictionary containing optimization results
        
    Returns:
        Tuple of (best_objective, best_angles)
    """
    best_obj = None
    best_angles = None
    best_improvement = -np.inf
    
    for obj, result in results.items():
        if result is not None:
            # Use improvement in positive fraction as the main criterion
            improvement = result['improvement_positive']
            if improvement > best_improvement:
                best_improvement = improvement
                best_obj = obj
                best_angles = result['optimal_angles']
    
    return best_obj, best_angles

