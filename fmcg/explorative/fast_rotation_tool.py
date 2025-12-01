"""
Fast and Responsive Rotation Overlay Tool for fMCG Data Analysis
==============================================================

Optimized version with:
- Debounced updates to prevent lag
- Manual update control
- Efficient rendering
- No automatic updates after manual edits
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import time
from threading import Timer

def create_rotation_matrix(rx, ry, rz):
    """Create 3D rotation matrix from Euler angles in degrees (optimized)"""
    rx_rad = np.radians(rx)
    ry_rad = np.radians(ry)
    rz_rad = np.radians(rz)
    
    # Pre-compute sin/cos values
    cx, sx = np.cos(rx_rad), np.sin(rx_rad)
    cy, sy = np.cos(ry_rad), np.sin(ry_rad)
    cz, sz = np.cos(rz_rad), np.sin(rz_rad)
    
    # Combined rotation matrix (optimized calculation)
    return np.array([
        [cy*cz, -cy*sz, sy],
        [cx*sz + sx*sy*cz, cx*cz - sx*sy*sz, -sx*cy],
        [sx*sz - cx*sy*cz, sx*cz + cx*sy*sz, cx*cy]
    ])

def apply_rotation_to_data(data, rx, ry, rz):
    """Apply rotation to 3D data (optimized)"""
    rotation_matrix = create_rotation_matrix(rx, ry, rz)
    return data @ rotation_matrix.T

class FastRotationOverlay:
    """Fast, responsive rotation overlay tool"""
    
    def __init__(self, data, title="Fast Rotation Overlay", update_delay=0.1):
        self.data = np.array(data)
        self.title = title
        self.update_delay = update_delay
        self.update_timer = None
        self.manual_mode = False
        self.last_update_time = 0
        
        self._setup_plot()
        self._setup_controls()
        self._connect_events()
        
    def _setup_plot(self):
        """Setup the main plot with optimized settings"""
        # Use lower DPI for faster rendering
        self.fig, self.ax = plt.subplots(figsize=(12, 8), dpi=80)
        plt.subplots_adjust(bottom=0.35)
        
        # Initial plot with optimized line styles
        rotated_data = self.data.copy()
        
        # Create lines with reduced detail for speed
        self.line_x, = self.ax.plot(rotated_data[:, 0], 'r-', label='X-axis', 
                                   linewidth=1.2, alpha=0.8)
        self.line_y, = self.ax.plot(rotated_data[:, 1], 'g-', label='Y-axis', 
                                   linewidth=1.2, alpha=0.8)
        self.line_z, = self.ax.plot(rotated_data[:, 2], 'b-', label='Z-axis', 
                                   linewidth=1.2, alpha=0.8)
        self.norm_line, = self.ax.plot(np.linalg.norm(rotated_data, axis=1), 
                                      'k:', label='Norm', linewidth=1.5)
        
        self.lines = [self.line_x, self.line_y, self.line_z, self.norm_line]
        
        # Setup axes with optimized settings
        self.ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        self.ax.set_xlabel('Time samples')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title(self.title)
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.2)
        
        # Pre-calculate y-limits for faster updates
        self._calculate_ylimits()
        
    def _calculate_ylimits(self):
        """Pre-calculate reasonable y-limits to avoid constant rescaling"""
        # Calculate limits from several sample rotations
        sample_rotations = [(0, 0, 0), (90, 0, 0), (0, 90, 0), (0, 0, 90), 
                           (45, 45, 45), (-45, -45, -45)]
        
        all_data = []
        for rx, ry, rz in sample_rotations:
            rotated = apply_rotation_to_data(self.data, rx, ry, rz)
            all_data.extend([rotated.flat, np.linalg.norm(rotated, axis=1)])
        
        all_values = np.concatenate(all_data)
        margin = 0.1 * (all_values.max() - all_values.min())
        self.ylim_min = all_values.min() - margin
        self.ylim_max = all_values.max() + margin
        
        self.ax.set_ylim(self.ylim_min, self.ylim_max)
        
    def _setup_controls(self):
        """Setup sliders and buttons with optimized positioning"""
        # Sliders with wider range and better spacing
        slider_height = 0.03
        slider_spacing = 0.05
        slider_start_y = 0.20
        
        slider_ax_x = plt.axes([0.15, slider_start_y, 0.4, slider_height])
        slider_ax_y = plt.axes([0.15, slider_start_y - slider_spacing, 0.4, slider_height])
        slider_ax_z = plt.axes([0.15, slider_start_y - 2*slider_spacing, 0.4, slider_height])
        
        self.slider_x = Slider(slider_ax_x, 'X Rot', -180, 180, valinit=0, 
                              valfmt='%d°', valstep=1)
        self.slider_y = Slider(slider_ax_y, 'Y Rot', -180, 180, valinit=0, 
                              valfmt='%d°', valstep=1)
        self.slider_z = Slider(slider_ax_z, 'Z Rot', -180, 180, valinit=0, 
                              valfmt='%d°', valstep=1)
        
        # Buttons with better layout
        btn_width = 0.12
        btn_height = 0.04
        btn_start_x = 0.6
        btn_spacing = 0.02
        
        # Manual update button
        update_ax = plt.axes([btn_start_x, slider_start_y, btn_width, btn_height])
        self.update_btn = Button(update_ax, 'Update Plot')
        
        # Auto/Manual mode toggle
        mode_ax = plt.axes([btn_start_x, slider_start_y - btn_height - btn_spacing, 
                           btn_width, btn_height])
        self.mode_btn = Button(mode_ax, 'Auto Mode')
        
        # Print parameters button
        print_ax = plt.axes([btn_start_x + btn_width + btn_spacing, slider_start_y, 
                           btn_width, btn_height])
        self.print_btn = Button(print_ax, 'Get Params')
        
        # Reset button
        reset_ax = plt.axes([btn_start_x + btn_width + btn_spacing, 
                           slider_start_y - btn_height - btn_spacing, 
                           btn_width, btn_height])
        self.reset_btn = Button(reset_ax, 'Reset')
        
        # Quick rotation presets
        preset_y = slider_start_y - 3*slider_spacing
        preset_buttons = []
        preset_labels = ['XY 45°', 'YZ 45°', 'XZ 45°', 'All 30°']
        preset_rotations = [(45, 45, 0), (0, 45, 45), (45, 0, 45), (30, 30, 30)]
        
        for i, (label, rotation) in enumerate(zip(preset_labels, preset_rotations)):
            btn_ax = plt.axes([0.15 + i * 0.1, preset_y, 0.08, 0.03])
            btn = Button(btn_ax, label)
            btn.rotation = rotation
            preset_buttons.append(btn)
        
        self.preset_buttons = preset_buttons
        
    def _connect_events(self):
        """Connect all event handlers"""
        # Slider events with debouncing
        self.slider_x.on_changed(self._debounced_update)
        self.slider_y.on_changed(self._debounced_update)
        self.slider_z.on_changed(self._debounced_update)
        
        # Button events
        self.update_btn.on_clicked(self._manual_update)
        self.mode_btn.on_clicked(self._toggle_mode)
        self.print_btn.on_clicked(self._print_parameters)
        self.reset_btn.on_clicked(self._reset_sliders)
        
        # Preset button events
        for btn in self.preset_buttons:
            btn.on_clicked(lambda event, r=btn.rotation: self._apply_preset(r))
            
    def _debounced_update(self, val):
        """Debounced update to prevent lag from rapid slider changes"""
        if self.manual_mode:
            return  # Don't auto-update in manual mode
            
        # Cancel previous timer
        if self.update_timer is not None:
            self.update_timer.cancel()
        
        # Start new timer
        self.update_timer = Timer(self.update_delay, self._update_plot)
        self.update_timer.start()
        
    def _update_plot(self):
        """Fast plot update with minimal recalculation"""
        current_time = time.time()
        if current_time - self.last_update_time < 0.05:  # Limit to 20 FPS
            return
            
        rx = self.slider_x.val
        ry = self.slider_y.val
        rz = self.slider_z.val
        
        # Apply rotation (optimized)
        rotated_data = apply_rotation_to_data(self.data, rx, ry, rz)
        norm_data = np.linalg.norm(rotated_data, axis=1)
        
        # Update lines efficiently
        self.lines[0].set_ydata(rotated_data[:, 0])
        self.lines[1].set_ydata(rotated_data[:, 1])
        self.lines[2].set_ydata(rotated_data[:, 2])
        self.lines[3].set_ydata(norm_data)
        
        # Only redraw the canvas, don't rescale
        self.fig.canvas.draw_idle()
        self.last_update_time = current_time
        
    def _manual_update(self, event):
        """Force update when in manual mode"""
        self._update_plot()
        
    def _toggle_mode(self, event):
        """Toggle between auto and manual update modes"""
        self.manual_mode = not self.manual_mode
        if self.manual_mode:
            self.mode_btn.label.set_text('Manual Mode')
            print("Switched to MANUAL mode - use 'Update Plot' button")
        else:
            self.mode_btn.label.set_text('Auto Mode')
            print("Switched to AUTO mode - sliders update automatically")
        self.fig.canvas.draw_idle()
        
    def _apply_preset(self, rotation):
        """Apply preset rotation"""
        rx, ry, rz = rotation
        self.slider_x.set_val(rx)
        self.slider_y.set_val(ry)
        self.slider_z.set_val(rz)
        if self.manual_mode:
            self._update_plot()
            
    def _print_parameters(self, event):
        """Print current rotation parameters"""
        rx = self.slider_x.val
        ry = self.slider_y.val
        rz = self.slider_z.val
        
        print(f"\n{'='*60}")
        print(f"CURRENT ROTATION PARAMETERS")
        print(f"{'='*60}")
        print(f"X-axis rotation: {rx:.1f}°")
        print(f"Y-axis rotation: {ry:.1f}°")
        print(f"Z-axis rotation: {rz:.1f}°")
        print(f"\nCOPY-PASTE CODE:")
        print(f"# Apply this rotation to your data:")
        print(f"rotated_data = apply_rotation_to_data(x_, {rx:.1f}, {ry:.1f}, {rz:.1f})")
        print(f"\n# Or create rotation matrix:")
        print(f"R = create_rotation_matrix({rx:.1f}, {ry:.1f}, {rz:.1f})")
        print(f"rotated_data = x_ @ R.T")
        print(f"{'='*60}")
        
    def _reset_sliders(self, event):
        """Reset all sliders to zero"""
        self.slider_x.set_val(0)
        self.slider_y.set_val(0)
        self.slider_z.set_val(0)
        if self.manual_mode:
            self._update_plot()
            
    def show(self):
        """Show the plot"""
        print(f"\nFast Rotation Overlay Tool")
        print(f"{'='*30}")
        print(f"Controls:")
        print(f"  • Drag sliders to rotate data")
        print(f"  • Click 'Auto/Manual Mode' to toggle update behavior")
        print(f"  • In Manual mode: drag sliders then click 'Update Plot'")
        print(f"  • Use preset buttons for common rotations")
        print(f"  • Click 'Get Params' for rotation values")
        print(f"  • Click 'Reset' to return to 0°")
        print(f"{'='*30}")
        
        plt.show()
        return self.fig, [self.slider_x, self.slider_y, self.slider_z], self.update_btn

def fast_rotation_overlay(data, title="Fast Rotation Overlay"):
    """
    Create fast, responsive rotation overlay tool
    
    Parameters:
    -----------
    data : array-like, shape (N, 3)
        3D time series data
    title : str
        Title for the plot
        
    Returns:
    --------
    FastRotationOverlay instance
    """
    tool = FastRotationOverlay(data, title)
    tool.show()
    return tool

def quick_rotation_compare(data, rotations, labels=None):
    """
    Quickly compare multiple rotations in static plots
    
    Parameters:
    -----------
    data : array-like, shape (N, 3)
        3D time series data
    rotations : list of tuples
        List of (rx, ry, rz) rotation parameters
    labels : list of str, optional
        Labels for each rotation
    """
    n_rotations = len(rotations)
    fig, axes = plt.subplots(n_rotations, 1, figsize=(10, 3*n_rotations), 
                            sharex=True, dpi=100)
    
    if n_rotations == 1:
        axes = [axes]
    
    for i, (rx, ry, rz) in enumerate(rotations):
        rotated_data = apply_rotation_to_data(data, rx, ry, rz)
        
        axes[i].plot(rotated_data[:, 0], 'r-', label='X', linewidth=1.2, alpha=0.8)
        axes[i].plot(rotated_data[:, 1], 'g-', label='Y', linewidth=1.2, alpha=0.8)
        axes[i].plot(rotated_data[:, 2], 'b-', label='Z', linewidth=1.2, alpha=0.8)
        axes[i].plot(np.linalg.norm(rotated_data, axis=1), 'k:', 
                    label='Norm', linewidth=1.5)
        
        axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[i].set_ylabel('Amplitude')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.2)
        
        title = labels[i] if labels else f"Rotation {i+1}: ({rx}°, {ry}°, {rz}°)"
        axes[i].set_title(title)
    
    axes[-1].set_xlabel('Time samples')
    plt.tight_layout()
    plt.show()

def run_fast_rotation_analysis(heartbeats_dict):
    """
    Fast version of rotation analysis with optimized performance
    """
    print("="*60)
    print("FAST FETAL HEARTBEAT ROTATION ANALYSIS")
    print("="*60)
    
    # Extract fetal data
    try:
        x_ = np.stack([heartbeats_dict["fetal"][key]["mean_beat"] 
                      for key in heartbeats_dict["fetal"].keys()], axis=0).T
        print(f"✓ Extracted fetal data: {x_.shape}")
    except Exception as e:
        print(f"✗ Error extracting data: {e}")
        return None
    
    # Create fast rotation tool
    tool = fast_rotation_overlay(x_, "Fast Fetal Heartbeat Rotation")
    return tool

if __name__ == "__main__":
    # Demo with synthetic data
    print("Fast Rotation Tool - Demo Mode")
    t = np.linspace(0, 2*np.pi, 300)
    x = np.sin(t) + 0.3*np.sin(3*t)
    y = 0.7*np.cos(t) + 0.2*np.sin(5*t)
    z = 0.5*np.sin(2*t) + 0.1*np.cos(7*t)
    demo_data = np.column_stack([x, y, z])
    
    tool = fast_rotation_overlay(demo_data, "Demo - Fast Rotation Tool")
