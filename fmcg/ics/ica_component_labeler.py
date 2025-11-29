import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.gridspec import GridSpec
from copy import deepcopy

import scipy


import fmcg

try:
    import ipywidgets as widgets
    from IPython.display import display
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    import matplotlib.widgets as mpl_widgets
    IPYWIDGETS_AVAILABLE = False

from scipy import stats
import pandas as pd
import neurokit2 as nk
from datetime import datetime


class ICAComponentLabeler:


    def __init__(self, params, n_components=30, seed=42, xlim=None,
                 fig_size=(12.0, 8.0), canvas_size=("1000px", "600px"), 
                 grid_cols=2, max_height="600px", save_suffix="", dpi=100, base_path=None,
                 hr_tolerance=5.0, correlation_threshold=0.7):
        """Interactive GUI for labeling ICA components with scrollable grid layout.

        sources: array (T x C)
        ica: fitted FastICA (has mixing_)
        fs: sampling rate
        time: optional time vector (len T)
        xlim: optional (t0, t1) seconds window to crop
        fig_size: matplotlib figure size in inches for grid view
        canvas_size: ipympl canvas size (width, height) as CSS strings
        grid_cols: number of components per row in grid
        max_height: maximum height of scrollable container
        save_suffix: optional suffix for saved files
        hr_tolerance: heart rate tolerance in BPM for similarity highlighting (default: 5.0)
        correlation_threshold: threshold for highlighting cross-correlation scores (default: 0.7)
        """

        self.n_components = n_components
        self.current_component = 0
        self.seed = seed
        self.base_path = base_path
    
        self._load_and_preprocess_data(params)
        self._apply_ica()
        
        # Store original data before any windowing
        self.time_orig = self.time.copy() if self.time is not None else None
        self.sources_orig_full = self.sources.copy()

        # Optional crop
        if xlim is not None and self.time is not None:
            mask = (self.time >= xlim[0]) & (self.time <= xlim[1])

            # Only apply mask if dimensions are compatible
            if self.sources.shape[0] == self.time.shape[0]:
                self.sources = self.sources[mask, :]
                self.time = self.time[mask]
            else:
                print(f"Warning: Cannot apply time mask - shape mismatch!")
                print(f"sources has {self.sources.shape[0]} samples, time has {self.time.shape[0]} samples")
        self.xlim = xlim
        self.save_suffix = save_suffix
        self.dpi = dpi

        # Sizing
        self._fig_size = fig_size
        self._canvas_size = canvas_size
        self._grid_cols = grid_cols
        self._max_height = max_height

        # Labels
        self.labels = {}
        self.label_options = ['Fetal', 'Maternal', 'Mixed', 'Noise']
        self.label_colors = ['red', 'blue', 'orange', 'gray']
        
        # Reference components for peak plotting
        self.fetal_reference = None
        self.maternal_reference = None
        self.reference_peaks = {}  # Store peaks for reference components
        
        # Cross-correlation similarity scores
        self.fetal_similarities = {}  # Store cross-correlation scores to fetal reference
        self.maternal_similarities = {}  # Store cross-correlation scores to maternal reference
        
        # Heart rate similarity settings
        self.hr_tolerance = hr_tolerance  # BPM tolerance for similar heart rates
        self.correlation_threshold = correlation_threshold  # Threshold for highlighting correlations
        self.fetal_hr_similar = set()  # Components with similar HR to fetal reference
        self.maternal_hr_similar = set()  # Components with similar HR to maternal reference
        
        # Pre-compute peaks for all components and sort by metrics
        self.all_peaks = {}
        self._compute_all_peaks()
        
        # Compute sorting metrics and create sorted component order
        self.component_order = self._compute_sorted_component_order()

        # Set initial window parameters using original time data
        if self.time_orig is not None:
            self.total_duration = float(self.time_orig[-1] - self.time_orig[0])
            if xlim is not None:
                self.window_start = float(xlim[0])
                self.window_length = float(xlim[1] - xlim[0])
            else:
                self.window_start = float(self.time_orig[0])
                self.window_length = self.total_duration
        else:
            self.total_duration = 1.0
            self.window_start = 0.0
            self.window_length = 1.0

        self.setup_gui()

    def _load_and_preprocess_data(self, params):
        print("Loading and preprocessing data...")
        # load patient data
        sig_data_dict_raw, time, fs, noise_data_dict_raw = fmcg.data.load_structured_patient_data_and_noise(
            params["SystemConfig"],
            params["MeasurementConfig"],
            **{k: params[k] for k in params.keys() & {"ds_path", "patient", "series", "sig_group_names", "noise_group_names"}},
            files="all",
            print_groups=True,
        )

        print(f"Signal Length: {len(list(sig_data_dict_raw.values())[0])/fs} s")
        print(f"Noise Length: {len(list(noise_data_dict_raw.values())[0])/fs} s")

        # performing basic preprocessing (filtering and removal of outlying sensor channels)

        # frequency filtering (bandpass + notch)
        sensor_dict = fmcg.signal.filter_sensor_dict(sig_data_dict_raw, fs, low=params['bandpass_low'], high=params["bandpass_high"], order=4)
        noise_sensor_dict = fmcg.signal.filter_sensor_dict(noise_data_dict_raw, fs, low=params['bandpass_low'], high=params["bandpass_high"], order=4)

        # remove outlying channels with unphysical range (by variability threshold + comparison across array)
        sensor_dict, axis_mask = fmcg.signal.remove_outlier_dict(sensor_dict, physical_range_threshold=1, verbose=True)
        noise_sensor_dict, axis_mask_noise = fmcg.signal.remove_outlier_dict(noise_sensor_dict, physical_range_threshold=.1, verbose=True)

        # axis mask provides valid channels
        axis_mask = axis_mask & axis_mask_noise


        field_maps = np.array([v for k, v in sensor_dict.items()]).transpose(1, 0, 2)
        noise_maps = np.array([v for k, v in noise_sensor_dict.items()]).transpose(1, 0, 2)

        if params["whitening"]:
            # Apply whitening to the recording using noise structure
            field_maps, W = fmcg.whitening.apply_whitening(
                field_maps,
                noise_maps,
                axis_mask,
                method="PCA",
                return_whitening_matrix=True,
                rescale=True,
            )

        sampfrom = params.get("sampfrom", None)
        sampto = params.get("sampto", None)
        if sampfrom is not None:
            sampfrom_idx = int(sampfrom * fs)
            if sampto is not None:
                sampto_idx = int(sampto * fs)
                field_maps = field_maps[sampfrom_idx:sampto_idx]
                time = time[sampfrom_idx:sampto_idx] - time[sampfrom_idx]
            else:
                field_maps = field_maps[sampfrom_idx:]
                time = time[sampfrom_idx:] - time[sampfrom_idx]
            print(f"Cropped data from {sampfrom} to {sampto} samples")

        if params.get("downsample_factor", False):
            q = params.get("downsample_factor")
            print(f"Downsampling by factor {q}")
            fs = int(fs / q)
            field_maps = scipy.signal.decimate(field_maps, q, axis=0)
            time = np.linspace(time[0], time[-1], num=field_maps.shape[0], endpoint=True)  

        self.bp_data = field_maps[:, axis_mask]
        self.axis_mask = axis_mask
        self.time = time
        self.fs = fs

    def _apply_ica(self):
        print("Applying ICA...")
        sources, self.ica = fmcg.ics.apply_fastICA(self.bp_data, n_comp=self.n_components, algo='parallel')
        self.sources_orig = sources.copy()
        self.ica_orig = self.ica

        self.sources = sources
        self.n_components = sources.shape[1]
        print(f"Extracted {self.n_components} ICA components")
   


    def _compute_hr_from_peaks(self, peaks):
        """Compute heart rate in BPM from R-peaks"""
        if len(peaks) < 2:
            return 0.0
        rr_intervals = np.diff(peaks) / self.fs  # Convert to seconds
        return 60.0 / np.mean(rr_intervals)  # Convert to BPM
    
    def _find_hr_similar_components(self, ref_comp_idx, ref_type='fetal'):
        """Find components with similar heart rates to reference component
        
        Args:
            ref_comp_idx: Index of reference component
            ref_type: 'fetal' or 'maternal' to specify which similarity set to update
        """
        ref_peaks = self.all_peaks[ref_comp_idx]
        ref_hr = self._compute_hr_from_peaks(ref_peaks)
        
        similar_components = set()
        
        for comp_idx in range(self.n_components):
            if comp_idx == ref_comp_idx:
                continue
                
            comp_peaks = self.all_peaks[comp_idx]
            comp_hr = self._compute_hr_from_peaks(comp_peaks)
            
            # Check if HR is within tolerance
            if abs(comp_hr - ref_hr) <= self.hr_tolerance and comp_hr > 0:
                similar_components.add(comp_idx)
        
        # Store in appropriate similarity set
        if ref_type == 'fetal':
            self.fetal_hr_similar = similar_components
        else:
            self.maternal_hr_similar = similar_components
        
        print(f"{ref_type.capitalize()} reference HR: {ref_hr:.1f} BPM")
        print(f"Found {len(similar_components)} components with similar HR (±{self.hr_tolerance} BPM)")

    def _compute_cross_correlation(self, ref_comp_idx, target_comp_idx):
        """Compute normalized cross-correlation between two components
        
        Returns the maximum cross-correlation coefficient (absolute value)
        """
        ref_signal = self.sources[:, ref_comp_idx]
        target_signal = self.sources[:, target_comp_idx]
        
        # Normalize signals to zero mean and unit variance
        ref_norm = (ref_signal - np.mean(ref_signal)) / np.std(ref_signal)
        target_norm = (target_signal - np.mean(target_signal)) / np.std(target_signal)
        
        # Compute cross-correlation
        correlation = np.correlate(ref_norm, target_norm, mode='full')
        
        # Normalize by signal length to get correlation coefficient
        correlation = correlation / len(ref_norm)
        
        # Return maximum absolute correlation
        return float(np.max(np.abs(correlation)))
    
    def _compute_all_similarities(self, ref_comp_idx, ref_type='fetal'):
        """Compute cross-correlation similarities for all components relative to reference
        
        Args:
            ref_comp_idx: Index of reference component
            ref_type: 'fetal' or 'maternal' to specify which similarity dict to update
        """
        similarities = {}
        
        for comp_idx in range(self.n_components):
            if comp_idx == ref_comp_idx:
                similarities[comp_idx] = 1.0  # Perfect correlation with itself
            else:
                similarities[comp_idx] = self._compute_cross_correlation(ref_comp_idx, comp_idx)
        
        # Store in appropriate similarity dictionary
        if ref_type == 'fetal':
            self.fetal_similarities = similarities
        else:
            self.maternal_similarities = similarities

    def _compute_all_peaks(self):
        """Pre-compute R-peaks for all components using original full sources"""
        for comp_idx in range(self.n_components):
            # Use original full sources for consistent peak detection
            sig = self.sources_orig_full[:, comp_idx]
            sig, _ = nk.ecg_invert(sig, self.fs)
            signal_clean = nk.ecg_clean(sig, sampling_rate=self.fs, method="vg")
            peaks = nk.ecg_findpeaks(signal_clean, sampling_rate=self.fs, method="vg")[
                "ECG_R_Peaks"
            ]
            self.all_peaks[comp_idx] = peaks

    def _compute_sorted_component_order(self):
        """Compute component metrics and return sorted order by kurtosis and SDNN"""
        metrics = []
        
        for comp_idx in range(self.n_components):
            sig = self.sources[:, comp_idx]
            peaks = self.all_peaks[comp_idx]
            
            # Compute kurtosis
            kurtosis = float(stats.kurtosis(sig))
            
            # Compute SDNN (standard deviation of NN intervals)
            sdnn = (np.diff(peaks).std() / self.fs) if len(peaks) > 1 else 0
            
            metrics.append({
                'comp_idx': comp_idx,
                'kurtosis': kurtosis,
                'sdnn': sdnn
            })
        
        # Sort by kurtosis (descending) then by SDNN (descending)
        sorted_metrics = sorted(metrics, key=lambda x: (x['sdnn'], -x['kurtosis']))
        
        # Return the sorted component indices
        return [m['comp_idx'] for m in sorted_metrics]

    def setup_gui(self):
        if IPYWIDGETS_AVAILABLE:
            self._setup_jupyter_gui()
        else:
            self._setup_matplotlib_gui()

    def _setup_jupyter_gui(self):
        # Create individual component rows
        self.component_rows = []
        self.component_figures = []
        self.component_selectors = []
        self.fetal_checkboxes = []
        self.maternal_checkboxes = []
        self.stats_widgets = []  # Store references to stats widgets
        
        # Create mapping from original component index to display position
        self.comp_idx_to_display_pos = {comp_idx: i for i, comp_idx in enumerate(self.component_order)}
        
        # Create scrollable container for all rows
        all_rows = []
        
        for i, comp_idx in enumerate(self.component_order):
            row_widget = self._create_component_row(comp_idx, display_rank=i+1)
            all_rows.append(row_widget)
        
        # Add global controls at the top
        global_controls = self._create_global_controls()
        
        # Add global controls at the bottom (same as top)
        bottom_controls = self._create_global_controls(is_bottom=True)
        
        # Create container without height constraint - let it grow naturally
        scrollable_content = widgets.VBox(
            [global_controls] + all_rows + [bottom_controls],
            layout=widgets.Layout(
                overflow_y='visible',  # No scrolling constraint
                overflow_x='hidden',
                width='100%'
                # Remove height constraint to prevent row truncation
            )
        )
        
        self.app = scrollable_content

    def _create_component_row(self, comp_idx, display_rank=None):
        """Create a row widget for a single component with plot, stats, and controls"""

        # Create individual figure for this component
        was_interactive = plt.isinteractive()
        plt.ioff()
        
        # Use larger figure size for better visibility
        fig = plt.figure(figsize=self._fig_size, dpi=self.dpi)
        # Remove figure number/title to avoid "Figure 2" labels
        fig.suptitle('')
        try:
            fig.canvas.set_window_title('')  # Remove window title
        except AttributeError:
            pass  # Ignore if not supported by backend
        
        ax = fig.add_subplot(111)
        
        # Disable axis navigation and formatting that shows coordinates
        ax.format_coord = lambda x, y: ''  # Remove coordinate display on hover
        
        # Plot the component
        sig = self.sources[:, comp_idx]

        # Use windowed peaks for display metrics
        windowed_peak_indices = self._get_windowed_peaks(comp_idx)
        hr, sdnn = self._get_windowed_hr_and_sdnn(comp_idx)

        ax.plot(self.time, sig, 'k-', linewidth=0.7)
        
        # Plot peaks using windowed peak indices
        windowed_peak_indices = self._get_windowed_peaks(comp_idx)
        if len(windowed_peak_indices) > 0:
            ax.plot(self.time[windowed_peak_indices], sig[windowed_peak_indices], 'rx', markersize=3)
        
        # Update title to show both rank and original component number
        title = f'#{display_rank} (Comp {comp_idx + 1})' if display_rank else f'Component {comp_idx + 1}'
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Time [s]', fontsize=7)
        ax.set_ylabel('Amplitude', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        
        if self.xlim is not None:
            ax.set_xlim(self.xlim)
        
        # Reduce padding to minimum and adjust subplot parameters for tighter layout
        fig.subplots_adjust(left=0.05, bottom=0.08, right=.98, top=.92, wspace=0, hspace=0)
        fig.tight_layout(pad=0.01)
        
        if was_interactive:
            plt.ion()
        
        # Store figure reference
        self.component_figures.append(fig)
        
        # Use canvas widget directly without height constraints
        canvas_widget = fig.canvas
        canvas_widget.layout.width = '450px'
        canvas_widget.layout.margin = '0px'
        canvas_widget.layout.padding = '0px'
        
        # Disable interactive features completely
        # Remove toolbar to disable zoom, pan, etc.
        canvas_widget.toolbar_visible = False
        # Disable cursor coordinate display
        canvas_widget.header_visible = False
        # Disable figure title/number display
        canvas_widget.footer_visible = False
        # Disable resizing
        canvas_widget.resizable = False
        
        # Disable all matplotlib navigation and interaction
        try:
            # Handle different backend toolbar hiding approaches
            if hasattr(fig.canvas, 'toolbar') and fig.canvas.toolbar:
                toolbar = fig.canvas.toolbar
                # Try different methods for different backends
                if hasattr(toolbar, 'pack_forget'):
                    toolbar.pack_forget()  # tkinter backend
                elif hasattr(toolbar, 'setVisible'):
                    toolbar.setVisible(False)  # Qt backend
                elif hasattr(toolbar, 'hide'):
                    toolbar.hide()  # Some other backends
        except (AttributeError, TypeError):
            # If toolbar hiding fails, continue silently
            pass
        
        # Remove all event connections to disable mouse interactions
        try:
            # Disconnect all existing callbacks
            fig.canvas.mpl_disconnect('all')
            # Disable navigation mode
            if hasattr(fig.canvas, 'toolbar') and fig.canvas.toolbar:
                fig.canvas.toolbar.mode = ''
        except (AttributeError, TypeError):
            pass  # Ignore if disconnection fails
        
        # Remove height constraint to allow full plot display
        
        # Create statistics panel
        kurt = self.compute_kurtosis(sig)
        var = float(np.var(sig))
        mix_weights = self.ica.mixing_[:, comp_idx]
        max_mix = float(np.max(np.abs(mix_weights)))

        # Get similarity scores if available
        fetal_sim = self.fetal_similarities.get(comp_idx, None)
        maternal_sim = self.maternal_similarities.get(comp_idx, None)
        
        # Check HR similarity
        is_fetal_hr_similar = comp_idx in self.fetal_hr_similar
        is_maternal_hr_similar = comp_idx in self.maternal_hr_similar
        
        # Color the HR value based on similarity
        hr_text = f"HR: {hr:.3f} bpm"
        if is_fetal_hr_similar:
            hr_text = f"<span style='color: red; font-weight: bold;'>HR: {hr:.3f} bpm</span>"
        elif is_maternal_hr_similar:
            hr_text = f"<span style='color: blue; font-weight: bold;'>HR: {hr:.3f} bpm</span>"
        
        # Build similarity text
        sim_text = ""
        if fetal_sim is not None:
            if fetal_sim > self.correlation_threshold:
                sim_text += f"<br><span style='color: red; font-weight: bold;'>Fetal Sim: {fetal_sim:.3f}</span>"
            else:
                sim_text += f"<br>Fetal Sim: {fetal_sim:.3f}"
        if maternal_sim is not None:
            if maternal_sim > self.correlation_threshold:
                sim_text += f"<br><span style='color: blue; font-weight: bold;'>Maternal Sim: {maternal_sim:.3f}</span>"
            else:
                sim_text += f"<br>Maternal Sim: {maternal_sim:.3f}"
        
        stats_html = f"""
        <b>#{display_rank} (Comp {comp_idx + 1})</b><br>
        Kurtosis: {kurt:.3f}<br>
        Variance: {var:.3f}<br>
        Max |Mix|: {max_mix:.3f}<br>
        #R-peaks: {len(windowed_peak_indices)}<br>
        {hr_text}<br>
        SDNN: {sdnn:.3f}{sim_text}
        """ if display_rank else f"""
        <b>Comp {comp_idx + 1}</b><br>
        Kurtosis: {kurt:.3f}<br>
        Variance: {var:.3f}<br>
        Max |Mix|: {max_mix:.3f}<br>
        #R-peaks: {len(windowed_peak_indices)}<br>
        {hr_text}<br>
        SDNN: {sdnn:.3f}{sim_text}
        """
        
        stats_widget = widgets.HTML(
            value=stats_html,
            layout=widgets.Layout(width='160px')
            # Remove height constraint to allow natural sizing
        )
        
        # Store stats widget reference
        self.stats_widgets.append(stats_widget)
        
        # Create label selection radio buttons
        label_selector = widgets.RadioButtons(
            options=[('Unlabeled', None)] + [(label, label) for label in self.label_options],
            value=None,
            description='',
            layout=widgets.Layout(width='160px'),
            # Remove height constraint to allow natural sizing
            style={'description_width': '0px'}
        )
        
        # Store selector reference and bind event
        self.component_selectors.append(label_selector)
        label_selector.observe(lambda change, idx=comp_idx: self._on_label_changed(idx, change), names='value')
        
        # Create reference checkboxes
        # Create reference checkboxes with compact styling
        fetal_checkbox = widgets.Checkbox(
            value=False,
            description='Fetal Ref',
            disabled=False,
            layout=widgets.Layout(width='160px'),
            style={'description_width': '0px'}
        )
        maternal_checkbox = widgets.Checkbox(
            value=False,
            description='Maternal Ref',
            disabled=False,
            layout=widgets.Layout(width='160px'),
            style={'description_width': '0px'}
        )
        
        # Store checkbox references and bind events
        self.fetal_checkboxes.append(fetal_checkbox)
        self.maternal_checkboxes.append(maternal_checkbox)
        fetal_checkbox.observe(lambda change, idx=comp_idx: self._on_fetal_reference_changed(idx, change), names='value')
        maternal_checkbox.observe(lambda change, idx=comp_idx: self._on_maternal_reference_changed(idx, change), names='value')
        
        # Create controls column with checkboxes directly below radio buttons
        controls_column = widgets.VBox([
            label_selector,
            widgets.HTML("<small><b>References:</b></small>", layout=widgets.Layout(margin='3px 0px 1px 0px')),
            fetal_checkbox,
            maternal_checkbox
        ], layout=widgets.Layout(width='160px'))
        
        # Create horizontal layout for this row with natural height
        row = widgets.HBox([
            canvas_widget,
            stats_widget, 
            controls_column
        ], layout=widgets.Layout(
            margin='1px 0px',  # Reduced margin from 10px to 2px
            border='1px solid #ddd', 
            padding='1px',  # Reduced padding from 10px to 5px
            align_items='flex-start'  # Align items to top instead of stretching
        ))
        
        return row

    def _create_global_controls(self, is_bottom=False):
        """Create global control panel"""
        # Progress info
        if not is_bottom:
            self.progress_label = widgets.HTML(
                value=f"<h3>ICA Component Labeler - Progress: 0/{self.n_components} labeled</h3>"
            )
        else:
            # For bottom controls, create a separate progress label that will be updated
            self.progress_label_bottom = widgets.HTML(
                value=f"<h3>ICA Component Labeler - Progress: 0/{self.n_components} labeled</h3>"
            )
        
        # Export and clear buttons
        export_button = widgets.Button(
            description='Export Labels', 
            button_style='success',
            layout=widgets.Layout(width='120px')
        )
        export_button.on_click(self.export_labels)
        
        clear_button = widgets.Button(
            description='Clear All', 
            button_style='warning',
            layout=widgets.Layout(width='120px')
        )
        clear_button.on_click(self.clear_all_labels)
        
        # Window control sliders (only for top controls to avoid duplication)
        window_controls = None
        if not is_bottom and self.time_orig is not None and IPYWIDGETS_AVAILABLE:
            # Window start slider
            self.window_start_slider = widgets.FloatSlider(
                value=self.window_start,
                min=float(self.time_orig[0]),
                max=float(self.time_orig[-1] - self.window_length),
                step=0.1,
                description='Window Start (s):',
                layout=widgets.Layout(width='400px'),
                style={'description_width': '120px'}
            )
            self.window_start_slider.observe(self._on_window_changed, names='value')
            
            # Window length slider
            self.window_length_slider = widgets.FloatSlider(
                value=self.window_length,
                min=1.0,
                max=self.total_duration,
                step=0.1,
                description='Window Length (s):',
                layout=widgets.Layout(width='400px'),
                style={'description_width': '120px'}
            )
            self.window_length_slider.observe(self._on_window_changed, names='value')
            
            # Reset to full view button
            reset_view_button = widgets.Button(
                description='Reset View',
                button_style='info',
                layout=widgets.Layout(width='120px')
            )
            reset_view_button.on_click(self._reset_window_view)
            
            window_controls = widgets.VBox([
                widgets.HTML("<b>Time Window Controls:</b>"),
                self.window_start_slider,
                self.window_length_slider,
                reset_view_button,
                widgets.HTML("<hr>")
            ])
        
        # Store references to buttons for the first (top) instance only
        if not is_bottom:
            self.export_button = export_button
            self.clear_button = clear_button
               
        buttons = widgets.HBox([
            export_button, 
            clear_button, 
        ])
        
        if not is_bottom:
            controls = [self.progress_label, buttons]
            if window_controls is not None:
                controls.insert(1, window_controls)  # Insert window controls between progress and buttons
            controls.append(widgets.HTML("<hr>"))
            return widgets.VBox(controls)
        else:
            return widgets.VBox([
                widgets.HTML("<hr>"),
                self.progress_label_bottom,
                buttons
            ])

    def _setup_matplotlib_gui(self):
        # Non-widget fallback - create grid layout
        grid_rows = int(np.ceil(self.n_components / self._grid_cols))
        self.fig = plt.figure(figsize=self._fig_size, dpi=self.dpi)
        self.fig.suptitle(f'ICA Components Overview ({self.n_components} components)', fontsize=14)
        
        # Create subplots for all components
        self.axes = []
        for i in range(self.n_components):
            ax = plt.subplot(grid_rows, self._grid_cols, i + 1)
            self.axes.append(ax)
            
        self._setup_matplotlib_controls()
        self._draw_all_components()

    def _on_label_changed(self, comp_idx, change):
        """Handle label change for a specific component"""
        new_label = change['new']
        
        if new_label is None:
            # Remove label if "Unlabeled" is selected
            if comp_idx in self.labels:
                del self.labels[comp_idx]
        else:
            # Set new label
            self.labels[comp_idx] = new_label
        
        # Update the component's plot background color
        self._update_component_appearance(comp_idx)
        
        # Update progress
        self._update_progress()

    def _on_fetal_reference_changed(self, comp_idx, change):
        """Handle fetal reference checkbox change"""
        if change['new']:  # Checkbox checked
            # Uncheck previous fetal reference if any
            if self.fetal_reference is not None:
                prev_display_pos = self.comp_idx_to_display_pos[self.fetal_reference]
                self.fetal_checkboxes[prev_display_pos].value = False
            
            # Set new fetal reference
            self.fetal_reference = comp_idx
            self.reference_peaks['fetal'] = self.all_peaks[comp_idx]
            
            # Compute cross-correlation similarities
            self._compute_all_similarities(comp_idx, 'fetal')
            
            # Find HR similar components
            self._find_hr_similar_components(comp_idx, 'fetal')
            
        else:  # Checkbox unchecked
            if self.fetal_reference == comp_idx:
                self.fetal_reference = None
                if 'fetal' in self.reference_peaks:
                    del self.reference_peaks['fetal']
                # Clear similarities when reference is removed
                self.fetal_similarities = {}
                self.fetal_hr_similar = set()
        
        # Update all plots and stats
        self._update_all_plots()
        self._update_all_stats()

    def _on_maternal_reference_changed(self, comp_idx, change):
        """Handle maternal reference checkbox change"""
        if change['new']:  # Checkbox checked
            # Uncheck previous maternal reference if any
            if self.maternal_reference is not None:
                prev_display_pos = self.comp_idx_to_display_pos[self.maternal_reference]
                self.maternal_checkboxes[prev_display_pos].value = False
            
            # Set new maternal reference
            self.maternal_reference = comp_idx
            self.reference_peaks['maternal'] = self.all_peaks[comp_idx]
            
            # Compute cross-correlation similarities
            self._compute_all_similarities(comp_idx, 'maternal')
            
            # Find HR similar components
            self._find_hr_similar_components(comp_idx, 'maternal')
            
        else:  # Checkbox unchecked
            if self.maternal_reference == comp_idx:
                self.maternal_reference = None
                if 'maternal' in self.reference_peaks:
                    del self.reference_peaks['maternal']
                # Clear similarities when reference is removed
                self.maternal_similarities = {}
                self.maternal_hr_similar = set()
        
        # Update all plots and stats
        self._update_all_plots()
        self._update_all_stats()

    def _update_all_stats(self):
        """Update all statistics widgets to show current similarity scores"""
        for i, comp_idx in enumerate(self.component_order):
            display_pos = i
            self._update_single_stats_at_position(comp_idx, display_pos)
    
    def _update_single_stats_at_position(self, comp_idx, display_pos):
        """Update statistics widget at specific display position"""
        if display_pos >= len(self.stats_widgets):
            return
            
        stats_widget = self.stats_widgets[display_pos]
        
        # Recalculate stats using windowed data
        sig = self.sources[:, comp_idx]
        windowed_peak_indices = self._get_windowed_peaks(comp_idx)
        hr, sdnn = self._get_windowed_hr_and_sdnn(comp_idx)
        kurt = self.compute_kurtosis(sig)
        var = float(np.var(sig))
        mix_weights = self.ica.mixing_[:, comp_idx]
        max_mix = float(np.max(np.abs(mix_weights)))
        
        # Get similarity scores if available
        fetal_sim = self.fetal_similarities.get(comp_idx, None)
        maternal_sim = self.maternal_similarities.get(comp_idx, None)
        
        # Check HR similarity
        is_fetal_hr_similar = comp_idx in self.fetal_hr_similar
        is_maternal_hr_similar = comp_idx in self.maternal_hr_similar
        
        # Color the HR value based on similarity
        hr_text = f"HR: {hr:.3f} bpm"
        if is_fetal_hr_similar:
            hr_text = f"<span style='color: red; font-weight: bold;'>HR: {hr:.3f} bpm</span>"
        elif is_maternal_hr_similar:
            hr_text = f"<span style='color: blue; font-weight: bold;'>HR: {hr:.3f} bpm</span>"
        
        # Build similarity text with configurable threshold
        sim_text = ""
        if fetal_sim is not None:
            if fetal_sim > self.correlation_threshold:  # Use configurable threshold
                sim_text += f"<br><span style='color: red; font-weight: bold;'>Fetal Sim: {fetal_sim:.3f}</span>"
            else:
                sim_text += f"<br>Fetal Sim: {fetal_sim:.3f}"
        if maternal_sim is not None:
            if maternal_sim > self.correlation_threshold:  # Use configurable threshold
                sim_text += f"<br><span style='color: blue; font-weight: bold;'>Maternal Sim: {maternal_sim:.3f}</span>"
            else:
                sim_text += f"<br>Maternal Sim: {maternal_sim:.3f}"
        
        display_rank = display_pos + 1
        stats_html = f"""
        <b>#{display_rank} (Comp {comp_idx + 1})</b><br>
        Kurtosis: {kurt:.3f}<br>
        Variance: {var:.3f}<br>
        Max |Mix|: {max_mix:.3f}<br>
        #R-peaks: {len(windowed_peak_indices)}<br>
        {hr_text}<br>
        SDNN: {sdnn:.3f}{sim_text}
        """
        
        stats_widget.value = stats_html

    def _update_all_plots(self):
        """Update all component plots to show reference peaks in sorted order"""
        for i, comp_idx in enumerate(self.component_order):
            self._update_single_plot_at_position(comp_idx, i)

    def _update_single_plot_at_position(self, comp_idx, display_pos):
        """Update a single component plot with reference peaks at the correct display position"""
        fig = self.component_figures[display_pos]
        ax = fig.axes[0]
        
        # Clear the axes and redraw
        ax.clear()
        
        # Disable coordinate display on hover after clearing
        ax.format_coord = lambda x, y: ''
        
        # Plot the component signal
        sig = self.sources[:, comp_idx]
        ax.plot(self.time, sig, 'k-', linewidth=0.7)
        
        # Plot own peaks using windowed peak indices
        windowed_peak_indices = self._get_windowed_peaks(comp_idx)
        if len(windowed_peak_indices) > 0:
            ax.plot(self.time[windowed_peak_indices], sig[windowed_peak_indices], 'rx', markersize=3, label='Own peaks')
        
        # Plot reference peaks if they exist
        y_min, y_max = ax.get_ylim()
        
        # Plot fetal reference peaks
        if 'fetal' in self.reference_peaks and comp_idx != self.fetal_reference:
            fetal_peak_times = self._get_windowed_reference_peak_times(self.reference_peaks['fetal'])
            if len(fetal_peak_times) > 0:
                ax.vlines(fetal_peak_times, y_min, y_max, colors='red', alpha=0.5, 
                         linestyles='--', linewidth=1, label='Fetal ref')
        
        # Plot maternal reference peaks
        if 'maternal' in self.reference_peaks and comp_idx != self.maternal_reference:
            maternal_peak_times = self._get_windowed_reference_peak_times(self.reference_peaks['maternal'])
            if len(maternal_peak_times) > 0:
                ax.vlines(maternal_peak_times, y_min, y_max, colors='blue', alpha=0.5, 
                         linestyles=':', linewidth=1, label='Maternal ref')
        
        # Restore plot formatting with correct rank
        display_rank = display_pos + 1
        title = f'#{display_rank} (Comp {comp_idx + 1})'
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Time [s]', fontsize=7)
        ax.set_ylabel('Amplitude', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        
        if self.xlim is not None:
            ax.set_xlim(self.xlim)
        
        # Add legend if there are reference peaks
        if ('fetal' in self.reference_peaks and comp_idx != self.fetal_reference) or \
           ('maternal' in self.reference_peaks and comp_idx != self.maternal_reference):
            ax.legend(fontsize=6, loc='upper right')
        
        # Apply tight layout to reduce margins
        fig.subplots_adjust(left=0.05, bottom=0.08, right=.98, top=.92, wspace=0, hspace=0)
        
        # Restore background color based on label
        self._update_component_appearance_at_position(comp_idx, display_pos)
        
        # Redraw
        fig.canvas.draw_idle()

    def _update_single_plot(self, comp_idx):
        """Update a single component plot with reference peaks"""
        fig = self.component_figures[comp_idx]
        ax = fig.axes[0]
        
        # Clear the axes and redraw
        ax.clear()
        
        # Disable coordinate display on hover after clearing
        ax.format_coord = lambda x, y: ''
        
        # Plot the component signal
        sig = self.sources[:, comp_idx]
        ax.plot(self.time, sig, 'k-', linewidth=0.7)
        
        # Plot own peaks using windowed peak indices
        windowed_peak_indices = self._get_windowed_peaks(comp_idx)
        if len(windowed_peak_indices) > 0:
            ax.plot(self.time[windowed_peak_indices], sig[windowed_peak_indices], 'rx', markersize=3, label='Own peaks')
        
        # Plot reference peaks if they exist
        y_min, y_max = ax.get_ylim()
        
        # Plot fetal reference peaks
        if 'fetal' in self.reference_peaks and comp_idx != self.fetal_reference:
            fetal_peak_times = self._get_windowed_reference_peak_times(self.reference_peaks['fetal'])
            if len(fetal_peak_times) > 0:
                ax.vlines(fetal_peak_times, y_min, y_max, colors='red', alpha=0.5, 
                         linestyles='--', linewidth=1, label='Fetal ref')
        
        # Plot maternal reference peaks
        if 'maternal' in self.reference_peaks and comp_idx != self.maternal_reference:
            maternal_peak_times = self._get_windowed_reference_peak_times(self.reference_peaks['maternal'])
            if len(maternal_peak_times) > 0:
                ax.vlines(maternal_peak_times, y_min, y_max, colors='blue', alpha=0.5, 
                         linestyles=':', linewidth=1, label='Maternal ref')
        
        # Restore plot formatting
        ax.set_title(f'Component {comp_idx + 1}', fontsize=9)
        ax.set_xlabel('Time [s]', fontsize=7)
        ax.set_ylabel('Amplitude', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        
        if self.xlim is not None:
            ax.set_xlim(self.xlim)
        
        # Add legend if there are reference peaks
        if ('fetal' in self.reference_peaks and comp_idx != self.fetal_reference) or \
           ('maternal' in self.reference_peaks and comp_idx != self.maternal_reference):
            ax.legend(fontsize=6, loc='upper right')
        
        # Apply tight layout to reduce margins
        fig.subplots_adjust(left=0.05, bottom=0.08, right=.98, top=.92, wspace=0, hspace=0)
        
        # Restore background color based on label
        self._update_component_appearance(comp_idx)
        
        # Redraw
        fig.canvas.draw_idle()

    def _update_component_appearance_at_position(self, comp_idx, display_pos):
        """Update visual appearance of a component based on its label at specific display position"""
        
        fig = self.component_figures[display_pos]
        ax = fig.axes[0]
        
        # Clear and redraw with new styling
        cur_label = self.labels.get(comp_idx, 'Unlabeled')
        
        # Set background color based on label
        if cur_label != 'Unlabeled':
            color_name = self.label_colors[self.label_options.index(cur_label)]
            # Convert color name to RGBA with 20% opacity (0.2 alpha)
            rgba_color = mcolors.to_rgba(color_name, alpha=0.2)
            ax.set_facecolor(rgba_color)
        else:
            ax.set_facecolor('white')

    def _update_component_appearance(self, comp_idx):
        """Update visual appearance of a component based on its label"""
        display_pos = self.comp_idx_to_display_pos[comp_idx]
        self._update_component_appearance_at_position(comp_idx, display_pos)
        
        # Redraw the specific figure
        fig = self.component_figures[display_pos]
        fig.canvas.draw_idle()

    def _update_progress(self):
        """Update the progress display"""
        labeled_count = len(self.labels)
        progress_text = f"<h3>ICA Component Labeler - Progress: {labeled_count}/{self.n_components} labeled</h3>"
        self.progress_label.value = progress_text
        # Update bottom progress label if it exists
        if hasattr(self, 'progress_label_bottom'):
            self.progress_label_bottom.value = progress_text


    def clear_all_labels(self, _b):
        """Clear all labels and references"""
        self.labels = {}
        
        # Reset all radio buttons (indexed by display position)
        for selector in self.component_selectors:
            selector.value = None
        
        # Reset all checkboxes (indexed by display position)
        for checkbox in self.fetal_checkboxes:
            checkbox.value = False
        for checkbox in self.maternal_checkboxes:
            checkbox.value = False
        
        # Clear reference data
        self.fetal_reference = None
        self.maternal_reference = None
        self.reference_peaks = {}
        
        # Clear similarity data
        self.fetal_similarities = {}
        self.maternal_similarities = {}
        self.fetal_hr_similar = set()
        self.maternal_hr_similar = set()
        
        # Update all appearances in sorted order
        for i, comp_idx in enumerate(self.component_order):
            self._update_single_plot_at_position(comp_idx, i)
        
        # Update all stats to remove similarity scores
        self._update_all_stats()
        
        self._update_progress()

    def _on_window_changed(self, change):
        """Handle window slider changes"""
        # Update window parameters from sliders
        self.window_start = self.window_start_slider.value
        self.window_length = self.window_length_slider.value
        
        # Ensure window doesn't exceed data bounds
        max_start = float(self.time_orig[-1] - self.window_length)
        if self.window_start > max_start:
            self.window_start = max_start
            self.window_start_slider.value = self.window_start
            
        # Update slider limits to prevent invalid combinations
        self.window_start_slider.max = max_start
        
        # Apply the new window
        self._apply_window_filter()
        
        # Update plots with new window
        self._update_all_plots()
        self._update_all_stats()

    def _reset_window_view(self, _button):
        """Reset window to show full data"""
        if self.time_orig is not None:
            self.window_start = float(self.time_orig[0])
            self.window_length = self.total_duration
            
            # Update slider values
            self.window_start_slider.value = self.window_start
            self.window_length_slider.value = self.window_length
            
            # Apply the reset window
            self._apply_window_filter()
            
            # Update plots
            self._update_all_plots()
            self._update_all_stats()

    def _apply_window_filter(self):
        """Apply current window settings to sources and time"""
        if self.time_orig is None:
            return
            
        # Calculate window end
        window_end = self.window_start + self.window_length
        
        # Create time mask
        mask = (self.time_orig >= self.window_start) & (self.time_orig <= window_end)
        
        # Apply mask to time and sources
        self.time = self.time_orig[mask]
        self.sources = self.sources_orig_full[mask, :]
        
        # Update xlim for plotting functions
        self.xlim = (self.window_start, window_end)

    def _get_windowed_peaks(self, comp_idx):
        """Get peak indices for a component that are valid for the current window
        
        Returns:
            windowed_peak_indices: array of indices valid for current windowed signal
        """
        peaks = self.all_peaks[comp_idx]
        
        if len(peaks) == 0 or self.time_orig is None:
            return np.array([])
        
        # Convert peak indices to time values using original time
        peak_times = self.time_orig[peaks]
        
        # Filter peaks that are within current window
        window_mask = (peak_times >= self.time[0]) & (peak_times <= self.time[-1])
        valid_peak_times = peak_times[window_mask]
        
        if len(valid_peak_times) == 0:
            return np.array([])
        
        # Find corresponding indices in windowed signal
        windowed_peak_indices = []
        for peak_time in valid_peak_times:
            # Find closest time index in windowed data
            idx = np.argmin(np.abs(self.time - peak_time))
            windowed_peak_indices.append(idx)
        
        return np.array(windowed_peak_indices)

    def _get_windowed_hr_and_sdnn(self, comp_idx):
        """Calculate HR and SDNN using peaks in current window"""
        windowed_peak_indices = self._get_windowed_peaks(comp_idx)
        
        if len(windowed_peak_indices) < 2:
            return 0.0, 0.0
        
        # Calculate intervals using windowed peak indices
        intervals = np.diff(windowed_peak_indices) / self.fs
        hr = 60.0 / np.mean(intervals)
        sdnn = np.std(intervals)
        
        return hr, sdnn

    def _get_windowed_reference_peak_times(self, ref_peaks):
        """Get reference peak times that are within the current window
        
        Args:
            ref_peaks: array of peak indices from the original full data
            
        Returns:
            array of time values for peaks within current window
        """
        if len(ref_peaks) == 0 or self.time_orig is None:
            return np.array([])
        
        # Convert peak indices to time values using original time
        peak_times = self.time_orig[ref_peaks]
        
        # Filter peaks that are within current window
        window_mask = (peak_times >= self.time[0]) & (peak_times <= self.time[-1])
        valid_peak_times = peak_times[window_mask]
        
        return valid_peak_times

    def _setup_matplotlib_controls(self):
        # Basic matplotlib button fallback (rarely used here)
        ax_prev = plt.axes([0.1, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.25, 0.02, 0.1, 0.04])
        self.btn_prev = mpl_widgets.Button(ax_prev, 'Previous')
        self.btn_next = mpl_widgets.Button(ax_next, 'Next')
        self.btn_prev.on_clicked(self.prev_component)
        self.btn_next.on_clicked(self.next_component)

    def _draw_all_components(self):
        """Draw all components in grid layout (for non-widget backend)"""
        
        for comp_idx in range(self.n_components):
            ax = self.axes[comp_idx]
            ax.clear()
            
            # Time series
            sig = self.sources[:, comp_idx]
            ax.plot(self.time, sig, 'k-', linewidth=0.5)
            ax.set_title(f'Comp {comp_idx + 1}', fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
            
            if self.xlim is not None:
                ax.set_xlim(self.xlim)
            
            # Color based on label
            cur_label = self.labels.get(comp_idx, 'Unlabeled')
            if cur_label != 'Unlabeled':
                color_name = self.label_colors[self.label_options.index(cur_label)]
                # Convert color name to RGBA with 20% opacity (0.2 alpha)
                rgba_color = mcolors.to_rgba(color_name, alpha=0.2)
                ax.set_facecolor(rgba_color)
                ax.text(0.02, 0.98, cur_label, transform=ax.transAxes, 
                       fontsize=6, va='top', ha='left',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor=color_name, alpha=0.7))

        # Tight layout and redraw
        try:
            self.fig.tight_layout(pad=0.05)
        except Exception:
            pass
        if hasattr(self.fig, 'canvas'):
            self.fig.canvas.draw_idle()

    def compute_kurtosis(self, signal):
        return float(stats.kurtosis(signal))

    def export_labels(self, _event):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix = f"_{self.save_suffix}" if self.save_suffix else ""
                
        # Always save the raw data as .npz for easy loading
        out_npz = f'ica_data_{ts}{suffix}.npz'
        if self.base_path is not None:
            out_npz = f'{self.base_path}/{out_npz}'
        #  ensure that the path exists otherwise create folder
        os.makedirs(os.path.dirname(out_npz), exist_ok=True)
        np.savez(out_npz,
                 bp_data=self.bp_data.astype(np.float32),
                 axis_mask=self.axis_mask,
                 ica=self.ica,
                 sources=self.sources_orig.astype(np.float32),
                 mixing_matrix=self.ica_orig.mixing_,
                 labels_dict=self.labels,
                 fs=self.fs,
                 component_labels=[self.labels.get(ci, 'Unlabeled') for ci in range(self.n_components)],
                 fetal_reference=self.fetal_reference,
                 maternal_reference=self.maternal_reference,
                 #all_peaks=self.all_peaks,
                 component_order=self.component_order,
                 seed=self.seed
                 )
        
        print(f'Exported complete ICA data: {out_npz}')
        print(f'Components sorted by kurtosis and SDNN')
        if self.fetal_reference is not None:
            print(f'Fetal reference component: {self.fetal_reference + 1}')
        if self.maternal_reference is not None:
            print(f'Maternal reference component: {self.maternal_reference + 1}')

    @staticmethod
    def load_exported_data(npz_file):
        """Load exported ICA data from .npz file
        
        Returns:
            dict with keys: sources, mixing_matrix, labels_dict, time, fs, component_labels,
                           fetal_reference, maternal_reference, all_peaks
        """
        data = np.load(npz_file, allow_pickle=True)
        result = {
            'sources': data['sources'],
            'mixing_matrix': data['mixing_matrix'], 
            'labels_dict': data['labels_dict'].item(),
            'fs': data['fs'].item(),
            'component_labels': data['component_labels'],
            'component_order': data['component_order'],
            'bp_data': data['bp_data'],
            'axis_mask': data['axis_mask'],
            'ica': data['ica'].item()

        }
        
        # Add reference data if available (for backward compatibility)
        if 'fetal_reference' in data:
            result['fetal_reference'] = data['fetal_reference'].item() if data['fetal_reference'].size > 0 else None
        if 'maternal_reference' in data:
            result['maternal_reference'] = data['maternal_reference'].item() if data['maternal_reference'].size > 0 else None
        if 'all_peaks' in data:
            result['all_peaks'] = data['all_peaks'].item()
            
        return result

    def show(self):
        if IPYWIDGETS_AVAILABLE:
            print("Launching ICA Component Labeler GUI...")
            # Initialize progress display
            self._update_progress()
            display(self.app)
        else:
            # Non-widget backend: use old grid layout
            self._draw_all_components()
            plt.tight_layout()
            plt.show()


