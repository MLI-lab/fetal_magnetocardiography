import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
import ipywidgets as widgets
from IPython.display import display, clear_output

from .plotting import plt_config


class JupyterCardiacAnnotator:
    """
    Jupyter-specific cardiac waveform annotation tool using ipywidgets.
    Supports multi-R annotations for RR interval and aligned controls.
    """

    def __init__(self, signal, fs=250, **kwargs):
        
        self.fs = fs
        if "heartbeats_dict" in kwargs:
            print(" === Using heartbeats_dict ===")
            self.heartbeats_dict = kwargs["heartbeats_dict"]
            self.component = kwargs["component"]
            self.plt_function = self._plt_avg_function
            self.time = self.heartbeats_dict[self.component]['M_x'][f"mean_beat"].index * 1e3
            self.signal = np.stack([self.heartbeats_dict[self.component][k][f"mean_beat"].values for k in self.heartbeats_dict[self.component].keys()], axis=1)
            self.signal_std = np.stack([self.heartbeats_dict[self.component][k][f"std_beat"].values for k in self.heartbeats_dict[self.component].keys()], axis=1)
        else:
            print("=== Using signal ===")
            self.heartbeats_dict = None
            self.component = None
            self.plt_function = self._plt_function
            self.time = np.arange(len(signal)) / fs
            self.signal = signal

        self.annotations = {}            # single-point annotations except R-peaks
        self.r_peaks = []                # list for multiple R annotations
        self.annotation_markers = {}
        self.annotation_texts = {}
        self.interval_lines = {}
        self.interval_texts = {}
        self.intervals = {}
        self.wave_labels = ['P onset', 'P peak', 'P offset', 'Q', 'R', 'S',
                            'T onset', 'T peak', 'T offset']
        self.current_label = None
        self.active_idx = len(self.time) // 2




        self.create_widgets()
        self.layout_widgets()
        self.connect_events()

    def create_widgets(self):
        # Plot output
        self.fig_output = widgets.Output()
        self.plt_function()

        # Slider
        self.position_slider = widgets.FloatSlider(
            value=self.time[self.active_idx], min=self.time[0], max=self.time[-1],
            step=(self.time[-1] - self.time[0]) / len(self.time), continuous_update=True,
            description='Position:'
        )
        self.position_slider.observe(self.on_slider_change, names='value')

        # Wave buttons
        self.wave_buttons = {}
        for label in self.wave_labels:
            btn = widgets.Button(description=label)
            btn.on_click(self.on_wave_select)
            self.wave_buttons[label] = btn

        # Action buttons
        self.mark_button = widgets.Button(description='Mark Point', button_style='success')
        self.mark_button.on_click(self.on_mark)
        self.compute_button = widgets.Button(description='Compute Intervals', button_style='info')
        self.compute_button.on_click(self.compute_intervals)
        self.reset_button = widgets.Button(description='Reset', button_style='danger')
        self.reset_button.on_click(self.reset_all)
        self.hide_maker_button = widgets.Button(description='Hide Markers', button_style='warning')
        self.hide_maker_button.on_click(self.clear_markers)
        self.status = widgets.Label('Select a wave')
        self.results_output = widgets.Output()

    def layout_widgets(self):
        # Arrange wave buttons in grid
        rows = [widgets.HBox([self.wave_buttons[l] for l in self.wave_labels[i:i+3]])
                for i in range(0, len(self.wave_labels), 3)]
        wave_panel = widgets.VBox(rows)
        action_panel = widgets.VBox([self.mark_button, self.compute_button, self.hide_maker_button, self.reset_button])
        controls = widgets.HBox([wave_panel, action_panel])

        self.app = widgets.VBox([
            self.fig_output,
            self.position_slider,
            controls,
            self.status,
            self.results_output
        ])

    def display(self):
        display(self.app)

    def connect_events(self):
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('key_press_event', self.on_key)

    def update_marker(self):
        x, y = self.time[self.active_idx], self.signal[self.active_idx]
        #self.marker.set_data([x],[y])
        if self.marker is None:
            self.marker = self.ax.axvline(x, 0, 1, color='r', linewidth=1, linestyle='--')
        self.marker.set_xdata([x, x])
        self.position_slider.unobserve(self.on_slider_change, names='value')
        self.position_slider.value = x
        self.position_slider.observe(self.on_slider_change, names='value')
        self.canvas.draw_idle()

    def on_slider_change(self, change):
        self.active_idx = np.argmin(np.abs(self.time - change['new']))
        self.update_marker()

    def on_click(self, event):
        if event.inaxes==self.ax:
            idx = np.argmin(np.abs(self.time - event.xdata))
            self.active_idx = idx
            self.update_marker()

    def on_key(self, event):
        if event.key=='left': self.active_idx=max(0,self.active_idx-1); self.update_marker()
        elif event.key=='right': self.active_idx=min(len(self.time)-1,self.active_idx+1); self.update_marker()
        elif event.key=='enter': self.on_mark(None)

    def on_wave_select(self, b):
        self.current_label = b.description
        for lbl, btn in self.wave_buttons.items(): btn.button_style = 'info' if lbl==self.current_label else ''
        self.status.value = f'Selected {self.current_label}'

    def on_mark(self, b):
        lbl = self.current_label
        if not lbl:
            self.status.value='Pick feature first'; return
        idx=self.active_idx
        # handle R2 separately
        if lbl=='R':
            self.r_peaks.append(idx)
            name=f'R'#{len(self.r_peaks)}
        else:
            self.annotations[lbl]=idx
            name=lbl
        #pt,=self.ax.plot(self.time[idx], 0,'rx', markersize=4)
        # txt=self.ax.text(self.time[idx], self.signal[idx,0]+0.05*(np.max(self.signal)-np.min(self.signal)),
        #                 name, fontsize=9, ha='center')
        #self.annotation_markers[name]=pt
        #self.annotation_texts[name]=txt
        self.canvas.draw_idle()
        self.status.value=f'Marked {name}'

    def reset_all(self, b):
        self.annotations={}
        self.r_peaks=[]
        for mk in self.annotation_markers.values(): mk.remove()
        for txt in self.annotation_texts.values(): txt.remove()
        self.annotation_markers={}; self.annotation_texts={}
        for line in self.interval_lines.values(): line.remove()
        for txt in self.interval_texts.values(): txt.remove()
        self.interval_lines=self.interval_texts={}; self.intervals={}
        with self.results_output: clear_output()
        self.status.value='Reset done'
        self.canvas.draw_idle()

    def clear_markers(self, b):
        if self.marker is None: return
        self.marker.remove()
        self.marker = None
        with self.results_output: clear_output()
        self.status.value='Hide Marker'
        self.canvas.draw_idle()



    def compute_intervals(self, b):
        ivals={}
        # standard intervals
        times={lbl:self.time[idx] for lbl,idx in self.annotations.items()}
        if 'P onset' in times and 'Q' in times: ivals['PR']=times['Q']-times['P onset']
        if 'Q' in times and 'S' in times: ivals['QRS']=times['S']-times['Q']
        if 'Q' in times and 'T offset' in times: ivals['QT']=times['T offset']-times['Q']
        if 'P onset' in times and 'P offset' in times: ivals['P']=times['P offset']-times['P onset']
        if len(self.r_peaks)>=2:
            t0,t1=np.sort([self.time[i] for i in self.r_peaks[:2]])
            ivals['RR']=t1-t0
        self.intervals=ivals
        with self.results_output:
            clear_output(); print('=== Intervals ===')
            for k,v in ivals.items(): print(f"{k}: {v:.1f} ms")
        self.status.value='Intervals computed'
        self.draw_interval_lines()

    def draw_interval_lines(self):
        colors={'PR':'green','QRS':'red','QT':'purple','RR':'black'}
        ys = {'PR':.8, 'P':.4, 'QRS':1.3, 'QT':.5, 'RR':1.1}
        for line in self.interval_lines.values(): line.remove()
        for txt in self.interval_texts.values(): txt.remove()
        self.interval_lines={}; self.interval_texts={}
        y0_=np.max(self.signal)+0.1*(np.max(self.signal)-np.min(self.signal))
        for name,dur in self.intervals.items():
            if name=='RR':
                i0,i1=self.r_peaks[0],self.r_peaks[1]
            else:
                lbls=name.split()[0]
                mapping={'PR':'P onset,Q','QRS':'Q,S','QT':'Q,T offset', 'P':'P onset,P offset'}
                a,b=mapping[lbls].split(','); i0,i1=self.annotations[a],self.annotations[b]
            x0,x1=self.time[i0],self.time[i1]
            y0 = ys[name] * y0_
            line,=self.ax.plot([x0,x1],[y0,y0], linewidth=1, color='r')
            self.ax.plot([x0,x0],[0,y0], linewidth=1, color='r', linestyle='--')
            self.ax.plot([x1,x1],[0,y0], linewidth=1, color='r', linestyle='--')
            mid=(x0+x1)/2
            txt=self.ax.text(mid,y0+0.075*(np.max(self.signal)-np.min(self.signal)),f"{name}: {dur:.1f} ms",ha='left' if name == 'QT' else 'center',va='top',fontsize=8,color='k')
            self.interval_lines[name]=line; self.interval_texts[name]=txt
        
        self.ax.set_ylim(np.min(self.signal)-0.05*(np.max(self.signal)-np.min(self.signal)),
                        np.max(list(ys.values()))*y0_+0.1*(np.max(self.signal)-np.min(self.signal)))
        self.canvas.draw_idle()

    def get_results(self):
        return {'annotations':{**{k:self.time[v] for k,v in self.annotations.items()},
                               **{f'R{i+1}':self.time[idx] for i,idx in enumerate(self.r_peaks)}},
                'intervals':self.intervals}
    
    def _plt_function(self):
        with self.fig_output:
            self.fig, self.ax = plt.subplots(figsize=(10, 5), dpi=100)
            self.line, = self.ax.plot(self.time, self.signal, 'b-', linewidth=1.5)
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Amplitude')
            self.ax.set_title('Cardiac Waveform Annotation')
            self.marker = self.ax.axvline(self.time[self.active_idx], 0,1, color='r', linewidth=1, linestyle='--')
            # self.marker, = self.ax.plot(self.time[self.active_idx], self.signal[self.active_idx],
            #                             'rx', markersize=4)
            y_range = np.max(self.signal) - np.min(self.signal)
            self.ax.set_ylim(np.min(self.signal) - 0.2*y_range, np.max(self.signal) + 0.2*y_range)
            display(self.fig.canvas)
            print("h",self.fig.canvas)
            self.canvas = self.fig.canvas

    def _plt_avg_function(self):
        with self.fig_output:
            self.fig, self.ax = plt.subplots(figsize=(7.11,2.5), dpi=250)

            xmin, xmax = self.time[0], self.time[-1]
            self.ax.hlines(0, xmin, xmax, color="k", linestyle="--")
            self.ax.plot(self.time, self.signal, label=[rf"${k.lower()}$" for k in self.heartbeats_dict[self.component].keys()], linewidth=1.5)
            
            for i, k in enumerate(self.heartbeats_dict[self.component].keys()):
                self.ax.fill_between(
                    self.time,
                    self.signal[:, i] - self.signal_std[:, i],
                    self.signal[:, i] + self.signal_std[:, i],
                    alpha=0.2,
                )

            # Calculate the magnitude
            magnitude = np.linalg.norm(self.signal, axis=1)
            self.line, = self.ax.plot(self.time, magnitude, label="Magnitude", color="k", linestyle=":")
            self.ax.set_xlabel("Time [ms]")
            self.ax.set_ylabel(r"Mean Dipole Moment [nAm$^2$]")
            self.ax.legend(ncol=4, loc="lower left")
            self.ax.grid()
            self.ax.grid(which='minor', linestyle=':', linewidth=0.5)  # Add gridlines between major ticks
            self.ax.minorticks_on()  # Enable minor ticks without adding labels
            self.marker = self.ax.axvline(self.time[self.active_idx], 0,1, color='r', linewidth=1, linestyle='--')

            #self.marker, = self.ax.plot(self.time[self.active_idx], self.signal[self.active_idx, 0],
            #                            'rx', markersize=4)
            y_range = np.max(self.signal) - np.min(self.signal)
            self.ax.set_ylim(np.min(self.signal) - 0.2*y_range, np.max(self.signal)+ 0.2*y_range)
            self.ax.set_xlim(xmin, xmax)
            display(self.fig.canvas)
            #print("h",self.fig.canvas)

            self.canvas = self.fig.canvas


def annotate_cardiac_waveform_jupyter(signal, fs=250, **kwargs):
    """
    Launch the Jupyter-compatible cardiac waveform annotation tool.
    
    Parameters:
    -----------
    signal : numpy.ndarray
        The cardiac waveform data (should contain three consecutive beats)
    fs : float
        Sampling frequency in Hz
    
    Returns:
    --------
    annotator : JupyterCardiacAnnotator
        The annotator object with results accessible via get_results()
    """
    print("=== Cardiac Waveform Annotation Tool ===")
    print("Instructions:")
    print("1. Click on a feature button (P onset, Q, R, etc.) to select it")
    print("2. Click in the Figure to place the marker, move it with the slider or arrow keys to position it at the correct location")
    print("3. Click 'Mark Selected Point' or the press Enter key to annotate that feature")
    print("4. Repeat for all features you wish to annotate")
    print("5. Click 'Compute Intervals' to calculate and display intervals")
    print("6. Use 'Reset Annotations' to start over if needed")
    print("=================================================\n")
    
    # Normalize signal if needed for better visualization
    # signal = signal - np.mean(signal)
    # signal = signal / (np.max(np.abs(signal)) * 1.2)
    
    annotator = JupyterCardiacAnnotator(signal, fs, **kwargs)
    annotator.display()
    return annotator

