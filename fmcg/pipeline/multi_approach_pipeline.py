"""
Multi-Approach fMCG Pipeline.

Orchestrates multiple signal processing approaches (ICA, Spatial Filtering,
Forward Model) on shared preprocessed data for fair comparison.
"""

import datetime
import json
import logging
import os
import pickle

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional

from fmcg.utils.plotting.report import plot_signal
from fmcg.utils.plotting.plot_utils import plot_sensor_signals
from fmcg.analysis.heartbeat_averaging import average_heartbeats
from fmcg.utils.utils import as_2d, reduced2full
from fmcg.utils import data
from fmcg.pipeline._pipeline_utils import _find_continuous_segments
from fmcg.pipeline.preprocessing import PreprocessedData, preprocess

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class ApproachResult:
    """Output of a single processing approach."""
    name: str
    domains: Dict[str, Any]             # {"field": ndarray, "dipole": ndarray, "source": ndarray}
    metadata: Dict[str, Any]            # Approach-specific extras (mixing matrix, loss, etc.)
    peak_detection_signal: Optional[np.ndarray] = None           # 1D fetal signal for peak detection
    maternal_peak_detection_signal: Optional[np.ndarray] = None  # 1D maternal signal


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class MultiApproachPipeline:
    """
    Multi-approach fMCG processing pipeline.

    Runs shared preprocessing once, then executes enabled approaches
    sequentially, followed by unified and per-method post-processing.

    Usage
    -----
    with MultiApproachPipeline(config) as pipeline:
        pipeline.run()

    Results are stored in pipeline.results (hierarchical) and saved to
    pipeline_results.pkl (legacy) and pipeline_results_v2.pkl (hierarchical).
    """

    def __init__(self, config: dict, data_loader=None):
        """
        Parameters
        ----------
        config : dict
            Full pipeline configuration (see Configuration Schema in plan).
        data_loader : optional
            Synthetic data loader. If provided, bypasses TDMS loading.
        """
        self.config = config
        self.data_loader = data_loader
        self.results: Dict[str, Any] = {}
        self._preprocessed: Optional[PreprocessedData] = None
        self._ica_orchestrator = None  # Shared between ICA and SF in concat mode
        self.output_dir: Optional[str] = None
        self.plots_dir: Optional[str] = None
        self.metrics: Dict[str, Any] = {}  # For processed_records.csv integration

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        device = self.config.get("device", "")
        if device and "cuda" in str(device):
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        return False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self):
        """Execute the full pipeline."""
        self._setup_output_dir()
        self._save_config()

        logger.info("=" * 60)
        logger.info("Multi-Approach fMCG Pipeline")
        logger.info("=" * 60)

        # 1. Shared preprocessing
        logger.info("\n[1/4] Preprocessing")
        self._preprocessed = self._load_and_preprocess()

        # 2. Approach execution (sequential)
        logger.info("\n[2/4] Approach Execution")
        approaches = self.config.get("approaches", {})

        if approaches.get("ica", {}).get("enabled"):
            logger.info("  Running ICA approach...")
            self.results["ica"] = self._run_approach(self._run_ica, self._preprocessed)

        if approaches.get("spatial_filtering", {}).get("enabled"):
            logger.info("  Running Spatial Filtering approach...")
            self.results["spatial_filtering"] = self._run_approach(self._run_spatial_filtering, self._preprocessed)

        if approaches.get("forward_model", {}).get("enabled"):
            logger.info("  Running Forward Model approach...")
            self.results["forward_model"] = self._run_approach(self._run_forward_model, self._preprocessed)

        # 3. Post-processing
        logger.info("\n[3/4] Post-Processing")
        self._post_process()

        # 4. Save
        logger.info("\n[4/4] Saving Results")
        self._save()

        logger.info("\nPipeline completed successfully.")

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _load_and_preprocess(self) -> PreprocessedData:
        """
        Load data then delegate to shared preprocess(). Computes pipeline metrics
        and optional plots from the returned PreprocessedData.
        """
        # 1. Load data
        sig_dict, time, fs, noise_dict, configs = self._load_data()
        r_sensors = configs.get("r_sensors")

        # log data info
        signal_len = len(next(iter(sig_dict.values()))) / fs
        noise_len = len(next(iter(noise_dict.values()))) / fs
        logger.info(f"  Signal: {signal_len:.1f}s  |  Noise: {noise_len:.1f}s  |  fs: {fs} Hz")

        # 2. Preprocess (shared by all approaches)
        # Build flat params dict for preprocess()
        cfg_pre = self.config["preprocessing"]
        cfg_data = self.config["data"]
        params = {
            **cfg_pre,
            "sampfrom": cfg_data.get("sampfrom", 0),
            "sampto": cfg_data.get("sampto"),
        }
        preprocessed = preprocess(sig_dict, noise_dict, fs, params, time=time, r_sensors=r_sensors)

        # Pipeline metrics (computed on full data before artifact handling)
        artifact_frac = np.mean(preprocessed.artifacts_mask) * 100
        self.metrics["artifact_fraction"] = round(float(artifact_frac), 2)

        min_seg_samples = int(cfg_pre.get("min_segment_length", 30) * preprocessed.fs)
        clean_segs = _find_continuous_segments(preprocessed.artifacts_mask, min_length=min_seg_samples)
        total_clean = sum(e - s for s, e in clean_segs) / preprocessed.fs
        self.metrics["n_clean_segments"] = len(clean_segs)
        self.metrics["total_clean_s"] = round(total_clean, 2)
        logger.info(f"  Clean segments: {len(clean_segs)}  ({total_clean:.1f}s total clean data)")

        # Optional plots (before artifact handling so artifact mask is still meaningful)
        if self.config.get("output", {}).get("save_plots"):
            clean_only = np.array(preprocessed.field_data, copy=True)
            clean_only[preprocessed.artifacts_mask] = np.nan

            plot_sensor_signals(
                preprocessed.field_data,
                time=preprocessed.time,
                xlim=[50,52],
                ylabel="Field [pT]",
                savename=os.path.join(self.output_dir, "plots/01_field_filtered.pdf"),
                artifacts_mask=preprocessed.artifacts_mask,
            )

            plot_signal(preprocessed.time,
                        preprocessed.artifacts_mask,
                        xlabel="Time [s]",
                        ylabel="Artifact",
                        color="red",
                        savename=os.path.join(self.output_dir, "plots/01_artifact_mask.pdf"))

        return preprocessed

    def _load_data(self):
        """Load raw signal and noise. Returns (sig_dict, time, fs, noise_dict, configs)."""
        cfg = self.config["data"]

        if self.data_loader is not None:
            sig_dict_raw, time, fs, noise_dict_raw = (
                self.data_loader.load_structured_patient_data_and_noise()
            )

            # Preserve sensor positions for forward-model runs when synthetic
            # data is supplied via a custom loader.
            loader_cfg = {}
            r_sensors = None
            if hasattr(self.data_loader, "params") and isinstance(self.data_loader.params, dict):
                r_sensors = self.data_loader.params.get("r_sensors")
            if r_sensors is None:
                r_sensors = self.config.get("r_sensors")
            if r_sensors is None and isinstance(self.config.get("data"), dict):
                r_sensors = self.config["data"].get("r_sensors")
            if r_sensors is not None:
                loader_cfg["r_sensors"] = np.array(r_sensors, copy=True)

            return sig_dict_raw, time, fs, noise_dict_raw, loader_cfg

        sig_dict_raw, time, fs, noise_dict_raw, configs = (
            data.load_structured_patient_data_and_noise(
                **{k: cfg[k] for k in cfg.keys() & {
                    "ds_path", "patient", "series",
                    "sig_group_names", "noise_group_names",
                    "noise_patient", "noise_series",
                }},
                files="all",
                return_configs=True,
            )
        )
        return sig_dict_raw, time, fs, noise_dict_raw, configs

    # ------------------------------------------------------------------
    # Approach execution wrapper
    # ------------------------------------------------------------------

    def _run_approach(
        self, approach_fn, preprocessed: PreprocessedData
    ) -> Optional[ApproachResult]:
        """
        Execute an approach under the configured artifact_handling mode.

        "individual" (default, matches old pipeline):
            Loop over clean segments >= min_segment_length. Call approach_fn on each
            segment independently. Combine results into full-timeline arrays (NaN in
            artifact gaps). Peak indices are naturally in original time space because
            each segment's arrays are placed at their correct position in the output.

        "concat":
            Concatenate clean segments into one continuous array. Call approach_fn
            once. Remap results back to full timeline via segment boundary map.
            Pro: ICA runs once → pre-computed annotations directly usable.

        Note on ICA sources in "individual" mode: sources are excluded from the
        combined result because each segment yields an independently-ordered set of
        components — concatenating them is meaningless. Reconstructed field arrays
        (sensor space) are combinable and are always included.
        """
        cfg_pre = self.config["preprocessing"]
        handling = cfg_pre.get("artifact_handling", "individual")
        min_seg_samples = int(cfg_pre.get("min_segment_length", 30) * preprocessed.fs)
        clean_segs = _find_continuous_segments(preprocessed.artifacts_mask, min_length=min_seg_samples)

        if not clean_segs:
            logger.warning("  No clean segments meet min_segment_length. Skipping approach.")
            return None

        T = len(preprocessed.field_data)

        if handling == "concat":
            concat_field = np.concatenate([preprocessed.field_data[s:e] for s, e in clean_segs])
            concat_time = np.concatenate([preprocessed.time[s:e] for s, e in clean_segs])
            seg_map = []  # (orig_start, concat_start, seg_len)
            c = 0
            for s, e in clean_segs:
                seg_map.append((s, c, e - s))
                c += e - s

            seg_data = PreprocessedData(
                field_data=concat_field,
                noise_maps=preprocessed.noise_maps,
                fs=preprocessed.fs,
                artifacts_mask=np.zeros(len(concat_field), dtype=bool),
                axis_mask=preprocessed.axis_mask,
                whitening_matrix=preprocessed.whitening_matrix,
                r_sensors=preprocessed.r_sensors,
                time=concat_time,
                sensor_names=preprocessed.sensor_names,
            )
            result = approach_fn(seg_data)
            return self._remap_concat_result(result, seg_map, T)

        else:  # "individual"
            seg_results = []
            for seg_start, seg_end in clean_segs:
                seg_data = PreprocessedData(
                    field_data=preprocessed.field_data[seg_start:seg_end],
                    noise_maps=preprocessed.noise_maps,
                    fs=preprocessed.fs,
                    artifacts_mask=np.zeros(seg_end - seg_start, dtype=bool),
                    axis_mask=preprocessed.axis_mask,
                    whitening_matrix=preprocessed.whitening_matrix,
                    r_sensors=preprocessed.r_sensors,
                    time=preprocessed.time[seg_start:seg_end],
                    sensor_names=preprocessed.sensor_names,
                )
                logger.info(f"    Segment [{seg_start}:{seg_end}] ({(seg_end - seg_start) / preprocessed.fs:.1f}s)")
                self._ica_orchestrator = None  # Fresh ICA per segment
                result = approach_fn(seg_data)
                seg_results.append((seg_start, seg_end, result))

            return self._combine_individual_results(seg_results, T)

    def _remap_concat_result(
        self, result: ApproachResult, seg_map: list, T: int
    ) -> ApproachResult:
        """Map concat-space arrays back to full-timeline (NaN in artifact gaps)."""
        concat_len = sum(l for _, _, l in seg_map)

        def remap_array(arr):
            out = np.full((T, *arr.shape[1:]), np.nan)
            for orig_start, concat_start, seg_len in seg_map:
                out[orig_start:orig_start + seg_len] = arr[concat_start:concat_start + seg_len]
            return out

        domains_full = {}
        for key, val in result.domains.items():
            if isinstance(val, np.ndarray) and val.shape[0] == concat_len:
                domains_full[key] = remap_array(val)
            elif isinstance(val, dict):
                domains_full[key] = {
                    k: remap_array(v) if isinstance(v, np.ndarray) and v.shape[0] == concat_len else v
                    for k, v in val.items()
                }
            else:
                domains_full[key] = val

        peak_sig_full = None
        if result.peak_detection_signal is not None:
            peak_sig_full = np.full(T, np.nan)
            for orig_start, concat_start, seg_len in seg_map:
                peak_sig_full[orig_start:orig_start + seg_len] = (
                    result.peak_detection_signal[concat_start:concat_start + seg_len]
                )

        maternal_peak_sig_full = None
        if result.maternal_peak_detection_signal is not None:
            maternal_peak_sig_full = np.full(T, np.nan)
            for orig_start, concat_start, seg_len in seg_map:
                maternal_peak_sig_full[orig_start:orig_start + seg_len] = (
                    result.maternal_peak_detection_signal[concat_start:concat_start + seg_len]
                )

        return ApproachResult(
            name=result.name,
            domains=domains_full,
            metadata=result.metadata,
            peak_detection_signal=peak_sig_full,
            maternal_peak_detection_signal=maternal_peak_sig_full,
        )

    def _combine_individual_results(
        self, seg_results: list, T: int
    ) -> ApproachResult:
        """
        Combine per-segment ApproachResults into a full-timeline result.

        ICA 'source' domain is excluded: each segment yields independently-ordered
        components that cannot be meaningfully concatenated. Reconstructed fields
        (sensor space) are combinable and always included.
        """
        EXCLUDE_DOMAINS = {"source"}

        first = seg_results[0][2]

        def alloc_and_fill(key):
            out = None
            for seg_start, seg_end, result in seg_results:
                val = result.domains.get(key)
                if val is None or not isinstance(val, np.ndarray):
                    continue
                if out is None:
                    out = np.full((T, *val.shape[1:]), np.nan)
                out[seg_start:seg_end] = val
            return out

        domains_full = {}
        for key, val in first.domains.items():
            if key in EXCLUDE_DOMAINS:
                continue
            if isinstance(val, np.ndarray):
                domains_full[key] = alloc_and_fill(key)
            else:
                domains_full[key] = val

        peak_sig_full = None
        if first.peak_detection_signal is not None:
            peak_sig_full = np.full(T, np.nan)
            for seg_start, seg_end, result in seg_results:
                if result.peak_detection_signal is not None:
                    peak_sig_full[seg_start:seg_end] = result.peak_detection_signal

        maternal_peak_sig_full = None
        if first.maternal_peak_detection_signal is not None:
            maternal_peak_sig_full = np.full(T, np.nan)
            for seg_start, seg_end, result in seg_results:
                if result.maternal_peak_detection_signal is not None:
                    maternal_peak_sig_full[seg_start:seg_end] = result.maternal_peak_detection_signal

        return ApproachResult(
            name=first.name,
            domains=domains_full,
            metadata={**seg_results[-1][2].metadata, "n_segments": len(seg_results)},
            peak_detection_signal=peak_sig_full,
            maternal_peak_detection_signal=maternal_peak_sig_full,
        )

    # ------------------------------------------------------------------
    # Approach wrappers
    # ------------------------------------------------------------------

    def _run_ica(self, preprocessed: PreprocessedData) -> ApproachResult:
        """
        ICA approach: blind source separation → component selection → ICS field reconstruction.
        Uses ICAOrchestrator for component selection (annotation / heuristic / manual).
        """
        from fmcg.pipeline.ica_orchestration import ICAOrchestrator
        from fmcg.ica.ics import ics

        cfg = self.config["approaches"]["ica"]
        field_data = preprocessed.field_data

        # Reshape to (T, n_valid_channels) for ICA
        field_2d = field_data[:, preprocessed.axis_mask == 1]  # (T, n_valid)

        methods_cfg = cfg.get("methods")
        if not isinstance(methods_cfg, list) or len(methods_cfg) == 0:
            raise ValueError(
                "approaches.ica.methods must be a non-empty list. "
                "Single-variant ICA config is no longer supported."
            )

        first_method_cfg = methods_cfg[0]
        if not isinstance(first_method_cfg, dict):
            raise ValueError(
                "approaches.ica.methods[0] must be a dict."
            )
        if first_method_cfg.get("type") is None:
            raise ValueError(
                "ICA methods must define 'type' (e.g. 'ics', 'splined_ics', 'ica')."
            )
        first_comp_sel = first_method_cfg.get("component_selection", {})
        if not isinstance(first_comp_sel, dict):
            raise ValueError("component_selection for ICA method 0 must be a dict.")

        # Use first method config to run ICA so annotation-based model loading
        # (from_annotation) is applied consistently in methods-only mode.
        self._ica_orchestrator = ICAOrchestrator({**cfg, "component_selection": {**first_comp_sel}})
        ica_result = self._ica_orchestrator.run_ica(field_2d, preprocessed.fs)
        sources = ica_result["sources"]

        domains = {"source": sources}
        methods_meta = {}
        primary_method = None
        primary_fetal_idx = None
        primary_maternal_idx = None
        used_method_ids = set()

        for i, method_cfg in enumerate(methods_cfg):
            if not isinstance(method_cfg, dict):
                raise ValueError(
                    f"approaches.ica.methods[{i}] must be a dict, got {type(method_cfg).__name__}."
                )

            base_method_id = (
                method_cfg.get("name")
                or method_cfg.get("type")
                or f"ica_method_{i}"
            )
            method_id = str(base_method_id)
            suffix = 2
            while method_id in used_method_ids:
                method_id = f"{base_method_id}_{suffix}"
                suffix += 1
            used_method_ids.add(method_id)

            method_type = method_cfg.get("type")
            if method_type is None:
                raise ValueError(
                    f"ICA method '{method_id}' must define 'type'."
                )

            comp_sel = method_cfg.get("component_selection", {})
            if not isinstance(comp_sel, dict):
                raise ValueError(
                    f"component_selection for ICA method '{method_id}' must be a dict."
                )

            if i == 0:
                method_orch = self._ica_orchestrator
            else:
                method_orch = ICAOrchestrator({**cfg, "component_selection": {**comp_sel}})
            maternal_idx_m, fetal_idx_m = method_orch.select_components(sources, preprocessed.fs)
            if not maternal_idx_m or not fetal_idx_m:
                raise ValueError(
                    "ICA component selection must provide both maternal and fetal components. "
                    f"Method={method_id} got maternal={maternal_idx_m}, fetal={fetal_idx_m}."
                )

            if method_type in ("ica", "ica_reconstruction"):
                field_fetal_m, field_maternal_m = ics(
                    ica_result["ica"],
                    field_2d,
                    sources,
                    fetal_idx_m,
                    preprocessed.fs,
                    keep_sources=True,
                    field_subtract=False,
                )
            elif method_type in ("ics", "splined_ics"):
                field_fetal_m, field_maternal_m = ics(
                    ica_result["ica"],
                    field_2d,
                    sources,
                    maternal_idx_m,
                    preprocessed.fs,
                    keep_sources=(method_type == "splined_ics"),
                    field_subtract=(method_type == "splined_ics"),
                    spline=(method_type == "splined_ics"),
                )
            else:
                raise ValueError(f"Unknown ICA method type for method {method_id}: {method_type}")

            domains[f"{method_id}_field_fetal"] = field_fetal_m
            domains[f"{method_id}_field_maternal"] = field_maternal_m

            methods_meta[method_id] = {
                "type": method_type,
                "fetal_indices": fetal_idx_m,
                "maternal_indices": maternal_idx_m,
            }

            if i == 0:
                primary_method = method_id
                primary_fetal_idx = fetal_idx_m
                primary_maternal_idx = maternal_idx_m
                domains["field_fetal"] = field_fetal_m
                domains["field_maternal"] = field_maternal_m

        peak_sig = (
            np.linalg.norm(sources[:, primary_fetal_idx], axis=1)
            if len(primary_fetal_idx) > 1
            else sources[:, primary_fetal_idx[0]]
        )
        maternal_sig = (
            np.linalg.norm(sources[:, primary_maternal_idx], axis=1)
            if len(primary_maternal_idx) > 1
            else sources[:, primary_maternal_idx[0]]
        )

        metadata = {
            "mixing_matrix": ica_result["ica"].mixing_,
            "methods": methods_meta,
            "primary_method": primary_method,
            "fetal_indices": primary_fetal_idx,
            "maternal_indices": primary_maternal_idx,
            "component_labels": {
                **{i: "fetal" for i in primary_fetal_idx},
                **{i: "maternal" for i in primary_maternal_idx},
            },
        }

        return ApproachResult(
            name="ica",
            domains=domains,
            metadata=metadata,
            peak_detection_signal=peak_sig,
            maternal_peak_detection_signal=maternal_sig,
        )

    def _run_spatial_filtering(self, preprocessed: PreprocessedData) -> ApproachResult:
        """
        Spatial Filtering approach: template-based linear projection.
        Requires ICA sources for template construction (auto-runs ICA if not cached).
        """
        from fmcg.pipeline.ica_orchestration import ICAOrchestrator
        from fmcg.spatial_filtering.spatial_filters import SpatialFilter

        cfg = self.config["approaches"]["spatial_filtering"]
        field_data = preprocessed.field_data
        field_2d = field_data[:, preprocessed.axis_mask == 1]

        # Reuse ICA orchestrator if already run (concat mode or ICA approach ran first).
        # Otherwise create one now (SF-only mode or individual-mode reset already happened).
        ica_cfg = self.config.get("approaches", {}).get("ica", {})
        if self._ica_orchestrator is None:
            self._ica_orchestrator = ICAOrchestrator(ica_cfg or {})
        ica_result = self._ica_orchestrator.run_ica(field_2d, preprocessed.fs)
        maternal_idx, fetal_idx = self._ica_orchestrator.select_components(
            ica_result["sources"], preprocessed.fs
        )
        if not maternal_idx or not fetal_idx:
            raise ValueError(
                "Spatial filtering requires both maternal and fetal ICA component indices. "
                f"Got maternal={maternal_idx}, fetal={fetal_idx}."
            )

        # Always pass both suppress and enhance with auto-detected indices.
        # SpatialFilter._validate_config() raises only when a *required* dict is missing,
        # so passing an unrequired one is silently ignored — no need to categorise methods here.
        sources = ica_result["sources"]
        results_per_method = {}
        methods_cfg = cfg.get("methods", [])
        if not methods_cfg:
            raise ValueError("spatial_filtering requires at least one configured method")

        for method_cfg in methods_cfg:
            method_type = method_cfg["type"]
            suppress_comps = method_cfg.get("suppress", {}).get("comps", maternal_idx)
            enhance_comps  = method_cfg.get("enhance", {}).get("comps", fetal_idx)

            sf = SpatialFilter(
                method=method_type,
                fs=preprocessed.fs,
                suppress={"sources": sources, "comps": suppress_comps},
                enhance ={"sources": sources, "comps": enhance_comps},
                window_size=method_cfg.get("window_size", 0.25),
                bpm_tol=method_cfg.get("bpm_tol", 5),
            )
            apply_kwargs = {k: method_cfg[k] for k in ("reg_power", "k", "explained_variance",
                                                        "threshold_power") if k in method_cfg}
            filtered = sf.apply(field_2d, **apply_kwargs)
            results_per_method[method_type] = filtered

        # Primary output: first method's filtered field
        primary_method = cfg["methods"][0]["type"]
        primary_field = results_per_method[primary_method]
        peak_sig = np.linalg.norm(primary_field, axis=1)
        maternal_peak_sig = (
            np.linalg.norm(sources[:, maternal_idx], axis=1)
            if len(maternal_idx) > 1
            else sources[:, maternal_idx[0]]
        )

        return ApproachResult(
            name="spatial_filtering",
            domains=results_per_method,  # {method_type: (T, n_valid) ndarray}
            metadata={
                "methods": list(results_per_method.keys()),
                "ica_sources": ica_result["sources"],
                "fetal_indices": fetal_idx,
                "maternal_indices": maternal_idx,
            },
            peak_detection_signal=peak_sig,
            maternal_peak_detection_signal=maternal_peak_sig,
        )

    def _run_forward_model(self, preprocessed: PreprocessedData) -> ApproachResult:
        """
        Forward Model approach: non-linear dipole fitting via InverseSolver.
        Returns dipole moments, positions, and reconstructed field.
        """
        import torch
        from fmcg.fitting.field_model import ForwardModel
        from fmcg.pipeline._pipeline_utils import (
            _create_inverse_solver,
            _initialize_solver_parameters,
            _solve_segment,
        )

        def _contiguous(value):
            if isinstance(value, np.ndarray):
                return np.ascontiguousarray(value)
            if isinstance(value, dict):
                return {k: _contiguous(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_contiguous(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_contiguous(v) for v in value)
            return value

        cfg = self.config["approaches"]["forward_model"]
        solver_cfg = _contiguous(cfg["solver"])
        num_dipoles = solver_cfg.get("num_dipoles", 2)
        if num_dipoles != 2:
            raise ValueError(
                "MultiApproachPipeline currently supports forward_model with num_dipoles=2 only. "
                f"Got num_dipoles={num_dipoles}."
            )
        field_data = np.ascontiguousarray(preprocessed.field_data)  # (T, S, 3) — solver applies axis_mask internally

        if preprocessed.r_sensors is None:
            raise ValueError("Forward model requires sensor positions (r_sensors). "
                             "Check that data loading returns r_sensors in configs.")

        # Build a config dict compatible with _pipeline_utils helpers:
        # they expect config["solver"] and config["device"]
        device = self.config.get("device", "cpu")
        compat_config = {
            "solver": solver_cfg,
            "device": device,
            "output_dir": self.output_dir or ".",
        }

        N = len(field_data)
        mdl = _create_inverse_solver(
            compat_config, N,
            np.ascontiguousarray(preprocessed.axis_mask),
            np.ascontiguousarray(preprocessed.whitening_matrix),
            np.ascontiguousarray(preprocessed.r_sensors),
        )

        # r_init shape in config: (1, num_dipoles, 3) → repeat across time
        r_init = np.ascontiguousarray(np.repeat(np.asarray(solver_cfg["r_init"]), N, axis=0))  # (N, num_dipoles, 3)
        initial_params, field_scaled = _initialize_solver_parameters(
            mdl, r_init, field_data, compat_config
        )

        m_hat, r_hat = _solve_segment(
            mdl, field_scaled, initial_params, compat_config,
            plot=self.config.get("output", {}).get("save_plots", False),
        )
        # m_hat: (T, num_dipoles, 3) [A·m²],  r_hat: (T, num_dipoles, 3) [m]

        # Reconstruct forward field from fitted dipoles → (T, S, 3)
        fm = ForwardModel(
            np.ascontiguousarray(preprocessed.r_sensors),
            device=device,
            axis_mask=np.ascontiguousarray(preprocessed.axis_mask),
        )
        r_t = torch.tensor(np.ascontiguousarray(r_hat), dtype=torch.float32, device=device)
        m_t = torch.tensor(np.ascontiguousarray(m_hat), dtype=torch.float32, device=device)
        field_reconstructed = fm.forward_linear(r_t, m_t, as_numpy=True)  # (T, S, 3)

        return ApproachResult(
            name="forward_model",
            domains={
                # Pre-split per entity so post-processing is generic (no dipole indexing needed)
                "dipole_fetal":    m_hat[:, 0, :],   # (T, 3)  [A·m²]
                "dipole_maternal": m_hat[:, 1, :],   # (T, 3)
                "position_fetal":  r_hat[:, 0, :],  # (T, 3)  [m]
                "position_maternal": r_hat[:, 1, :],
                "field": field_reconstructed,        # (T, S, 3)
            },
            metadata={
                "solver_loss": getattr(mdl, "loss_history", None),
                "num_dipoles": num_dipoles,
            },
            peak_detection_signal=np.linalg.norm(m_hat[:, 0, :], axis=-1),
            maternal_peak_detection_signal=np.linalg.norm(m_hat[:, 1, :], axis=-1),
        )

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _post_process(self):
        """
        Detect peaks and average heartbeats for each approach result.

        Two modes (configurable, both can run simultaneously):
          - per_method: each approach detects peaks from its own output signal
          - unified: ICA peaks used to average beats from ALL approach outputs
        """
        if not self.results:
            logger.warning("No approach results to post-process.")
            return

        cfg_post = self.config.get("postprocessing", {})
        preprocessed = self._preprocessed

        # Per-method post-processing
        if cfg_post.get("per_method_peaks", True):
            for name, result in self.results.items():
                if result is None:
                    continue
                logger.info(f"  Post-processing [{name}]...")
                self.results[name] = self._post_process_result(
                    result, preprocessed, suffix=name
                )

        # Add compact diagnostics for processed_records.csv style logging.
        metrics_source = None
        if self.results.get("ica") is not None:
            metrics_source = ("ica", self.results["ica"])
        else:
            for name, result in self.results.items():
                if result is not None:
                    metrics_source = (name, result)
                    break
        if metrics_source is not None:
            self.metrics["postprocessing_source"] = metrics_source[0]
            self._update_metrics_from_postprocessing(metrics_source[1])

        # Unified post-processing (use ICA peaks for all methods)
        if cfg_post.get("unified_peaks", False) and "ica" in self.results:
            logger.info("  Unified post-processing (ICA peaks)...")
            ica_peaks = self._extract_peaks_from_result(self.results["ica"])
            if ica_peaks is not None:
                self._unified_results = {}
                for name, result in self.results.items():
                    if result is None:
                        continue
                    unified = {}
                    for domain_name, domain_data in result.domains.items():
                        mean_beat, std_beat, time_axis = average_heartbeats(
                            ica_peaks, as_2d(domain_data), preprocessed.fs
                        )
                        unified[domain_name] = {
                            "mean": mean_beat,
                            "std": std_beat,
                            "time": time_axis,
                        }
                    self._unified_results[name] = unified
                    

        # Reports
        if self.config.get("output", {}).get("save_plots", True) and self.output_dir:
            from fmcg.pipeline.reports import generate_reports
            generate_reports(
                self.results,
                preprocessed,
                self.output_dir,
                self.config,
            )

    def _post_process_result(
        self,
        result: ApproachResult,
        preprocessed: PreprocessedData,
        suffix: str,
    ) -> ApproachResult:
        """
        Detect peaks (fetal + maternal) and average heartbeats for each domain.

        For each entity the best near-baseline segment is used for beat averaging.
        Results are stored in result.metadata["postprocessing"] as:
            {"fetal": {...}, "maternal": {...}}
        each containing keys: peaks, outlier_mask, segments, hr_data, beat_averages.
        """
        from fmcg.analysis.hr import detect_peaks, detect_hr_outlier, get_near_baseline_segments
        from fmcg.analysis.heartbeat_averaging import average_heartbeats, compute_segment_averages
        from fmcg.utils.utils import as_2d

        fs = preprocessed.fs
        top_k_segments = int(self.config.get("postprocessing", {}).get("top_k_segments", 3))

        def _process_entity(sig, label):
            if sig is None or (hasattr(sig, '__len__') and len(sig) == 0):
                return None
            # Skip if entirely NaN (can happen in individual mode on short segments)
            if np.all(np.isnan(sig)):
                return None

            # Keep peak detection robust at segment boundaries where artifact gaps are NaN.
            sig = np.nan_to_num(np.asarray(sig), nan=0.0)

            peaks = detect_peaks(sig, fs)
            if peaks is None or len(peaks) < 5:
                logger.warning(
                    f"  [{result.name}/{label}] Too few peaks "
                    f"({len(peaks) if peaks is not None else 0}) — skipping"
                )
                return None

            outlier = detect_hr_outlier(peaks, fs)

            try:
                segments, hr_data = get_near_baseline_segments(
                    peaks, fs, signal=sig, return_data=True, plot=False
                )
            except Exception as e:
                logger.warning(f"  [{result.name}/{label}] Segment detection failed: {e}")
                segments = []
                hr_data = {"heart_rate": 60.0 / (np.diff(peaks) / fs) if len(peaks) > 1 else np.array([])}

            # Choose peaks for averaging: best near-baseline segment, or all clean peaks
            if segments:
                _, seg_start_idx, seg_end_idx = segments[0]  # peak-array indices
                avg_peaks = peaks[seg_start_idx:seg_end_idx]
            else:
                avg_peaks = peaks[~outlier]

            # Average each ndarray domain using these peaks.
            # Include domains that are entity-specific (contain the label in the key)
            # or entity-neutral (contain neither "fetal" nor "maternal" in the key).
            # Skip raw source arrays — they combine all components and are not
            # meaningful as a single averaged trace.
            OTHER = "maternal" if label == "fetal" else "fetal"
            SKIP = {"source", "sources"}
            beat_averages = {}
            for domain_name, domain_data in result.domains.items():
                if OTHER in domain_name or domain_name in SKIP:
                    continue
                if not isinstance(domain_data, np.ndarray) or len(avg_peaks) < 3:
                    continue
                sig_2d = as_2d(domain_data)

                try:
                    mean_e, std_e, time_a = average_heartbeats(avg_peaks, sig_2d, fs)
                    entry = {"mean": mean_e, "std": std_e, "time": time_a}

                    # Additive richer metadata for diagnostics/reporting while keeping
                    # legacy keys unchanged for downstream compatibility.
                    global_mean, global_std, global_time = average_heartbeats(peaks, sig_2d, fs)
                    entry["global"] = {
                        "mean": global_mean,
                        "std": global_std,
                        "time": global_time,
                    }

                    seg_avgs = compute_segment_averages(peaks, sig_2d, fs, segments)
                    entry["segments"] = [
                        {
                            "mean": seg_avg["mean"],
                            "std": seg_avg["std"],
                            "time": seg_avg["time_axis"],
                            "beat_count": seg_avg["segment"]["count"],
                            "start_sec": seg_avg["segment"]["start_sec"],
                            "end_sec": seg_avg["segment"]["end_sec"],
                            "mean_hr": seg_avg["segment"]["mean_hr"],
                        }
                        for seg_avg in seg_avgs[:top_k_segments]
                    ]

                    beat_averages[domain_name] = entry
                except Exception as e:
                    logger.warning(f"  [{result.name}/{label}/{domain_name}] Beat avg failed: {e}")

            logger.info(
                f"  [{result.name}/{label}] {len(peaks)} peaks, "
                f"{len(segments)} near-baseline segments"
            )
            return {
                "peaks": peaks,
                "outlier_mask": outlier,
                "segments": segments,   # list of (count, start_idx, end_idx) in peak-array space
                "hr_data": hr_data,     # {"heart_rate", "baseline_hr", "bpm_threshold", ...}
                "beat_averages": beat_averages,
            }

        pp = {}
        fetal = _process_entity(result.peak_detection_signal, "fetal")
        if fetal:
            pp["fetal"] = fetal
        maternal = _process_entity(result.maternal_peak_detection_signal, "maternal")
        if maternal:
            pp["maternal"] = maternal

        result.metadata["postprocessing"] = pp if pp else None
        return result

    def _extract_peaks_from_result(self, result: ApproachResult) -> Optional[np.ndarray]:
        """Extract fetal clean peaks (outlier-filtered) from a post-processed result."""
        pp = result.metadata.get("postprocessing")
        if not pp or "fetal" not in pp:
            return None
        fetal = pp["fetal"]
        peaks = fetal.get("peaks")
        outlier = fetal.get("outlier_mask")
        if peaks is None:
            return None
        return peaks[~outlier] if outlier is not None else peaks

    def _update_metrics_from_postprocessing(self, result: ApproachResult) -> None:
        """Store key post-processing diagnostics for downstream record summaries."""
        pp = result.metadata.get("postprocessing") or {}

        for entity_name in ("fetal", "maternal"):
            entity = pp.get(entity_name)
            if not entity:
                continue

            peaks = entity.get("peaks")
            segments = entity.get("segments")
            hr = np.asarray((entity.get("hr_data") or {}).get("heart_rate", []))

            self.metrics[f"{entity_name}_peak_count"] = int(len(peaks)) if peaks is not None else 0
            self.metrics[f"{entity_name}_segment_count"] = int(len(segments)) if segments is not None else 0
            self.metrics[f"{entity_name}_mean_hr"] = float(np.nanmean(hr)) if hr.size > 0 else np.nan



    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _save(self):
        """Save both legacy (pipeline_results.pkl) and hierarchical (pipeline_results_v2.pkl) formats."""
        from fmcg.pipeline.data_formats import build_legacy_format, build_hierarchical_format

        output_cfg = self.config.get("output", {})
        if not output_cfg.get("save_data", True):
            return

        # Build hierarchical format first (full data)
        v2 = build_hierarchical_format(
            preprocessed=self._preprocessed,
            approach_results=self.results,
            unified_results=getattr(self, "_unified_results", None),
            config=self.config,
        )
        v2_path = os.path.join(self.output_dir, "pipeline_results_v2.pkl")
        with open(v2_path, "wb") as f:
            pickle.dump(v2, f)
        logger.info(f"  Saved hierarchical results → {v2_path}")

        # Build legacy format from primary approach
        primary = self.config.get("output", {}).get("primary_approach", "forward_model")
        if primary not in self.results or self.results.get(primary) is None:
            available = [name for name, res in self.results.items() if res is not None]
            raise ValueError(
                "Configured output.primary_approach is unavailable. "
                f"primary_approach={primary!r}, available={available}."
            )
        legacy = build_legacy_format(
            approach_results=self.results,
            preprocessed=self._preprocessed,
            primary_approach=primary,
            config=self.config,
        )
        legacy_path = os.path.join(self.output_dir, "pipeline_results.pkl")
        with open(legacy_path, "wb") as f:
            pickle.dump(legacy, f)
        logger.info(f"  Saved legacy results     → {legacy_path}")

        # Legacy-style segmented time-trace reports (HR, dipole moments, positions)
        if output_cfg.get("save_plots", True):
            from fmcg.pipeline.reports import save_legacy_reports
            save_legacy_reports(
                legacy.get("data_dict", {}),
                legacy.get("heartbeats_dict", {}),
                self._preprocessed.time,
                self.output_dir,
            )

    # ------------------------------------------------------------------
    # Output directory setup
    # ------------------------------------------------------------------

    def _setup_output_dir(self):
        """Create timestamped output directory."""
        output_cfg = self.config.get("output", {})
        base_dir = output_cfg.get("output_dir", "output")

        cfg_data = self.config.get("data", {})
        patient = cfg_data.get("patient", "unknown")
        series = cfg_data.get("series", "")
        sig_groups = cfg_data.get("sig_group_names", [""])
        sampfrom = cfg_data.get("sampfrom", 0)
        sampto = cfg_data.get("sampto", "end")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sig_label = "_".join(sig_groups) if isinstance(sig_groups, list) else str(sig_groups)
        dirname = f"{patient}_{series}_{sig_label}_{sampfrom}s-{sampto}s_{timestamp}"

        self.output_dir = os.path.join(base_dir, dirname)
        os.makedirs(self.output_dir, exist_ok=True)
        self.plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

    def _save_config(self):
        """Save config as JSON (numpy arrays excluded)."""
        config_path = os.path.join(self.output_dir, "config.json")
        try:
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save config: {e}")

