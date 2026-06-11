"""
ICA Orchestration for the multi-approach fMCG pipeline.

Wraps ICA execution and component selection. Shared between the ICA approach
and the Spatial Filtering approach (which needs ICA sources for template
construction) so that ICA is computed only once per data segment.

Caching behaviour
-----------------
ICAOrchestrator caches the ICA result on first call to run_ica(). In concat
mode the same instance is reused for both the ICA approach and SF approach
(stored on the pipeline as self._ica_orchestrator), so ICA runs once.
In individual mode the pipeline resets self._ica_orchestrator before each
segment call, giving each segment a fresh ICA run.
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np

from fmcg.ica.ica import apply_ICA, select_ica_components

logger = logging.getLogger(__name__)


class ICAOrchestrator:
    """
    Wraps ICA execution and component selection.

    Parameters
    ----------
    config : dict
        ICA approach config block. Relevant keys:
          n_components (int, optional)
          seed (int, default 42)
          component_selection (dict):
            method : "clustering_heuristic" | "from_annotation" | "manual_indices"
            n_clusters (int, default 3) — clustering_heuristic only
            annotation_file (str) — path to .npz from ICAComponentLabeler
            maternal_indices (list[int]) — manual_indices only
            fetal_indices (list[int]) — manual_indices only
    """

    def __init__(self, config: dict):
        self.config = config
        self._ica_result: Optional[dict] = None  # {"sources": ndarray, "ica": FastICA}
        self._annotation_data: Optional[dict] = None
        self._annotation_file: Optional[str] = None
        self._annotation_file_mtime: Optional[float] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_ica(self, field_2d: np.ndarray, fs: int) -> dict:
        """
        Run ICA on field_2d, returning {"sources": ..., "ica": ...}.

        If method is "from_annotation" and an annotation_file is configured,
        sources are loaded directly from the .npz file (ICA is not re-run).
        Result is cached — subsequent calls return the cache unchanged.

        Parameters
        ----------
        field_2d : np.ndarray
            (T, n_valid_channels) array.
        fs : int
            Sampling rate [Hz].
        """
        if self._ica_result is not None:
            cached_len = self._ica_result["sources"].shape[0]
            if field_2d.shape[0] != cached_len:
                logger.warning(
                    "  ICA cache length mismatch (cached=%d, current=%d). Invalidating cache.",
                    cached_len,
                    field_2d.shape[0],
                )
                self._ica_result = None
            else:
                return self._ica_result

        sel_cfg = self.config.get("component_selection", {})
        method = sel_cfg.get("method", "clustering_heuristic")

        # from_annotation: load the trained ICA model from the .npz, then apply
        # transform() on the current field_2d.  We never use the precomputed sources
        # stored in the .npz because they may have a different length (annotation was
        # made on the full recording; pipeline may process a shorter window or
        # individual segments of different lengths).
        if method == "from_annotation":
            annotation_file = sel_cfg.get("annotation_file")
            if annotation_file:
                logger.info(f"  ICA: loading model from annotation file, transforming current segment")
                npz_data = self._load_from_annotation(annotation_file)
                ica = npz_data["ica"]
                sources = ica.transform(field_2d)
                self._ica_result = {"sources": sources, "ica": ica}
                return self._ica_result
            logger.warning(
                "  ICA: method=from_annotation but no annotation_file — falling back to FastICA"
            )

        n_comp = self.config.get("n_components")
        seed = self.config.get("seed", 42)
        logger.info(f"  ICA: running FastICA (n_components={n_comp}, seed={seed})")
        sources, ica = apply_ICA(field_2d, n_comp=n_comp, seed=seed)
        self._ica_result = {"sources": sources, "ica": ica}
        return self._ica_result

    def select_components(
        self, sources: np.ndarray, fs: int
    ) -> Tuple[List[int], List[int]]:
        """
        Select maternal and fetal component indices.

        Returns
        -------
        (maternal_indices, fetal_indices) — both lists of ints.
        """
        sel_cfg = self.config.get("component_selection", {})
        method = sel_cfg.get("method", "clustering_heuristic")

        if method == "from_annotation":
            return self._select_from_annotation(sel_cfg)
        elif method == "clustering_heuristic":
            return self._select_clustering(sources, fs, sel_cfg)
        elif method == "manual_indices":
            return self._select_manual(sel_cfg)
        else:
            raise ValueError(f"Unknown component selection method: {method!r}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_from_annotation(self, annotation_file: str) -> dict:
        mtime = None
        try:
            mtime = os.path.getmtime(annotation_file)
        except OSError:
            # File-not-found and permissions are handled by loader with a clearer error.
            pass

        if (
            self._annotation_data is not None
            and self._annotation_file == annotation_file
            and self._annotation_file_mtime == mtime
        ):
            return self._annotation_data

        from fmcg.ica.ica_component_labeler import ICAComponentLabeler
        npz = ICAComponentLabeler.load_exported_data(annotation_file)
        self._annotation_data = {
            "sources": npz.get("sources"),
            "ica": npz.get("ica"),
            "labels_dict": npz.get("labels_dict"),
            "component_labels": npz.get("component_labels"),
        }
        self._annotation_file = annotation_file
        self._annotation_file_mtime = mtime
        return self._annotation_data

    def _select_from_annotation(self, sel_cfg: dict) -> Tuple[List[int], List[int]]:
        annotation_file = sel_cfg.get("annotation_file")
        if not annotation_file:
            raise ValueError("from_annotation requires annotation_file in component_selection config")

        npz = self._load_from_annotation(annotation_file)
        labels_dict = npz.get("labels_dict")
        if labels_dict is None:
            labels = np.asarray(npz.get("component_labels", [])).astype(str)
            labels_dict = {i: lbl for i, lbl in enumerate(labels)}

        maternal_idx = [
            int(i) for i, lbl in labels_dict.items()
            if str(lbl).strip().lower() == "maternal"
        ]
        fetal_idx = [
            int(i) for i, lbl in labels_dict.items()
            if str(lbl).strip().lower() == "fetal"
        ]
        if not maternal_idx or not fetal_idx:
            raise ValueError(
                "Annotation-based component selection requires both maternal and fetal labels. "
                f"Found maternal={maternal_idx}, fetal={fetal_idx}."
            )
        logger.info(f"  Component selection (annotation): maternal={maternal_idx}, fetal={fetal_idx}")
        return maternal_idx, fetal_idx

    def _select_clustering(
        self, sources: np.ndarray, fs: int, sel_cfg: dict
    ) -> Tuple[List[int], List[int]]:
        n_comp = sel_cfg.get("n_components", self.config.get("n_components", 5))
        n_clusters = sel_cfg.get("n_clusters", 3)
        maternal, fetal, _ = select_ica_components(
            sources,
            fs,
            method="clustering_heuristic",
            clustering_args={"n_components": n_comp, "n_clusters": n_clusters},
        )
        if maternal is None or fetal is None:
            raise ValueError(
                "Clustering heuristic failed to identify both maternal and fetal components. "
                f"Got maternal={maternal}, fetal={fetal}."
            )
        logger.info(f"  Component selection (clustering): maternal=[{maternal}], fetal=[{fetal}]")
        return [maternal], [fetal]

    def _select_manual(self, sel_cfg: dict) -> Tuple[List[int], List[int]]:
        maternal_idx = list(sel_cfg.get("maternal_indices", []))
        fetal_idx = list(sel_cfg.get("fetal_indices", []))
        if not maternal_idx or not fetal_idx:
            raise ValueError(
                "manual_indices requires both maternal_indices and fetal_indices "
                "in component_selection config"
            )
        logger.info(f"  Component selection (manual): maternal={maternal_idx}, fetal={fetal_idx}")
        return maternal_idx, fetal_idx
