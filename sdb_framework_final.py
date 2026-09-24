#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
SDB Framework — Professional, leakage-safe Satellite-Derived Bathymetry
========================================================================

A single, coherent implementation for:

* field-data loading and raster sampling;
* fit-on-training-only spectral feature engineering;
* physics-based, ensemble and optional deep-learning models;
* spatial holdout and small-sample cross-validation;
* optional Bayesian hyper-parameter optimization;
* bootstrap confidence intervals, sensitivity and uncertainty analysis;
* tiled GeoTIFF inference and uncertainty-raster generation;
* model-complexity comparison, model serialization and CLI execution;
* RSE-complete sensitivity figures (Palatino Linotype, 10 pt) in ``plots/``.

The entire framework is tuned for shallow-water field data with a maximum
depth of 3 m: the default depth bins, the prediction ceiling and the
depth-dependent error model all follow this 3 m design envelope.

The module deliberately keeps optional dependencies lazy.  Importing this
file does not require rasterio, Optuna, XGBoost, LightGBM, CatBoost, PyTorch
or matplotlib; the corresponding feature raises a clear installation error
only when it is actually requested.

Recommended Python: 3.10+

Example
-------

    python sdb_framework_final.py \
        --mode full \
        --excel-path "field_points.xlsx" \
        --raster-path "sentinel2.tif" \
        --output-dir "sdb_outputs" \
        --models catboost lightgbm extra_trees random_forest lyzenga

For deployment on a new scene:

    python sdb_framework_final.py \
        --mode predict \
        --model-path "sdb_outputs/models/sdb_catboost_final.joblib" \
        --input "new_scene.tif" \
        --output "new_scene_bathymetry.tif"
"""

from __future__ import annotations

# =============================================================================
# Standard library
# =============================================================================

import argparse
import hashlib
import itertools
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

# =============================================================================
# Third-party core
# =============================================================================

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

try:  # Optional statistical test used by the complexity ladder.
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover - depends on the SciPy build.
    wilcoxon = None

try:  # Optional geospatial dependency.
    import rasterio  # type: ignore
    from rasterio.windows import Window  # type: ignore
    from rasterio.warp import transform as rio_transform  # type: ignore
except Exception:  # pragma: no cover - exercised only without rasterio.
    rasterio = None
    Window = Any  # type: ignore
    rio_transform = None

__version__ = "3.0.0-professional"

__all__ = [
    "SDBConfig",
    "RasterSampler",
    "FeatureEngineer",
    "BaseSDBModel",
    "StumpfBandRatioModel",
    "LyzengaModel",
    "CaballeroStumpfModel",
    "ClusterBasedRegression",
    "RandomForestSDB",
    "ExtraTreesSDB",
    "SVR_SDB",
    "MLP_SDB",
    "XGBoost_SDB",
    "LightGBM_SDB",
    "CatBoost_SDB",
    "ModelFactory",
    "HPOResult",
    "BayesianOptimizer",
    "SpatialValidator",
    "ModelEvaluator",
    "BootstrapEvaluator",
    "SensitivityAnalyzer",
    "UncertaintyQuantifier",
    "ErrorPropagator",
    "RasterInferenceEngine",
    "SDBExplainer",
    "RSEPublicationPlotter",
    "GMMSoftMixtureSDB",
    "ExpertSpec",
    "HeterogeneousMoESDB",
    "MixtureOfExpertsSDB",
    "StructuralUncertaintyDecomposer",
    "ModelTier",
    "ModelRegistryEntry",
    "MODEL_REGISTRY_METADATA",
    "ModelRegistry",
    "ModelComplexityLadder",
    "SDBOperationalPipeline",
    "run_sdb_for_files",
    "load_deployment_bundle",
    "predict_new_scene",
    "set_global_seed",
    "detect_gpu",
    "detect_torch_available",
    "setup_logger",
    "get_adaptive_depth_bins",
    "safe_pearson_corr",
]

# =============================================================================
# Infrastructure and utilities
# =============================================================================

def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and (when installed) PyTorch.

    The function is intentionally explicit and does not silently change the
    global seed anywhere else in the module.  It is called once at pipeline
    construction and by the optional neural model before training.
    """
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be in [0, 4294967295]")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # These flags improve reproducibility.  They may reduce GPU
            # throughput, which is preferable for a scientific workflow.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def detect_torch_available() -> bool:
    """Return whether PyTorch can be imported."""
    try:
        import torch  # noqa: F401  # type: ignore
        return True
    except Exception:
        return False


def detect_gpu() -> bool:
    """Detect a usable NVIDIA GPU without making CUDA a hard dependency."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=3,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def setup_logger(
    name: str = "SDB_Framework",
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    use_rotating: bool = False,
) -> logging.Logger:
    """Create an idempotent console/file logger.

    Existing handlers belonging to this logger are replaced, but propagation
    is disabled so importing the module cannot duplicate messages through the
    root logger.
    """
    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(level)
    logger_obj.propagate = False
    for handler in list(logger_obj.handlers):
        logger_obj.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger_obj.addHandler(console)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        if use_rotating:
            from logging.handlers import RotatingFileHandler
            file_handler: logging.Handler = RotatingFileHandler(
                path,
                maxBytes=10 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
        else:
            file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger_obj.addHandler(file_handler)
    return logger_obj


logger = setup_logger()


def _require_rasterio() -> Any:
    """Return rasterio or raise a precise optional-dependency error."""
    if rasterio is None:
        raise ImportError(
            "Raster operations require rasterio. Install it with "
            "`pip install rasterio` (or conda-forge on Windows)."
        )
    return rasterio


def _jsonable(value: Any) -> Any:
    """Recursively convert NumPy/dataclass/path values to JSON-safe values."""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _finite_pair(
    y_true: Sequence[float], y_pred: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return aligned finite arrays and reject empty input."""
    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if true.size != pred.size:
        raise ValueError("y_true and y_pred must have the same length")
    mask = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[mask], pred[mask]
    if true.size == 0:
        raise ValueError("No finite paired observations")
    return true, pred


def safe_pearson_corr(a: Sequence[float], b: Sequence[float]) -> float:
    """NaN-safe Pearson correlation without zero-variance warnings."""
    x, y = _finite_pair(a, b)
    if x.size < 2:
        return float("nan")
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denom <= np.finfo(float).eps:
        return float("nan")
    corr = float(np.sum(x_centered * y_centered) / denom)
    return corr if np.isfinite(corr) else float("nan")


def get_adaptive_depth_bins(
    y: Sequence[float], user_bins: Optional[Sequence[float]] = None
) -> List[float]:
    """Return validated depth-bin edges that always cover the data range.

    The original implementation could omit the maximum depth, fail for a
    one-element user bin list, or return duplicate upper edges.  This version
    normalizes all of those cases and uses a tiny upper margin so the largest
    observation is included in the final interval.  The framework-wide design
    ceiling is 3 m: whenever the data range is within 0-3 m the finest 0.5 m
    binning is applied automatically.
    """
    values = np.asarray(y, dtype=np.float64).ravel()
    values = values[np.isfinite(values) & (values >= 0)]
    max_depth = float(np.max(values)) if values.size else 0.0
    if user_bins is not None:
        edges = sorted(
            {
                float(v)
                for v in user_bins
                if np.isfinite(v) and float(v) >= 0
            }
        )
        if len(edges) < 2:
            edges = [0.0, max(1.0, max_depth + 0.1)]
        if edges[0] > 0:
            edges.insert(0, 0.0)
        if edges[-1] <= max_depth:
            edges.append(float(np.nextafter(max_depth, np.inf)))
        return edges
    if max_depth <= 3.0:
        base = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    elif max_depth <= 6.0:
        base = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    elif max_depth <= 12.0:
        base = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    else:
        base = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]
    if max_depth >= base[-1]:
        base.append(float(np.nextafter(max_depth, np.inf)))
    else:
        base[-1] = max(base[-1], float(np.nextafter(max_depth, np.inf)))
    # Remove accidental duplicates while preserving order.
    result: List[float] = []
    for edge in base:
        if not result or edge > result[-1]:
            result.append(float(edge))
    return result


def _safe_stumpf_ratio(
    ri: np.ndarray,
    rj: np.ndarray,
    n_const: float = 1000.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Numerically stable Stumpf log-band ratio."""
    ri = np.asarray(ri, dtype=np.float64)
    rj = np.asarray(rj, dtype=np.float64)
    log_i = np.log(np.clip(n_const * ri, eps, None))
    log_j = np.log(np.clip(n_const * rj, eps, None))
    safe_denominator = np.where(
        np.abs(log_j) < eps,
        np.where(log_j < 0.0, -eps, eps),
        log_j,
    )
    return log_i / safe_denominator


def _safe_normalized_difference(
    a: np.ndarray, b: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    denominator = a + b
    safe_denominator = np.where(
        np.abs(denominator) < eps,
        np.where(denominator < 0.0, -eps, eps),
        denominator,
    )
    return (a - b) / safe_denominator

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SDBConfig:
    """Single source of truth for paths, schema, physics and computation.

    This framework is designed for shallow-water field data with a maximum
    depth of 3 m.  The default depth bins (0.5 m resolution up to 3 m), the
    prediction ceiling (``prediction_max_depth=3.0``) and the depth-dependent
    error model all implement that 3 m design envelope.
    """

    # I/O. These are placeholders; production runs should pass explicit paths
    # or a YAML/JSON configuration file.
    excel_path: Path = Path(
        r"E:\PSEEZ\SDB_Article\Data_CTD_N\All\SDBNew Latlong.xlsx"
    )
    raster_path: Path = Path(r"G:\NC\20220213_SDB_5bands.tif")
    output_dir: Path = Path(r"E:\PSEEZ\SDB_Article\Final_SDB_Art\0213Final")

    # Field-data schema. RasterSampler also recognizes common aliases such as
    # Long/Lat and Lon/Lat when these canonical names are not present.
    id_col: str = "En_Name"
    lon_col: str = "Longitude"
    lat_col: str = "Latitude"
    depth_col: str = "Depth"

    # Reproducibility and parallelism.
    random_seed: int = 42
    use_gpu: bool = field(default_factory=detect_gpu)
    n_jobs: int = -1
    hpo_n_jobs: int = 1

    # Sensor/physics.
    band_order: Tuple[str, ...] = ("B2", "B3", "B4", "B8", "B11")
    points_crs: str = "EPSG:4326"
    n_const: float = 1000.0
    deep_water_percentile: float = 1.0
    eps: float = 1e-6
    reflectance_scale: Optional[float] = None

    # Uncertainty and bootstrap.
    mc_realizations: int = 500
    n_bootstrap_ci: int = 2000
    reflectance_relative_uncertainty: float = 0.05
    reflectance_absolute_uncertainty: float = 0.0005

    # Small-sample validation.
    cv_strategy: str = "spatial_loo"
    n_repeats: int = 10
    n_trials: int = 50
    spatial_cv_folds: int = 5
    n_spatial_blocks: int = 20
    holdout_fraction: float = 0.20

    # Raster inference and depth uncertainty.  The whole framework assumes a
    # 3 m maximum depth, hence the 0.5 m depth bins and the 3 m clip ceiling.
    depth_bins: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    spatial_correlation_length_px: float = 5.0
    tile_size: int = 1024
    nodata_value: float = -9999.0
    zero_is_nodata: bool = True
    prediction_min_depth: Optional[float] = 0.0
    prediction_max_depth: Optional[float] = 3.0
    min_valid_samples: int = 10

    _config_path: Optional[Path] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.excel_path = Path(self.excel_path)
        self.raster_path = Path(self.raster_path)
        self.output_dir = Path(self.output_dir)
        self.band_order = tuple(str(b) for b in self.band_order)
        self.depth_bins = tuple(float(v) for v in self.depth_bins)
        self.cv_strategy = str(self.cv_strategy).strip().lower()
        if len(self.band_order) == 0:
            raise ValueError("band_order cannot be empty")
        if len(set(self.band_order)) != len(self.band_order):
            raise ValueError("band_order contains duplicate band names")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if not 0.0 < self.holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must be in (0, 1)")
        if self.spatial_cv_folds < 2:
            raise ValueError("spatial_cv_folds must be >= 2")
        if self.n_spatial_blocks < 2:
            raise ValueError("n_spatial_blocks must be >= 2")
        if self.n_repeats < 1:
            raise ValueError("n_repeats must be >= 1")
        if self.n_bootstrap_ci < 100:
            raise ValueError("n_bootstrap_ci must be >= 100")
        if self.mc_realizations < 1:
            raise ValueError("mc_realizations must be >= 1")
        if self.tile_size < 1:
            raise ValueError("tile_size must be positive")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        if self.n_const <= 0.0:
            raise ValueError("n_const must be positive")
        if not 0.0 <= self.deep_water_percentile <= 100.0:
            raise ValueError("deep_water_percentile must be in [0, 100]")
        if self.reflectance_scale is not None and self.reflectance_scale <= 0:
            raise ValueError("reflectance_scale must be positive")
        if self.cv_strategy not in {
            "loo",
            "spatial_loo",
            "repeated_spatial_cv",
            "kfold",
        }:
            raise ValueError(
                "cv_strategy must be one of: loo, spatial_loo, "
                "repeated_spatial_cv, kfold"
            )
        if self.prediction_min_depth is not None and self.prediction_max_depth is not None:
            if self.prediction_max_depth < self.prediction_min_depth:
                raise ValueError("prediction_max_depth must be >= prediction_min_depth")
        # The framework is explicitly tuned for a 3 m shallow-water ceiling;
        # a larger ceiling is allowed but reported so the analyst is aware.
        if (
            self.prediction_max_depth is not None
            and self.prediction_max_depth > 3.0 + 1e-9
        ):
            logger.warning(
                "prediction_max_depth=%.2f m exceeds the 3 m design ceiling "
                "of this shallow-water framework; predictions are clipped to "
                "the configured ceiling regardless.",
                self.prediction_max_depth,
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        set_global_seed(self.random_seed)
        logger.info(
            "SDBConfig | output=%s | seed=%d | GPU=%s | cv=%s | repeats=%d | bootstrap=%d",
            self.output_dir.resolve(),
            self.random_seed,
            "Enabled" if self.use_gpu else "Disabled",
            self.cv_strategy.upper(),
            self.n_repeats,
            self.n_bootstrap_ci,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("_config_path", None)
        return _jsonable(data)

    def save(self, filename: str = "config.yaml") -> Path:
        """Save YAML when PyYAML is available, otherwise JSON."""
        target = self.output_dir / filename
        try:
            import yaml  # type: ignore
            with target.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    self.to_dict(),
                    handle,
                    sort_keys=False,
                    allow_unicode=True,
                )
            logger.info("Configuration saved: %s", target)
            return target
        except ImportError:
            fallback = target.with_suffix(".json")
            with fallback.open("w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            logger.warning("PyYAML is not installed; configuration saved as %s", fallback)
            return fallback

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SDBConfig":
        """Load a YAML or JSON configuration, ignoring unknown keys."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise ImportError("YAML loading requires PyYAML") from exc
            with config_path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        elif config_path.suffix.lower() == ".json":
            with config_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        else:
            raise ValueError("Configuration must have .yaml, .yml or .json extension")
        if not isinstance(data, Mapping):
            raise ValueError("Configuration root must be a mapping/object")
        valid_keys = {
            item.name
            for item in fields(cls)
            if item.init and not item.name.startswith("_")
        }
        filtered = {key: value for key, value in data.items() if key in valid_keys}
        obj = cls(**filtered)
        obj._config_path = config_path
        logger.info("Configuration loaded from %s", config_path)
        return obj

    from_file = from_yaml

# =============================================================================
# Raster sampling and field-data ingestion
# =============================================================================

class RasterSampler:
    """Extract multi-band reflectance at field locations with fixed scaling."""

    _COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
        "id": ("id", "name", "station", "station_id", "sample", "en_name"),
        "lon": ("longitude", "long", "lon", "x", "easting"),
        "lat": ("latitude", "lat", "y", "northing"),
        "depth": ("depth", "depth_m", "water_depth", "z"),
    }

    def __init__(
        self,
        raster_path: Union[str, Path],
        band_order: Sequence[str],
        points_crs: str = "EPSG:4326",
    ) -> None:
        _require_rasterio()
        self.raster_path = Path(raster_path)
        if not self.raster_path.exists():
            raise FileNotFoundError(f"Raster not found: {self.raster_path}")
        self.band_order = list(band_order)
        self.points_crs = str(points_crs)
        self.reflectance_scale_: Optional[float] = None
        self.raster_crs: Optional[str] = None
        self.nodata: Optional[float] = None
        self.width: int = 0
        self.height: int = 0
        self.count: int = 0
        self.bounds: Any = None
        with rasterio.open(self.raster_path) as src:  # type: ignore[union-attr]
            self.raster_crs = str(src.crs) if src.crs else None
            self.nodata = src.nodata
            self.width = int(src.width)
            self.height = int(src.height)
            self.count = int(src.count)
            self.bounds = src.bounds
            if self.count < len(self.band_order):
                raise ValueError(
                    f"Raster contains {self.count} bands, but "
                    f"{len(self.band_order)} were requested"
                )
        logger.info(
            "RasterSampler | file=%s | bands=%s | raster CRS=%s | point CRS=%s",
            self.raster_path.name,
            self.band_order,
            self.raster_crs,
            self.points_crs,
        )

    @staticmethod
    def _normalize_column(name: Any) -> str:
        text = str(name).strip().lower()
        return "".join(ch for ch in text if ch.isalnum())

    def _resolve_column(
        self,
        columns: Sequence[Any],
        configured: Optional[str],
        semantic: str,
        required: bool = True,
    ) -> Optional[Any]:
        by_normalized = {
            self._normalize_column(column): column for column in columns
        }
        candidates: List[str] = []
        if configured:
            candidates.append(str(configured))
        candidates.extend(self._COLUMN_ALIASES.get(semantic, ()))
        for candidate in candidates:
            key = self._normalize_column(candidate)
            if key in by_normalized:
                return by_normalized[key]
        if required:
            raise KeyError(
                f"Could not resolve {semantic!r} column. Configured name={configured!r}; "
                f"available columns={list(columns)}"
            )
        return None

    def read_field_data(
        self,
        excel_path: Union[str, Path],
        config: Optional[SDBConfig] = None,
    ) -> pd.DataFrame:
        """Read CSV/Excel field data and return canonical ID/Lon/Lat/Depth."""
        path = Path(excel_path)
        if not path.exists():
            raise FileNotFoundError(f"Field-data file not found: {path}")
        if path.suffix.lower() in {".csv", ".txt"}:
            frame = pd.read_csv(path)
        else:
            try:
                frame = pd.read_excel(path, engine="openpyxl")
            except ImportError:
                frame = pd.read_excel(path)
        if frame.empty:
            raise ValueError("Field-data file is empty")
        frame.columns = [str(column).strip() for column in frame.columns]
        id_col = self._resolve_column(
            frame.columns,
            config.id_col if config else None,
            "id",
            required=False,
        )
        lon_col = self._resolve_column(
            frame.columns,
            config.lon_col if config else None,
            "lon",
        )
        lat_col = self._resolve_column(
            frame.columns,
            config.lat_col if config else None,
            "lat",
        )
        depth_col = self._resolve_column(
            frame.columns,
            config.depth_col if config else None,
            "depth",
        )
        result = pd.DataFrame(
            {
                "ID": (
                    frame[id_col].astype(str)
                    if id_col is not None
                    else pd.Series(
                        [f"P{i + 1:04d}" for i in range(len(frame))],
                        index=frame.index,
                    )
                ),
                "Lon": pd.to_numeric(frame[lon_col], errors="coerce"),
                "Lat": pd.to_numeric(frame[lat_col], errors="coerce"),
                "Depth": pd.to_numeric(frame[depth_col], errors="coerce"),
            }
        )
        initial_count = len(result)
        result = result.replace([np.inf, -np.inf], np.nan).dropna()
        result = result[(result["Depth"] > 0.0)]
        result = result.drop_duplicates(subset=["Lon", "Lat", "Depth"])
        result = result.reset_index(drop=True)
        if result.empty:
            raise ValueError("No valid field observations remain after quality control")
        max_depth = float(result["Depth"].max())
        if max_depth > 3.0 + 1e-9:
            logger.warning(
                "Field data contain depths up to %.2f m, above the 3 m design "
                "ceiling of this shallow-water framework.",
                max_depth,
            )
        logger.info(
            "Field QC: %d -> %d retained (%.1f%%) | max depth=%.3f m",
            initial_count,
            len(result),
            100.0 * len(result) / max(initial_count, 1),
            max_depth,
        )
        return result

    @staticmethod
    def _auto_scale(raw: np.ndarray) -> float:
        finite_positive = raw[np.isfinite(raw) & (raw > 0.0)]
        if finite_positive.size == 0:
            raise ValueError("Raster samples contain no finite positive values")
        p99 = float(np.percentile(finite_positive, 99.0))
        # Sentinel-style integer reflectance is normally 0-10000; surface
        # reflectance already in 0-1 is left unchanged.
        scale = 10000.0 if p99 > 1.5 else 1.0
        logger.info("Reflectance auto-scale | p99=%.6f | divisor=%.0f", p99, scale)
        return scale

    def sample_at_points(
        self,
        frame: pd.DataFrame,
        scale: Union[float, str] = "auto",
        nodata_threshold: float = 0.0,
    ) -> pd.DataFrame:
        """Sample all requested bands and apply one global reflectance scale."""
        _require_rasterio()
        required_columns = {"Lon", "Lat"}
        missing = required_columns.difference(frame.columns)
        if missing:
            raise KeyError(f"Point frame is missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError("Cannot sample an empty point frame")
        with rasterio.open(self.raster_path) as src:  # type: ignore[union-attr]
            lons = pd.to_numeric(frame["Lon"], errors="coerce").to_numpy()
            lats = pd.to_numeric(frame["Lat"], errors="coerce").to_numpy()
            finite_coordinates = np.isfinite(lons) & np.isfinite(lats)
            if src.crs and self.points_crs and str(src.crs).upper() != self.points_crs.upper():
                if rio_transform is None:
                    raise RuntimeError("rasterio.warp.transform is unavailable")
                xs, ys = rio_transform(
                    self.points_crs,
                    src.crs,
                    lons.tolist(),
                    lats.tolist(),
                )
            else:
                xs, ys = lons.tolist(), lats.tolist()
            coordinates = list(zip(xs, ys))
            raw = np.full(
                (len(coordinates), len(self.band_order)),
                np.nan,
                dtype=np.float64,
            )
            indexes = list(range(1, len(self.band_order) + 1))
            for row_index, values in enumerate(
                src.sample(coordinates, indexes=indexes, masked=True)
            ):
                raw[row_index] = np.ma.asarray(values).filled(np.nan)
            bounds = src.bounds
            inside = np.array(
                [
                    bounds.left <= x <= bounds.right
                    and bounds.bottom <= y <= bounds.top
                    for x, y in coordinates
                ],
                dtype=bool,
            )
            valid = finite_coordinates & inside & np.all(np.isfinite(raw), axis=1)
            valid &= np.all(raw > nodata_threshold, axis=1)
            if src.nodata is not None and np.isfinite(src.nodata):
                valid &= ~np.any(np.isclose(raw, src.nodata), axis=1)
        if isinstance(scale, str):
            if scale.strip().lower() != "auto":
                raise ValueError("scale must be a positive number or 'auto'")
            divisor = self._auto_scale(raw[valid]) if np.any(valid) else self._auto_scale(raw)
        else:
            divisor = float(scale)
            if not np.isfinite(divisor) or divisor <= 0.0:
                raise ValueError("reflectance scale must be a finite positive number")
        self.reflectance_scale_ = divisor
        output = frame.copy().reset_index(drop=True)
        for index, band in enumerate(self.band_order):
            output[band] = raw[:, index] / divisor
        output = output.loc[valid].reset_index(drop=True)
        logger.info(
            "Raster sampling | scale=1/%.0f | valid=%d/%d (%.1f%%)",
            divisor,
            len(output),
            len(frame),
            100.0 * len(output) / max(len(frame), 1),
        )
        if len(output) == 0:
            raise RuntimeError(
                "No field point was sampled successfully. Check CRS, raster extent "
                "and nodata/cloud/land masking."
            )
        return output

# =============================================================================
# Feature engineering
# =============================================================================

class FeatureEngineer:
    """Physically informed features with fit-on-training-only statistics.

    The fitted Lyzenga offsets and composite-index mean/std values are stored
    once in ``fit`` and reused unchanged by every later transform, including
    holdout points and raster tiles.  This prevents the subtle batch leakage
    present in the previous implementation.
    """

    def __init__(
        self,
        raw_bands: Sequence[str] = ("B2", "B3", "B4", "B8", "B11"),
        n_const: float = 1000.0,
        deep_water_percentile: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        self.raw_bands = list(raw_bands)
        self.n_const = float(n_const)
        self.deep_water_percentile = float(deep_water_percentile)
        self.eps = float(eps)
        self._lyzenga_offsets: Dict[str, float] = {}
        self._composite_stats: Dict[str, float] = {}
        self._feature_names: List[str] = []
        self._fitted = False
        self.reflectance_scale_: Optional[float] = None

    def fit(self, frame: pd.DataFrame) -> "FeatureEngineer":
        """Estimate all batch-dependent constants from ``frame`` only."""
        if frame.empty:
            raise ValueError("Cannot fit FeatureEngineer on an empty frame")
        for band in self.raw_bands:
            if band not in frame.columns:
                raise KeyError(f"Required spectral band is missing: {band}")
            values = pd.to_numeric(frame[band], errors="coerce").to_numpy(dtype=np.float64)
            valid = values[np.isfinite(values) & (values > 0.0)]
            if valid.size == 0:
                raise ValueError(f"Band {band} has no finite positive values")
            deep_value = float(np.percentile(valid, self.deep_water_percentile))
            self._lyzenga_offsets[band] = max(0.0, deep_value * 0.5)
        self._composite_stats = {}
        for index, (band_a, band_b) in enumerate(
            (("B2", "B3"), ("B2", "B4")), start=1
        ):
            if band_a not in frame.columns or band_b not in frame.columns:
                continue
            ratio = _safe_stumpf_ratio(
                frame[band_a].to_numpy(dtype=np.float64),
                frame[band_b].to_numpy(dtype=np.float64),
                n_const=self.n_const,
                eps=self.eps,
            )
            finite = ratio[np.isfinite(ratio)]
            if finite.size:
                self._composite_stats[f"mean_s{index}"] = float(np.mean(finite))
                self._composite_stats[f"std_s{index}"] = max(
                    float(np.std(finite)), self.eps
                )
        # Ensure both keys exist whenever one of the two ratios is unavailable.
        if "mean_s1" not in self._composite_stats:
            self._composite_stats.update({"mean_s1": 0.0, "std_s1": 1.0})
        if "mean_s2" not in self._composite_stats:
            self._composite_stats.update(
                {
                    "mean_s2": self._composite_stats["mean_s1"],
                    "std_s2": self._composite_stats["std_s1"],
                }
            )
        self._fitted = True
        dummy = {
            band: np.full(2, 0.02 + 0.001 * i, dtype=np.float64)
            for i, band in enumerate(self.raw_bands)
        }
        self._feature_names = list(self._compute_features(dummy).keys())
        logger.info(
            "FeatureEngineer fitted | samples=%d | features=%d | bands=%s",
            len(frame),
            len(self._feature_names),
            self.raw_bands,
        )
        return self

    def _compute_features(
        self, bands: Mapping[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        missing = [band for band in self.raw_bands if band not in bands]
        if missing:
            raise KeyError(f"Feature computation is missing bands: {missing}")
        arrays = {
            band: np.asarray(bands[band], dtype=np.float64) for band in self.raw_bands
        }
        features: Dict[str, np.ndarray] = {
            band: arrays[band] for band in self.raw_bands
        }
        # Pairwise algebraic features.
        for band_a, band_b in itertools.combinations(self.raw_bands, 2):
            a = arrays[band_a]
            b = arrays[band_b]
            a_safe = np.clip(a, self.eps, None)
            b_safe = np.clip(b, self.eps, None)
            features[f"R_{band_a}_{band_b}"] = a_safe / b_safe
            features[f"logR_{band_a}_{band_b}"] = np.log(a_safe / b_safe)
            features[f"D_{band_a}_{band_b}"] = a - b
            features[f"S_{band_a}_{band_b}"] = a + b
            features[f"P_{band_a}_{band_b}"] = a * b
        for band in self.raw_bands:
            features[f"log_{band}"] = np.log(np.clip(arrays[band], self.eps, None))
        # Stumpf features.
        for band_a, band_b in (("B2", "B3"), ("B2", "B4"), ("B3", "B4")):
            if band_a in arrays and band_b in arrays:
                features[f"Stumpf_{band_a}_{band_b}"] = _safe_stumpf_ratio(
                    arrays[band_a],
                    arrays[band_b],
                    n_const=self.n_const,
                    eps=self.eps,
                )
        # Lyzenga corrected log reflectance.
        for band in self.raw_bands:
            offset = self._lyzenga_offsets.get(band, 0.0)
            features[f"Lyzenga_{band}"] = np.log(
                np.clip(arrays[band] - offset, self.eps, None)
            )

        def has(*names: str) -> bool:
            return all(name in arrays for name in names)

        if has("B3", "B8"):
            features["NDWI"] = _safe_normalized_difference(
                arrays["B3"], arrays["B8"], self.eps
            )
        if has("B3", "B11"):
            features["MNDWI"] = _safe_normalized_difference(
                arrays["B3"], arrays["B11"], self.eps
            )
        if has("B4", "B3"):
            features["NDTI"] = _safe_normalized_difference(
                arrays["B4"], arrays["B3"], self.eps
            )
        if has("B8", "B4"):
            features["NDCI_proxy"] = _safe_normalized_difference(
                arrays["B8"], arrays["B4"], self.eps
            )
        if has("B2", "B3"):
            features["BlueGreen_Index"] = _safe_normalized_difference(
                arrays["B2"], arrays["B3"], self.eps
            )
        if has("B8", "B11"):
            features["Water_Absorption_Index"] = _safe_normalized_difference(
                arrays["B8"], arrays["B11"], self.eps
            )
        if has("B3", "B11", "B8"):
            features["AWEI_nsh_adapted"] = (
                4.0 * (arrays["B3"] - arrays["B11"])
                - 0.25 * arrays["B8"]
            )
        if has("B2", "B3", "B8", "B11"):
            features["AWEI_sh_adapted"] = (
                arrays["B2"]
                + 2.5 * arrays["B3"]
                - 1.5 * (arrays["B8"] + arrays["B11"])
            )
        if has("B4", "B3"):
            features["Turbidity_proxy"] = np.clip(
                arrays["B4"], self.eps, None
            ) / np.clip(arrays["B3"], self.eps, None)
        if has("B2", "B4"):
            nd_blue_red = _safe_normalized_difference(
                arrays["B2"], arrays["B4"], self.eps
            )
            ndti = features.get("NDTI", np.zeros_like(nd_blue_red))
            features["TMI_proxy"] = nd_blue_red / (
                1.0 + np.clip(ndti, -0.99, 0.99)
            )
        if "Stumpf_B2_B3" in features and "Stumpf_B2_B4" in features:
            s1 = features["Stumpf_B2_B3"]
            s2 = features["Stumpf_B2_B4"]
            stats = self._composite_stats or {
                "mean_s1": 0.0,
                "std_s1": 1.0,
                "mean_s2": 0.0,
                "std_s2": 1.0,
            }
            z1 = (s1 - stats["mean_s1"]) / max(stats["std_s1"], self.eps)
            z2 = (s2 - stats["mean_s2"]) / max(stats["std_s2"], self.eps)
            features["SDB_composite_index"] = 0.5 * (z1 + z2)
        return features

    def compute_features(
        self, bands: Mapping[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transformation")
        return self._compute_features(bands)

    def transform_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transformation")
        bands = {
            band: pd.to_numeric(frame[band], errors="coerce").to_numpy(dtype=np.float64)
            for band in self.raw_bands
        }
        features = self._compute_features(bands)
        output = pd.DataFrame(features, index=frame.index)
        output = output.replace([np.inf, -np.inf], np.nan)
        return output

    def transform_array(
        self, array: np.ndarray, band_order: Sequence[str]
    ) -> Dict[str, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transformation")
        data = np.asarray(array)
        if data.ndim != 3:
            raise ValueError(f"Expected raster array with shape (C,H,W); got {data.shape}")
        by_name = {
            name: data[index].astype(np.float64)
            for index, name in enumerate(band_order)
            if index < data.shape[0]
        }
        missing = [band for band in self.raw_bands if band not in by_name]
        if missing:
            raise KeyError(f"Raster tile is missing required bands: {missing}")
        return self._compute_features(by_name)

    @property
    def feature_names(self) -> List[str]:
        if not self._fitted:
            raise RuntimeError("Feature names are available after FeatureEngineer.fit")
        return list(self._feature_names)

    def build_dataset(self, sampled_frame: pd.DataFrame) -> pd.DataFrame:
        """Combine metadata with transformed features without refitting."""
        if not self._fitted:
            self.fit(sampled_frame)
        feature_frame = self.transform_dataframe(sampled_frame).reset_index(drop=True)
        metadata_columns = [
            column
            for column in ("ID", "Lon", "Lat", "Depth")
            if column in sampled_frame.columns
        ]
        metadata = sampled_frame[metadata_columns].reset_index(drop=True)
        dataset = pd.concat([metadata, feature_frame], axis=1)
        if not np.isfinite(
            dataset.drop(columns=metadata_columns, errors="ignore")
            .to_numpy(dtype=np.float64)
        ).any():
            raise ValueError("Feature engineering produced no finite values")
        return dataset

# =============================================================================
# Model base classes and physical models
# =============================================================================

def _validate_xy(
    X: pd.DataFrame, y: Sequence[float]
) -> Tuple[pd.DataFrame, np.ndarray]:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if X.empty:
        raise ValueError("X is empty")
    target = np.asarray(y, dtype=np.float64).ravel()
    if len(X) != len(target):
        raise ValueError("X and y have different lengths")
    if not np.all(np.isfinite(target)):
        raise ValueError("y contains non-finite values")
    return X.copy(), target


def _validate_sample_weight(
    sample_weight: Optional[Sequence[float]], n: int
) -> Optional[np.ndarray]:
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=np.float64).ravel()
    if len(weights) != n:
        raise ValueError("sample_weight has a different length than y")
    weights = np.where(np.isfinite(weights) & (weights >= 0.0), weights, 0.0)
    if float(weights.sum()) <= 0.0:
        return None
    return weights


def _require_columns(X: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in X.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")


class BaseSDBModel(BaseEstimator, RegressorMixin):
    """Common interface for all SDB regressors."""

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[float],
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "BaseSDBModel":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class StumpfBandRatioModel(BaseSDBModel):
    """Stumpf log-band-ratio regression: ``Z = a * ratio + b``."""

    def __init__(
        self,
        band_i: str = "B2",
        band_j: str = "B3",
        n_const: float = 1000.0,
        eps: float = 1e-6,
    ) -> None:
        self.band_i = band_i
        self.band_j = band_j
        self.n_const = float(n_const)
        self.eps = float(eps)
        self.model = LinearRegression()
        self.is_fitted_ = False

    def _ratio(self, X: pd.DataFrame) -> np.ndarray:
        _require_columns(X, [self.band_i, self.band_j])
        return _safe_stumpf_ratio(
            X[self.band_i].to_numpy(dtype=np.float64),
            X[self.band_j].to_numpy(dtype=np.float64),
            n_const=self.n_const,
            eps=self.eps,
        ).reshape(-1, 1)

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[float],
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "StumpfBandRatioModel":
        X, y = _validate_xy(X, y)
        weights = _validate_sample_weight(sample_weight, len(y))
        ratio = self._ratio(X)
        if not np.all(np.isfinite(ratio)):
            raise ValueError("Stumpf ratio contains non-finite values")
        self.model.fit(ratio, y, sample_weight=weights)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict()")
        return np.asarray(self.model.predict(self._ratio(X)), dtype=np.float64)


class LyzengaModel(BaseSDBModel):
    """Lyzenga multi-band log-linear regression."""

    def __init__(
        self,
        bands: Sequence[str] = ("B2", "B3", "B4"),
        deep_water_p: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        self.bands = list(bands)
        self.deep_water_p = float(deep_water_p)
        self.eps = float(eps)
        self.model = LinearRegression()
        self.offsets: Dict[str, float] = {}
        self.is_fitted_ = False

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        _require_columns(X, self.bands)
        columns = []
        for band in self.bands:
            corrected = X[band].to_numpy(dtype=np.float64) - self.offsets.get(band, 0.0)
            columns.append(np.log(np.clip(corrected, self.eps, None)))
        return np.column_stack(columns)

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[float],
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "LyzengaModel":
        X, y = _validate_xy(X, y)
        weights = _validate_sample_weight(sample_weight, len(y))
        self.offsets = {}
        for band in self.bands:
            _require_columns(X, [band])
            values = X[band].to_numpy(dtype=np.float64)
            finite_positive = values[np.isfinite(values) & (values > 0.0)]
            if finite_positive.size == 0:
                raise ValueError(f"Band {band} has no positive finite values")
            self.offsets[band] = max(
                0.0,
                float(np.percentile(finite_positive, self.deep_water_p)) * 0.5,
            )
        transformed = self._transform(X)
        if not np.all(np.isfinite(transformed)):
            raise ValueError("Lyzenga transformation contains non-finite values")
        self.model.fit(transformed, y, sample_weight=weights)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict()")
        return np.asarray(self.model.predict(self._transform(X)), dtype=np.float64)


class CaballeroStumpfModel(BaseSDBModel):
    """Adaptive blend of blue/green and blue/red Stumpf regressions."""

    def __init__(
        self,
        shallow_break: Optional[float] = None,
        deep_break: Optional[float] = None,
        n_const: float = 1000.0,
        eps: float = 1e-6,
    ) -> None:
        self.shallow_break = shallow_break
        self.deep_break = deep_break
        self.n_const = float(n_const)
        self.eps = float(eps)
        self.bg_model = StumpfBandRatioModel("B2", "B3", n_const, eps)
        self.br_model = StumpfBandRatioModel("B2", "B4", n_const, eps)
        self.shallow_break_eff_ = 2.0
        self.deep_break_eff_ = 4.0
        self.is_fitted_ = False

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[float],
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "CaballeroStumpfModel":
        X, y = _validate_xy(X, y)
        max_depth = float(np.max(y))
        if self.shallow_break is not None and self.deep_break is not None:
            shallow = float(self.shallow_break)
            deep = float(self.deep_break)
        elif max_depth <= 3.5:
            shallow = max(0.5, 0.45 * max_depth)
            deep = max(1.0, 0.85 * max_depth)
        else:
            shallow, deep = 2.0, 4.0
        if deep <= shallow:
            deep = shallow + 1e-6
        self.shallow_break_eff_ = shallow
        self.deep_break_eff_ = deep
        self.bg_model.fit(X, y, sample_weight=sample_weight)
        self.br_model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict()")
        blue_green = self.bg_model.predict(X)
        blue_red = self.br_model.predict(X)
        weight = np.clip(
            (blue_green - self.shallow_break_eff_)
            / (self.deep_break_eff_ - self.shallow_break_eff_),
            0.0,
            1.0,
        )
        return (weight * blue_green + (1.0 - weight) * blue_red).astype(np.float64)


class ClusterBasedRegression(BaseSDBModel):
    """KMeans spectral clusters with local Stumpf regressions."""

    def __init__(
        self,
        n_clusters: int = 4,
        random_state: int = 42,
        min_samples_per_cluster: int = 5,
        cluster_cols: Sequence[str] = ("B2", "B3", "B4", "B8"),
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.min_samples_per_cluster = int(min_samples_per_cluster)
        self.cluster_cols = list(cluster_cols)
        self.kmeans: Optional[KMeans] = None
        self.scaler = StandardScaler()
        self.models: Dict[int, BaseSDBModel] = {}
        self.global_model = StumpfBandRatioModel()
        self.is_fitted_ = False

    def _cluster_features(self, X: pd.DataFrame) -> np.ndarray:
        available = [column for column in self.cluster_cols if column in X.columns]
        if not available:
            raise KeyError(f"None of the cluster columns are present: {self.cluster_cols}")
        return X[available].to_numpy(dtype=np.float64)

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[float],
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "ClusterBasedRegression":
        X, y = _validate_xy(X, y)
        weights = _validate_sample_weight(sample_weight, len(y))
        raw_features = self._cluster_features(X)
        scaled = self.scaler.fit_transform(raw_features)
        n_unique = len(np.unique(scaled, axis=0))
        effective_k = max(1, min(self.n_clusters, n_unique, len(X)))
        self.kmeans = KMeans(
            n_clusters=effective_k,
            n_init=20,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            labels = self.kmeans.fit_predict(scaled)
        self.global_model.fit(X, y, sample_weight=weights)
        self.models = {}
        for cluster_id in range(effective_k):
            mask = labels == cluster_id
            if int(mask.sum()) < self.min_samples_per_cluster:
                continue
            try:
                local = StumpfBandRatioModel().fit(
                    X.loc[mask],
                    y[mask],
                    sample_weight=weights[mask] if weights is not None else None,
                )
                self.models[cluster_id] = local
            except Exception as exc:
                logger.debug("Cluster %d local fit skipped: %s", cluster_id, exc)
        if not self.models:
            self.models = {0: self.global_model}
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_ or self.kmeans is None:
            raise RuntimeError("Call fit() before predict()")
        scaled = self.scaler.transform(self._cluster_features(X))
        labels = self.kmeans.predict(scaled)
        prediction = np.empty(len(X), dtype=np.float64)
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            model = self.models.get(int(cluster_id), self.global_model)
            prediction[mask] = model.predict(X.loc[mask])
        return prediction

# =============================================================================
# Tabular ML model wrappers
# =============================================================================

class _PreprocessedModel(BaseSDBModel):
    """Shared median-imputation plus optional robust-scaling implementation."""

    scale_features: bool = True

    def _fit_preprocessor(self, X: pd.DataFrame) -> np.ndarray:
        self.feature_names_in_ = list(X.columns)
        steps: List[Tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median")),
        ]
        if self.scale_features:
            steps.append(("scaler", RobustScaler()))
        self.imputer = steps[0][1]
        transformed = self.imputer.fit_transform(X)
        if self.scale_features:
            self.scaler = steps[1][1]
            transformed = self.scaler.fit_transform(transformed)
        else:
            self.scaler = None
        return transformed

    def _transform_preprocessor(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "feature_names_in_"):
            raise RuntimeError("Model has not been fitted")
        missing = [column for column in self.feature_names_in_ if column not in X.columns]
        if missing:
            raise KeyError(f"Prediction data is missing columns: {missing}")
        values = X[self.feature_names_in_]
        transformed = self.imputer.transform(values)
        if self.scale_features and self.scaler is not None:
            transformed = self.scaler.transform(transformed)
        return transformed


class RandomForestSDB(_PreprocessedModel):
    scale_features = False

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Any = "sqrt",
        oob_score: bool = True,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.oob_score = oob_score
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.extra_params = dict(kwargs)
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            oob_score=oob_score,
            bootstrap=True,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[float],
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "RandomForestSDB":
        X, y = _validate_xy(X, y)
        weights = _validate_sample_weight(sample_weight, len(y))
        transformed = self._fit_preprocessor(X)
        self.model.fit(transformed, y, sample_weight=weights)
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(self._transform_preprocessor(X)), dtype=np.float64)


class ExtraTreesSDB(_PreprocessedModel):
    scale_features = False

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Any = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.extra_params = dict(kwargs)
        self.model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=False,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[float],
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "ExtraTreesSDB":
        X, y = _validate_xy(X, y)
        weights = _validate_sample_weight(sample_weight, len(y))
        self.model.fit(self._fit_preprocessor(X), y, sample_weight=weights)
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(self._transform_preprocessor(X)), dtype=np.float64)


class SVR_SDB(_PreprocessedModel):
    scale_features = True

    def __init__(
        self,
        C: float = 150.0,
        epsilon: float = 0.05,
        gamma: Union[str, float] = "scale",
        kernel: str = "rbf",
        tol: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.kernel = kernel
        self.tol = tol
        self.extra_params = dict(kwargs)
        self.model = SVR(
            C=C,
            epsilon=epsilon,
            gamma=gamma,
            kernel=kernel,
            tol=tol,
            **kwargs,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[float],
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "SVR_SDB":
        X, y = _validate_xy(X, y)
        weights = _validate_sample_weight(sample_weight, len(y))
        self.model.fit(self._fit_preprocessor(X), y, sample_weight=weights)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(self._transform_preprocessor(X)), dtype=np.float64)


class MLP_SDB(_PreprocessedModel):
    scale_features = True

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (128, 64, 32),
        activation: str = "relu",
        alpha: float = 1e-3,
        learning_rate_init: float = 0.0015,
        max_iter: int = 1500,
        early_stopping: bool = True,
        validation_fraction: float = 0.15,
        n_iter_no_change: int = 20,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.extra_params = dict(kwargs)
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            solver="adam",
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state,
            **kwargs,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[float],
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "MLP_SDB":
        X, y = _validate_xy(X, y)
        if sample_weight is not None:
            logger.debug("MLP_SDB ignores sample_weight because sklearn MLPRegressor has no weighted fit API")
        self.model.fit(self._fit_preprocessor(X), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(self._transform_preprocessor(X)), dtype=np.float64)


class XGBoost_SDB(_PreprocessedModel):
    scale_features = False

    def __init__(
        self,
        n_estimators: int = 600,
        max_depth: int = 7,
        learning_rate: float = 0.04,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        reg_alpha: float = 0.15,
        reg_lambda: float = 1.2,
        min_child_weight: int = 3,
        gamma: float = 0.0,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbosity": 0,
            **kwargs,
        }
        self._available = False
        self.model: Any = None
        try:
            from xgboost import XGBRegressor  # type: ignore
            self.model = XGBRegressor(**self.params)
            self._available = True
        except Exception as exc:
            logger.debug("XGBoost unavailable: %s", exc)

    def _require_available(self) -> None:
        if not self._available:
            raise ImportError("XGBoost model requires `pip install xgboost`")

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[float],
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "XGBoost_SDB":
        self._require_available()
        X, y = _validate_xy(X, y)
        weights = _validate_sample_weight(sample_weight, len(y))
        self.model.fit(
            self._fit_preprocessor(X),
            y,
            sample_weight=weights,
            verbose=False,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_available()
        return np.asarray(self.model.predict(self._transform_preprocessor(X)), dtype=np.float64)


class LightGBM_SDB(_PreprocessedModel):
    scale_features = False

    def __init__(
        self,
        n_estimators: int = 600,
        num_leaves: int = 64,
        learning_rate: float = 0.05,
        max_depth: int = -1,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        min_child_samples: int = 20,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        self.params = {
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "min_child_samples": min_child_samples,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbosity": -1,
            **kwargs,
        }
        self._available = False
        self.model: Any = None
        try:
            from lightgbm import LGBMRegressor  # type: ignore
            self.model = LGBMRegressor(**self.params)
            self._available = True
        except Exception as exc:
            logger.debug("LightGBM unavailable: %s", exc)

    def _require_available(self) -> None:
        if not self._available:
            raise ImportError("LightGBM model requires `pip install lightgbm`")

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[float],
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "LightGBM_SDB":
        self._require_available()
        X, y = _validate_xy(X, y)
        weights = _validate_sample_weight(sample_weight, len(y))
        self.model.fit(self._fit_preprocessor(X), y, sample_weight=weights)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_available()
        return np.asarray(self.model.predict(self._transform_preprocessor(X)), dtype=np.float64)


class CatBoost_SDB(_PreprocessedModel):
    scale_features = False

    def __init__(
        self,
        iterations: int = 600,
        depth: int = 8,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.5,
        bagging_temperature: float = 1.0,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.params = {
            "iterations": iterations,
            "depth": depth,
            "learning_rate": learning_rate,
            "l2_leaf_reg": l2_leaf_reg,
            "bagging_temperature": bagging_temperature,
            "random_seed": random_state,
            "verbose": False,
            "allow_writing_files": False,
            **kwargs,
        }
        self._available = False
        self.model: Any = None
        try:
            from catboost import CatBoostRegressor  # type: ignore
            self.model = CatBoostRegressor(**self.params)
            self._available = True
        except Exception as exc:
            logger.debug("CatBoost unavailable: %s", exc)

    def _require_available(self) -> None:
        if not self._available:
            raise ImportError("CatBoost model requires `pip install catboost`")

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[float],
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "CatBoost_SDB":
        self._require_available()
        X, y = _validate_xy(X, y)
        weights = _validate_sample_weight(sample_weight, len(y))
        self.model.fit(
            self._fit_preprocessor(X),
            y,
            sample_weight=weights,
            verbose=False,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_available()
        return np.asarray(self.model.predict(self._transform_preprocessor(X)), dtype=np.float64)


class ModelFactory:
    """Canonical model registry and construction API."""

    _REGISTRY: Dict[str, Type[BaseSDBModel]] = {
        "stumpf": StumpfBandRatioModel,
        "lyzenga": LyzengaModel,
        "caballero": CaballeroStumpfModel,
        "cbr": ClusterBasedRegression,
        "random_forest": RandomForestSDB,
        "extra_trees": ExtraTreesSDB,
        "svr": SVR_SDB,
        "mlp": MLP_SDB,
        "xgboost": XGBoost_SDB,
        "lightgbm": LightGBM_SDB,
        "catboost": CatBoost_SDB,
    }

    _ALIASES: Dict[str, str] = {
        "rf": "random_forest",
        "randomforest": "random_forest",
        "et": "extra_trees",
        "extratrees": "extra_trees",
        "support_vector": "svr",
        "xgb": "xgboost",
        "lgbm": "lightgbm",
        "cat": "catboost",
        "catboostregressor": "catboost",
        "cluster": "cbr",
    }

    @classmethod
    def canonical_name(cls, model_name: str) -> str:
        key = str(model_name).strip().lower()
        return cls._ALIASES.get(key, key)

    @classmethod
    def available_models(cls, include_aliases: bool = False) -> List[str]:
        names = list(cls._REGISTRY)
        if include_aliases:
            names.extend(cls._ALIASES)
        return sorted(set(names))

    @classmethod
    def is_registered(cls, model_name: str) -> bool:
        return cls.canonical_name(model_name) in cls._REGISTRY

    @classmethod
    def create(cls, model_name: str, **kwargs: Any) -> BaseSDBModel:
        canonical = cls.canonical_name(model_name)
        if canonical not in cls._REGISTRY:
            raise ValueError(
                f"Unknown model {model_name!r}. Available models: "
                f"{cls.available_models()}"
            )
        instance = cls._REGISTRY[canonical](**kwargs)
        logger.debug("ModelFactory | %s -> %s", model_name, type(instance).__name__)
        return instance

    @classmethod
    def create_all(
        cls,
        model_list: Optional[Sequence[str]] = None,
        **common_kwargs: Any,
    ) -> Dict[str, BaseSDBModel]:
        names = list(model_list) if model_list is not None else cls.available_models()
        result: Dict[str, BaseSDBModel] = {}
        for name in names:
            canonical = cls.canonical_name(name)
            try:
                result[canonical] = cls.create(canonical, **common_kwargs)
            except Exception as exc:
                logger.warning("Model %s was skipped: %s", name, exc)
        return result

# =============================================================================
# Bayesian optimization
# =============================================================================

@dataclass
class HPOResult:
    model_name: str
    best_params: Dict[str, Any]
    best_score: float
    study: Any
    n_trials: int
    best_trial: Optional[Any] = None
    cv_fold_scores: List[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"{self.model_name.upper():<16} | "
            f"best_RMSE={self.best_score:.4f} | "
            f"trials={self.n_trials} | "
            f"time={self.elapsed_seconds:.1f}s"
        )


SearchSpaceFn = Callable[[Any], Dict[str, Any]]

_SEARCH_SPACE_REGISTRY: Dict[str, SearchSpaceFn] = {}


def register_search_space(*aliases: str) -> Callable[[SearchSpaceFn], SearchSpaceFn]:
    def decorator(function: SearchSpaceFn) -> SearchSpaceFn:
        for alias in aliases:
            _SEARCH_SPACE_REGISTRY[ModelFactory.canonical_name(alias)] = function
        return function
    return decorator


@register_search_space("random_forest", "rf")
def _space_random_forest(trial: Any) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 24),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
    }


@register_search_space("extra_trees", "et")
def _space_extra_trees(trial: Any) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 24),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
    }


@register_search_space("xgboost", "xgb")
def _space_xgboost(trial: Any) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.25, log=True),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }


@register_search_space("lightgbm", "lgbm")
def _space_lightgbm(trial: Any) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=100),
        "num_leaves": trial.suggest_int("num_leaves", 20, 120),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", -1, 16),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
    }


@register_search_space("catboost", "cat")
def _space_catboost(trial: Any) -> Dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 300, 1200, step=100),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
    }


@register_search_space("svr")
def _space_svr(trial: Any) -> Dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-1, 500.0, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-4, 0.5, log=True),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }


_MLP_ARCHITECTURES: Tuple[Tuple[int, ...], ...] = (
    (32,),
    (64,),
    (128,),
    (64, 32),
    (128, 64),
    (128, 64, 32),
    (256, 128, 64),
)


@register_search_space("mlp")
def _space_mlp(trial: Any) -> Dict[str, Any]:
    # Fixed categorical architecture menu: the dimensionality of the search
    # space is identical in every trial, unlike a variable number of layer_i
    # parameters.
    architecture_idx = trial.suggest_categorical(
        "architecture_idx", list(range(len(_MLP_ARCHITECTURES)))
    )
    return {
        "architecture_idx": architecture_idx,
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float(
            "learning_rate_init", 5e-4, 5e-2, log=True
        ),
        "activation": trial.suggest_categorical(
            "activation", ["relu", "tanh", "logistic"]
        ),
        "early_stopping": True,
    }


@register_search_space("gmm_soft_mixture")
def _space_gmm(trial: Any) -> Dict[str, Any]:
    return {
        "n_components": trial.suggest_int("n_components", 2, 8),
        "covariance_type": trial.suggest_categorical(
            "covariance_type", ["full", "tied", "diag", "spherical"]
        ),
        "n_init": trial.suggest_int("n_init", 1, 5),
    }


class BayesianOptimizer:
    """Optuna optimizer with fixed search spaces and robust CV handling."""

    def __init__(
        self,
        cfg: SDBConfig,
        storage_name: str = "sdb_hpo_study.db",
    ) -> None:
        self.cfg = cfg
        self.storage_path = Path(cfg.output_dir) / storage_name
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_url = f"sqlite:///{self.storage_path.resolve()}"
        try:
            import optuna  # type: ignore
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            pass

    @staticmethod
    def register_model(*aliases: str) -> Callable[[SearchSpaceFn], SearchSpaceFn]:
        return register_search_space(*aliases)

    def _search_space(self, trial: Any, model_name: str) -> Dict[str, Any]:
        canonical = ModelFactory.canonical_name(model_name)
        builder = _SEARCH_SPACE_REGISTRY.get(canonical)
        return builder(trial) if builder is not None else {}

    @staticmethod
    def _normalize_params(model_name: str, params: Mapping[str, Any]) -> Dict[str, Any]:
        canonical = ModelFactory.canonical_name(model_name)
        normalized = dict(params)
        if canonical == "mlp" and "architecture_idx" in normalized:
            index = int(normalized.pop("architecture_idx"))
            index = max(0, min(index, len(_MLP_ARCHITECTURES) - 1))
            normalized["hidden_layer_sizes"] = _MLP_ARCHITECTURES[index]
        return normalized

    def _make_splits(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        cv_folds: int,
        spatial_splitter: Optional[Callable[..., Iterable[Tuple[np.ndarray, np.ndarray]]]],
        splits: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        if splits is not None:
            result = [(np.asarray(tr), np.asarray(val)) for tr, val in splits]
        elif spatial_splitter is not None:
            try:
                result = [
                    (np.asarray(tr), np.asarray(val))
                    for tr, val in spatial_splitter(X)
                ]
            except TypeError:
                result = [
                    (np.asarray(tr), np.asarray(val))
                    for tr, val in spatial_splitter()
                ]
        else:
            folds = min(max(2, int(cv_folds)), len(X))
            result = list(
                KFold(
                    n_splits=folds,
                    shuffle=True,
                    random_state=self.cfg.random_seed,
                ).split(X, y)
            )
        result = [
            (tr, val)
            for tr, val in result
            if len(tr) > 0 and len(val) > 0
        ]
        if not result:
            raise RuntimeError("HPO produced no valid CV splits")
        return result

    def _cross_val_rmse(
        self,
        trial: Any,
        model_name: str,
        params: Dict[str, Any],
        X: pd.DataFrame,
        y: np.ndarray,
        splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    ) -> float:
        try:
            from optuna.exceptions import TrialPruned  # type: ignore
        except ImportError:  # pragma: no cover
            TrialPruned = RuntimeError  # type: ignore
        fold_scores: List[float] = []
        for fold_index, (train_idx, valid_idx) in enumerate(splits):
            try:
                model = ModelFactory.create(model_name, **params)
                model.fit(X.iloc[train_idx], y[train_idx])
                prediction = model.predict(X.iloc[valid_idx])
                if not np.all(np.isfinite(prediction)):
                    raise ValueError("non-finite predictions")
                score = float(
                    math.sqrt(
                        mean_squared_error(y[valid_idx], prediction)
                    )
                )
            except Exception as exc:
                logger.debug(
                    "HPO trial=%s fold=%d model=%s failed: %s",
                    getattr(trial, "number", "?"),
                    fold_index,
                    model_name,
                    exc,
                )
                raise TrialPruned() from exc
            fold_scores.append(score)
            if hasattr(trial, "report"):
                trial.report(float(np.mean(fold_scores)), fold_index)
            if hasattr(trial, "should_prune") and trial.should_prune():
                raise TrialPruned()
        return float(np.mean(fold_scores))

    def optimize(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: Sequence[float],
        n_trials: Optional[int] = None,
        cv_folds: int = 5,
        spatial_splitter: Optional[Callable[..., Iterable[Tuple[np.ndarray, np.ndarray]]]] = None,
        splits: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
        timeout: Optional[int] = None,
        show_progress: bool = True,
    ) -> HPOResult:
        try:
            import optuna  # type: ignore
        except ImportError as exc:
            raise ImportError("Bayesian optimization requires `pip install optuna`") from exc
        X, target = _validate_xy(X, y)
        trial_count = int(n_trials if n_trials is not None else self.cfg.n_trials)
        if trial_count < 1:
            raise ValueError("n_trials must be >= 1")
        canonical = ModelFactory.canonical_name(model_name)
        cv_splits = self._make_splits(
            X, target, cv_folds, spatial_splitter, splits
        )
        # Include data shape and seed in the study name. This prevents a prior
        # experiment with a different feature matrix from silently contaminating
        # the current optimization.
        study_suffix = hashlib.sha1(
            f"{canonical}|{len(X)}|{X.shape[1]}|{self.cfg.random_seed}".encode()
        ).hexdigest()[:10]
        study_name = f"sdb_{canonical}_{study_suffix}"
        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage_url,
            load_if_exists=True,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.cfg.random_seed),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=min(8, max(1, trial_count // 3)),
                n_warmup_steps=2,
            ),
        )

        def objective(trial: Any) -> float:
            raw_params = self._search_space(trial, canonical)
            params = self._normalize_params(canonical, raw_params)
            return self._cross_val_rmse(
                trial, canonical, params, X, target, cv_splits
            )

        def callback(study_obj: Any, trial_obj: Any) -> None:
            if not show_progress:
                return
            interval = max(1, trial_count // 10)
            if (trial_obj.number + 1) % interval == 0 or trial_obj.number == 0:
                best_value = (
                    f"{study_obj.best_value:.4f}"
                    if study_obj.best_trial is not None
                    else "n/a"
                )
                logger.info(
                    "HPO | %-14s | trial %d/%d | best=%s",
                    canonical.upper(),
                    trial_obj.number + 1,
                    trial_count,
                    best_value,
                )

        start = time.time()
        study.optimize(
            objective,
            n_trials=trial_count,
            n_jobs=max(1, int(self.cfg.hpo_n_jobs)),
            timeout=timeout,
            callbacks=[callback] if show_progress else None,
        )
        elapsed = time.time() - start
        complete_trials = [
            trial
            for trial in study.trials
            if trial.state.name == "COMPLETE" and trial.value is not None
        ]
        if not complete_trials:
            raise RuntimeError(
                f"HPO completed without a valid trial for model {canonical!r}. "
                "Check model dependencies and feature values."
            )
        best_trial = min(complete_trials, key=lambda trial: float(trial.value))
        best_params = self._normalize_params(canonical, best_trial.params)
        best_score = float(best_trial.value)
        logger.info(
            "HPO complete | %s | RMSE=%.4f | trials=%d | %.1fs",
            canonical.upper(),
            best_score,
            len(study.trials),
            elapsed,
        )
        return HPOResult(
            model_name=canonical,
            best_params=best_params,
            best_score=best_score,
            study=study,
            n_trials=len(study.trials),
            best_trial=best_trial,
            elapsed_seconds=elapsed,
        )

    def batch_optimize(
        self,
        model_list: Sequence[str],
        X: pd.DataFrame,
        y: Sequence[float],
        **kwargs: Any,
    ) -> Dict[str, HPOResult]:
        results: Dict[str, HPOResult] = {}
        for model_name in model_list:
            try:
                result = self.optimize(model_name, X, y, **kwargs)
                results[result.model_name] = result
            except Exception as exc:
                logger.error("HPO failed for %s: %s", model_name, exc)
        return results

    @staticmethod
    def best_of(results: Mapping[str, HPOResult]) -> Optional[HPOResult]:
        return min(results.values(), key=lambda result: result.best_score) if results else None

# =============================================================================
# Evaluation and bootstrap confidence intervals
# =============================================================================

class ModelEvaluator:
    """Compute consistent regression metrics across all evaluation phases."""

    METRIC_NAMES = ("RMSE", "MAE", "R2", "Bias", "MAPE", "NRMSE", "CCC")

    @staticmethod
    def calculate_metrics(
        y_true: Sequence[float], y_pred: Sequence[float]
    ) -> Dict[str, float]:
        true, pred = _finite_pair(y_true, y_pred)
        rmse = float(math.sqrt(mean_squared_error(true, pred)))
        mae = float(mean_absolute_error(true, pred))
        bias = float(np.mean(pred - true))
        if len(true) >= 2 and float(np.var(true)) > np.finfo(float).eps:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r2 = float(r2_score(true, pred))
        else:
            r2 = float("nan")
        denominator = np.clip(np.abs(true), 0.1, None)
        mape = float(np.mean(np.abs((true - pred) / denominator)) * 100.0)
        depth_range = float(np.ptp(true))
        nrmse = float(rmse / depth_range * 100.0) if depth_range > 0.0 else float("nan")
        mean_true = float(np.mean(true))
        mean_pred = float(np.mean(pred))
        centered_true = true - mean_true
        centered_pred = pred - mean_pred
        covariance = float(np.mean(centered_true * centered_pred))
        variance_true = float(np.var(true))
        variance_pred = float(np.var(pred))
        ccc_denominator = variance_true + variance_pred + (mean_true - mean_pred) ** 2
        ccc = (
            float(2.0 * covariance / ccc_denominator)
            if ccc_denominator > np.finfo(float).eps
            else float("nan")
        )
        return {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Bias": bias,
            "MAPE": mape,
            "NRMSE": nrmse,
            "CCC": ccc,
        }

    def generate_report(
        self, results_dict: Mapping[str, Mapping[str, float]]
    ) -> pd.DataFrame:
        report = pd.DataFrame(results_dict).T
        if "RMSE" in report.columns:
            report = report.sort_values("RMSE", na_position="last")
        logger.info("\n%s\nFINAL MODEL PERFORMANCE\n%s", "=" * 72, report.round(4).to_string())
        return report

    def depth_bin_analysis(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        bins: Optional[Sequence[float]] = None,
    ) -> pd.DataFrame:
        true, pred = _finite_pair(y_true, y_pred)
        edges = get_adaptive_depth_bins(true, bins)
        rows: List[Dict[str, Any]] = []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (true >= low) & (true < high)
            if not np.any(mask):
                continue
            metrics = self.calculate_metrics(true[mask], pred[mask])
            rows.append(
                {
                    "Depth_Range": f"{low:g}-{high:g} m",
                    "N": int(mask.sum()),
                    **metrics,
                }
            )
        if not rows:
            return pd.DataFrame(columns=["N", *self.METRIC_NAMES])
        return pd.DataFrame(rows).set_index("Depth_Range")


class BootstrapEvaluator:
    """Non-parametric bootstrap intervals for small holdout datasets."""

    def __init__(
        self,
        n_bootstrap: int = 2000,
        random_seed: int = 42,
        alpha: float = 0.05,
        evaluator: Optional[ModelEvaluator] = None,
    ) -> None:
        if n_bootstrap < 100:
            raise ValueError("n_bootstrap must be >= 100")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.n_bootstrap = int(n_bootstrap)
        self.random_seed = int(random_seed)
        self.alpha = float(alpha)
        self.evaluator = evaluator or ModelEvaluator()

    def evaluate_bootstrap_ci(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
    ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        true, pred = _finite_pair(y_true, y_pred)
        point = self.evaluator.calculate_metrics(true, pred)
        distributions = {
            name: np.full(self.n_bootstrap, np.nan, dtype=np.float64)
            for name in point
        }
        rng = np.random.default_rng(self.random_seed)
        for iteration in range(self.n_bootstrap):
            indices = rng.integers(0, len(true), size=len(true))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics = self.evaluator.calculate_metrics(true[indices], pred[indices])
            for name, value in metrics.items():
                distributions[name][iteration] = value
        lower_q = 100.0 * self.alpha / 2.0
        upper_q = 100.0 * (1.0 - self.alpha / 2.0)
        summary: Dict[str, Any] = {}
        for name, values in distributions.items():
            finite = values[np.isfinite(values)]
            point_value = float(point[name])
            if finite.size == 0:
                lower = upper = mean = standard_error = float("nan")
            else:
                lower = float(np.percentile(finite, lower_q))
                upper = float(np.percentile(finite, upper_q))
                mean = float(np.mean(finite))
                standard_error = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            summary[f"Holdout_{name}_Point"] = point_value
            summary[f"Holdout_{name}_BootMean"] = mean
            summary[f"Holdout_{name}_BootSE"] = standard_error
            summary[f"Holdout_{name}_95CI_Lower"] = lower
            summary[f"Holdout_{name}_95CI_Upper"] = upper
            summary[f"Holdout_{name}_CI_str"] = (
                f"{point_value:.4f} ({lower:.4f} - {upper:.4f})"
                if np.isfinite(lower) and np.isfinite(upper)
                else "N/A"
            )
        return summary, distributions

# =============================================================================
# Spatial validation
# =============================================================================

class SpatialValidator:
    """Small-sample row-wise and spatial cross-validation engine.

    ``spatial_loo`` creates spatial blocks from coordinates and leaves one
    block out at a time.  For repeated spatial K-fold, the same coordinate
    blocks may be reused while fold assignment is independently shuffled per
    seed.  Clustering stability is reported separately and never used to
    incorrectly suppress variation in K-fold assignment.
    """

    ARI_DETERMINISTIC_THRESHOLD = 0.999

    def __init__(
        self,
        n_blocks: int = 15,
        n_folds: int = 5,
        random_seed: int = 42,
        cv_strategy: str = "spatial_loo",
        n_repeats: int = 10,
    ) -> None:
        if n_blocks < 2:
            raise ValueError("n_blocks must be >= 2")
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        if n_repeats < 1:
            raise ValueError("n_repeats must be >= 1")
        strategy = str(cv_strategy).strip().lower()
        if strategy not in {"loo", "spatial_loo", "repeated_spatial_cv", "kfold"}:
            raise ValueError(f"Unsupported cv_strategy: {cv_strategy}")
        self.n_blocks = int(n_blocks)
        self.n_folds = int(n_folds)
        self.random_seed = int(random_seed)
        self.cv_strategy = strategy
        self.n_repeats = int(n_repeats)
        logger.info(
            "SpatialValidator | strategy=%s | blocks=%d | folds=%d | repeats=%d",
            strategy.upper(),
            self.n_blocks,
            self.n_folds,
            self.n_repeats,
        )

    @staticmethod
    def _coordinate_array(frame: pd.DataFrame) -> np.ndarray:
        _require_columns(frame, ["Lon", "Lat"])
        coords = frame[["Lon", "Lat"]].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(coords)):
            raise ValueError("Spatial validation coordinates contain non-finite values")
        return coords

    def _cluster_coords(
        self, frame: pd.DataFrame, seed: Optional[int] = None
    ) -> np.ndarray:
        coords = self._coordinate_array(frame)
        n_samples = len(coords)
        n_unique = len(np.unique(coords, axis=0))
        if n_samples < 2 or n_unique < 2:
            return np.zeros(n_samples, dtype=int)
        effective_blocks = min(self.n_blocks, n_unique, n_samples)
        if effective_blocks < 2:
            return np.zeros(n_samples, dtype=int)
        # Standardizing coordinates prevents longitude/latitude scale from
        # dominating in unusual projected CRS ranges.
        coord_scaler = StandardScaler()
        scaled_coords = coord_scaler.fit_transform(coords)
        kmeans = KMeans(
            n_clusters=effective_blocks,
            n_init=20,
            random_state=self.random_seed if seed is None else int(seed),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return kmeans.fit_predict(scaled_coords)

    def assess_clustering_stability(
        self, frame: pd.DataFrame, seeds: Sequence[int]
    ) -> Dict[str, Any]:
        seed_list = list(dict.fromkeys(int(seed) for seed in seeds))
        labels_by_seed = {
            seed: self._cluster_coords(frame, seed=seed) for seed in seed_list
        }
        pairwise_ari: List[float] = []
        try:
            from sklearn.metrics import adjusted_rand_score
            for left_index, left_seed in enumerate(seed_list):
                for right_seed in seed_list[left_index + 1:]:
                    pairwise_ari.append(
                        float(
                            adjusted_rand_score(
                                labels_by_seed[left_seed], labels_by_seed[right_seed]
                            )
                        )
                    )
        except Exception:
            pairwise_ari = []
        ari_mean = float(np.mean(pairwise_ari)) if pairwise_ari else 1.0
        ari_min = float(np.min(pairwise_ari)) if pairwise_ari else 1.0
        deterministic = ari_min >= self.ARI_DETERMINISTIC_THRESHOLD
        logger.info(
            "Spatial block stability | ARI mean=%.4f min=%.4f | %s",
            ari_mean,
            ari_min,
            "deterministic" if deterministic else "varies by seed",
        )
        return {
            "ari_mean": ari_mean,
            "ari_min": ari_min,
            "is_deterministic": deterministic,
            "labels_by_seed": labels_by_seed,
        }

    def split_holdout(
        self, frame: pd.DataFrame, fraction: float = 0.20
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not 0.0 < fraction < 1.0:
            raise ValueError("fraction must be in (0, 1)")
        if len(frame) < 2:
            raise ValueError("At least two samples are required for a holdout split")
        labels = self._cluster_coords(frame, seed=self.random_seed)
        unique_blocks = np.unique(labels)
        if unique_blocks.size < 2:
            rng = np.random.default_rng(self.random_seed)
            indices = rng.permutation(len(frame))
            n_test = max(1, min(len(frame) - 1, int(round(len(frame) * fraction))))
            test_mask = np.zeros(len(frame), dtype=bool)
            test_mask[indices[:n_test]] = True
        else:
            n_test_blocks = max(
                1,
                min(unique_blocks.size - 1, int(round(unique_blocks.size * fraction))),
            )
            rng = np.random.default_rng(self.random_seed)
            test_blocks = rng.choice(unique_blocks, size=n_test_blocks, replace=False)
            test_mask = np.isin(labels, test_blocks)
            if test_mask.all() or (~test_mask).sum() == 0:
                test_mask[:] = False
                test_mask[rng.choice(len(frame), size=max(1, int(len(frame) * fraction)), replace=False)] = True
                if test_mask.all():
                    test_mask[-1] = False
        train = frame.loc[~test_mask].reset_index(drop=True)
        test = frame.loc[test_mask].reset_index(drop=True)
        logger.info(
            "Spatial holdout | train=%d | holdout=%d (%.1f%%)",
            len(train),
            len(test),
            100.0 * len(test) / len(frame),
        )
        return train, test

    def get_spatial_cv(
        self, frame: pd.DataFrame, seed: Optional[int] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        effective_seed = self.random_seed if seed is None else int(seed)
        n_samples = len(frame)
        if n_samples < 2:
            return
        if self.cv_strategy == "loo":
            yield from LeaveOneOut().split(frame)
            return
        labels = self._cluster_coords(frame, seed=effective_seed)
        unique_blocks = np.unique(labels)
        if unique_blocks.size < 2:
            yield from LeaveOneOut().split(frame)
            return
        if self.cv_strategy == "spatial_loo":
            for block in unique_blocks:
                valid = np.flatnonzero(labels == block)
                train = np.flatnonzero(labels != block)
                if len(train) and len(valid):
                    yield train, valid
            return
        n_splits = min(self.n_folds, len(unique_blocks))
        if n_splits < 2:
            yield from LeaveOneOut().split(frame)
            return
        block_splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=effective_seed,
        )
        for train_block_idx, valid_block_idx in block_splitter.split(unique_blocks):
            train_blocks = unique_blocks[train_block_idx]
            valid_blocks = unique_blocks[valid_block_idx]
            train = np.flatnonzero(np.isin(labels, train_blocks))
            valid = np.flatnonzero(np.isin(labels, valid_blocks))
            if len(train) and len(valid):
                yield train, valid

    @staticmethod
    def _feature_columns(frame: pd.DataFrame, feature_cols: Optional[Sequence[str]]) -> List[str]:
        if feature_cols is not None:
            columns = list(feature_cols)
        else:
            columns = [
                column
                for column in frame.columns
                if column not in {"ID", "Lon", "Lat", "Depth"}
            ]
        if not columns:
            raise ValueError("No feature columns supplied for validation")
        _require_columns(frame, columns + ["Depth"])
        return columns

    @staticmethod
    def _params_for_repetition(
        model_name: str,
        params: Mapping[str, Any],
        seed: int,
    ) -> Dict[str, Any]:
        """Inject a repetition-specific seed into stochastic model wrappers.

        A repeated CV pass is not genuinely independent when the spatial
        split and every model's ``random_state`` are kept fixed.  The previous
        implementation only varied the KMeans seed and then skipped repeated
        Spatial-LOO when ARI was 1.0.  This helper keeps deterministic physical
        models deterministic, while giving RF/ET/MLP/boosting/GMM/MoE models a
        different, reproducible seed on every requested repetition.
        """
        result = dict(params)
        stochastic_models = {
            "cbr",
            "gmm_soft_mixture",
            "random_forest",
            "extra_trees",
            "mlp",
            "xgboost",
            "lightgbm",
            "catboost",
            "hetero_moe",
            "moe",
        }
        canonical = ModelFactory.canonical_name(model_name)
        if canonical in stochastic_models:
            result["random_state"] = int(seed)
        return result

    def validate_model_single_run(
        self,
        model_factory: Any,
        frame: pd.DataFrame,
        model_name: str,
        params: Optional[Mapping[str, Any]] = None,
        feature_cols: Optional[Sequence[str]] = None,
        evaluator: Optional[ModelEvaluator] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]], np.ndarray]:
        """Run one CV pass and return summary, fold records and aligned OOF predictions."""
        evaluator = evaluator or ModelEvaluator()
        columns = self._feature_columns(frame, feature_cols)
        params = dict(params or {})
        X = frame[columns]
        y = frame["Depth"].to_numpy(dtype=np.float64)
        oof = np.full(len(frame), np.nan, dtype=np.float64)
        fold_records: List[Dict[str, Any]] = []
        generated_splits = list(self.get_spatial_cv(frame, seed=seed))
        if not generated_splits:
            raise RuntimeError("No valid CV folds were generated")
        repetition_seed = self.random_seed if seed is None else int(seed)
        for fold_number, (train_idx, valid_idx) in enumerate(generated_splits, start=1):
            # Keep the same repetition seed for all folds of one pass so the
            # comparison is controlled; the seed changes between repetitions.
            run_params = self._params_for_repetition(
                model_name, params, repetition_seed
            )
            model = model_factory.create(model_name, **run_params)
            model.fit(X.iloc[train_idx], y[train_idx])
            prediction = np.asarray(model.predict(X.iloc[valid_idx]), dtype=np.float64).ravel()
            if len(prediction) != len(valid_idx) or not np.all(np.isfinite(prediction)):
                raise RuntimeError(
                    f"Model {model_name!r} produced invalid predictions on fold {fold_number}"
                )
            oof[valid_idx] = prediction
            metrics = evaluator.calculate_metrics(y[valid_idx], prediction)
            fold_records.append(
                {
                    "Model": ModelFactory.canonical_name(model_name),
                    "Seed": self.random_seed if seed is None else int(seed),
                    "Fold": fold_number,
                    "N_Train": len(train_idx),
                    "N_Val": len(valid_idx),
                    **metrics,
                }
            )
        valid_oof = np.isfinite(oof)
        oof_metrics = (
            evaluator.calculate_metrics(y[valid_oof], oof[valid_oof])
            if np.any(valid_oof)
            else {name: float("nan") for name in evaluator.METRIC_NAMES}
        )
        summary: Dict[str, float] = {
            f"OOF_{name}": float(oof_metrics.get(name, np.nan))
            for name in evaluator.METRIC_NAMES
        }
        for name in evaluator.METRIC_NAMES:
            values = np.asarray(
                [record.get(name, np.nan) for record in fold_records],
                dtype=np.float64,
            )
            finite = values[np.isfinite(values)]
            summary[f"FoldMean_{name}"] = float(np.mean(finite)) if finite.size else float("nan")
            summary[f"FoldStd_{name}"] = float(np.std(finite)) if finite.size else float("nan")
        summary["N_Folds"] = float(len(fold_records))
        return summary, fold_records, oof

    def evaluate_repeated_cv(
        self,
        model_factory: Any,
        frame: pd.DataFrame,
        model_name: str,
        params: Optional[Mapping[str, Any]] = None,
        feature_cols: Optional[Sequence[str]] = None,
        evaluator: Optional[ModelEvaluator] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], np.ndarray]:
        """Evaluate a model over the configured small-sample CV repetitions."""
        evaluator = evaluator or ModelEvaluator()
        params = dict(params or {})
        metric_names = evaluator.METRIC_NAMES
        candidate_seeds = [self.random_seed + i for i in range(self.n_repeats)]
        stability: Dict[str, Any] = {
            "ari_mean": np.nan,
            "ari_min": np.nan,
            "is_deterministic": None,
        }
        if self.cv_strategy == "loo":
            # The row-wise LOO split itself is fixed, but stochastic models
            # receive a different reproducible random_state on every pass.
            effective_seeds = candidate_seeds
            status_note = (
                "row-wise LOO split is fixed; all requested model-seed "
                "repetitions are executed"
            )
        elif self.cv_strategy == "spatial_loo":
            stability = self.assess_clustering_stability(frame, candidate_seeds)
            # IMPORTANT: ARI=1 does not cancel the requested repetitions.
            # The spatial blocks may be identical, but stochastic models are
            # refit with a different seed on every pass.  This is the behavior
            # required by n_repeats and makes the reported mean/std genuine.
            effective_seeds = candidate_seeds
            if stability["is_deterministic"]:
                status_note = (
                    "spatial blocks are ARI-verified identical; all requested "
                    "repetitions are executed with different model seeds"
                )
            else:
                status_note = (
                    "spatial blocks and model seeds vary across all requested "
                    "repetitions"
                )
        else:
            # For spatial K-fold, fold assignment is shuffled per seed even if
            # KMeans produces the same blocks. All repetitions are executed.
            stability = self.assess_clustering_stability(frame, candidate_seeds)
            effective_seeds = candidate_seeds
            status_note = "K-fold assignment and model seeds are repeated"
        logger.info(
            "CV | %-18s | %s | executing %d/%d pass(es)",
            ModelFactory.canonical_name(model_name).upper(),
            status_note,
            len(effective_seeds),
            self.n_repeats,
        )
        seed_records: List[Dict[str, Any]] = []
        all_fold_records: List[Dict[str, Any]] = []
        first_oof: Optional[np.ndarray] = None
        for repetition, seed in enumerate(effective_seeds, start=1):
            logger.info(
                "CV pass | %-18s | repetition %d/%d | seed=%d",
                ModelFactory.canonical_name(model_name).upper(),
                repetition,
                len(effective_seeds),
                seed,
            )
            summary, folds, oof = self.validate_model_single_run(
                model_factory,
                frame,
                model_name,
                params=params,
                feature_cols=feature_cols,
                evaluator=evaluator,
                seed=seed,
            )
            if first_oof is None:
                first_oof = oof.copy()
            all_fold_records.extend(folds)
            seed_record: Dict[str, Any] = {
                "Model": ModelFactory.canonical_name(model_name),
                "Repetition": repetition,
                "Seed": seed,
                "N_Folds": summary["N_Folds"],
            }
            for metric in metric_names:
                seed_record[metric] = summary[f"OOF_{metric}"]
            seed_records.append(seed_record)
        if first_oof is None:
            raise RuntimeError("Repeated CV produced no OOF predictions")
        repeated_summary: Dict[str, Any] = {
            "Model": ModelFactory.canonical_name(model_name),
            "CV_Strategy": self.cv_strategy.upper(),
            "N_Repeats_Requested": self.n_repeats,
            "N_Repeats_Executed": len(effective_seeds),
            "Clustering_ARI_Mean": stability["ari_mean"],
            "Clustering_ARI_Min": stability["ari_min"],
            "Clustering_Verified_Deterministic": stability["is_deterministic"],
        }
        for metric in metric_names:
            values = np.asarray(
                [record.get(metric, np.nan) for record in seed_records],
                dtype=np.float64,
            )
            finite = values[np.isfinite(values)]
            repeated_summary[f"Repeated_{metric}_mean"] = (
                float(np.mean(finite)) if finite.size else float("nan")
            )
            repeated_summary[f"Repeated_{metric}_std"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            )
        rmse_mean = repeated_summary["Repeated_RMSE_mean"]
        rmse_std = repeated_summary["Repeated_RMSE_std"]
        repeated_summary["RMSE_Stability_COV_Percent"] = (
            float(100.0 * rmse_std / max(abs(rmse_mean), 1e-12))
            if np.isfinite(rmse_mean)
            else float("nan")
        )
        cov = repeated_summary["RMSE_Stability_COV_Percent"]
        if self.cv_strategy == "loo" and len(effective_seeds) > 1:
            stability_status = "Repeated model seeds on fixed row-wise LOO splits"
        elif self.cv_strategy == "spatial_loo" and stability["is_deterministic"]:
            if len(effective_seeds) > 1:
                stability_status = (
                    "Repeated model seeds on fixed spatial-LOO blocks "
                    f"(ARI_min={stability['ari_min']:.4f})"
                )
            else:
                stability_status = (
                    "Deterministic spatial-LOO partition "
                    f"(ARI_min={stability['ari_min']:.4f})"
                )
        elif not np.isfinite(cov):
            stability_status = "Undefined"
        elif cov < 5.0:
            stability_status = "Highly Stable (COV < 5%)"
        elif cov < 10.0:
            stability_status = "Stable (COV < 10%)"
        else:
            stability_status = f"Sensitive (COV={cov:.1f}%)"
        repeated_summary["Stability_Status"] = stability_status
        logger.info(
            "CV result | %-18s | RMSE=%.4f +/- %.4f m | R2=%.4f +/- %.4f | %s",
            ModelFactory.canonical_name(model_name).upper(),
            repeated_summary["Repeated_RMSE_mean"],
            repeated_summary["Repeated_RMSE_std"],
            repeated_summary["Repeated_R2_mean"],
            repeated_summary["Repeated_R2_std"],
            stability_status,
        )
        return repeated_summary, seed_records, all_fold_records, first_oof

    def validate_model_full(
        self,
        model_factory: Any,
        frame: pd.DataFrame,
        model_name: str,
        params: Optional[Mapping[str, Any]] = None,
        feature_cols: Optional[Sequence[str]] = None,
        evaluator: Optional[ModelEvaluator] = None,
    ) -> Dict[str, Any]:
        summary, seed_records, fold_records, first_oof = self.evaluate_repeated_cv(
            model_factory,
            frame,
            model_name,
            params=params,
            feature_cols=feature_cols,
            evaluator=evaluator,
        )
        return {
            "summary": summary,
            "seed_records": seed_records,
            "fold_records": fold_records,
            "first_run_oof": first_oof,
        }

    def validate_model(
        self,
        model_factory: Any,
        frame: pd.DataFrame,
        model_name: str,
        params: Optional[Mapping[str, Any]] = None,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> List[float]:
        """Backward-compatible return of per-repetition OOF RMSE values."""
        result = self.validate_model_full(
            model_factory,
            frame,
            model_name,
            params=params,
            feature_cols=feature_cols,
        )
        return [float(record["RMSE"]) for record in result["seed_records"]]

# =============================================================================
# Sensitivity, Monte-Carlo uncertainty and error propagation
# =============================================================================

class SensitivityAnalyzer:
    def __init__(self, model: Any, feature_names: Sequence[str]) -> None:
        self.model = model
        self.feature_names = list(feature_names)

    def run_oat_sensitivity(
        self,
        X_base: pd.DataFrame,
        perturbation: float = 0.05,
    ) -> pd.DataFrame:
        if perturbation <= 0.0:
            raise ValueError("perturbation must be positive")
        base_prediction = np.asarray(self.model.predict(X_base), dtype=np.float64)
        base_mean = float(np.nanmean(base_prediction))
        scale = base_mean if abs(base_mean) > 1e-12 else 1e-12
        rows: List[Dict[str, Any]] = []
        for feature in self.feature_names:
            if feature not in X_base.columns:
                continue
            perturbed = X_base.copy()
            perturbed[feature] = perturbed[feature] * (1.0 + perturbation)
            new_mean = float(np.nanmean(self.model.predict(perturbed)))
            relative_output_change = (new_mean - base_mean) / scale
            sensitivity = relative_output_change / perturbation
            rows.append(
                {
                    "Feature": feature,
                    "Sensitivity_Index": float(sensitivity),
                    "Absolute_Impact": float(abs(sensitivity)),
                    "Base_Mean": base_mean,
                    "Perturbed_Mean": new_mean,
                    "Delta_Mean": new_mean - base_mean,
                }
            )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(
            "Absolute_Impact", ascending=False
        ).reset_index(drop=True)

    def evaluate_noise_robustness(
        self,
        X: pd.DataFrame,
        noise_levels: Sequence[float] = (0.01, 0.02, 0.05, 0.10),
        random_state: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        base = np.asarray(self.model.predict(X), dtype=np.float64)
        feature_std = np.nanstd(X.to_numpy(dtype=np.float64), axis=0)
        records = []
        for level in noise_levels:
            level = float(level)
            noise = rng.normal(0.0, level, size=X.shape) * feature_std
            noisy = pd.DataFrame(
                X.to_numpy(dtype=np.float64) + noise,
                columns=X.columns,
                index=X.index,
            )
            prediction = np.asarray(self.model.predict(noisy), dtype=np.float64)
            records.append(
                {
                    "Noise_Level": f"{level * 100.0:.1f}%",
                    "Noise_Fraction": level,
                    "RMSE_Drift": float(np.sqrt(np.mean((base - prediction) ** 2))),
                    "Stability_Correlation": safe_pearson_corr(base, prediction),
                }
            )
        return pd.DataFrame(records)


class UncertaintyQuantifier:
    def __init__(
        self,
        model: Any,
        n_realizations: int = 500,
        random_seed: int = 42,
    ) -> None:
        if n_realizations < 1:
            raise ValueError("n_realizations must be >= 1")
        self.model = model
        self.n_realizations = int(n_realizations)
        self.random_seed = int(random_seed)

    def compute_mc_uncertainty(
        self,
        X: pd.DataFrame,
        rel_noise: float = 0.05,
        abs_noise: float = 0.0005,
    ) -> pd.DataFrame:
        if len(X) == 0:
            return pd.DataFrame(
                columns=[
                    "Depth_Mean",
                    "Uncertainty_Std",
                    "CI_Lower_95",
                    "CI_Upper_95",
                    "CV_Percent",
                ]
            )
        values = X.to_numpy(dtype=np.float64)
        rng = np.random.default_rng(self.random_seed)
        realizations = np.empty((self.n_realizations, len(X)), dtype=np.float64)
        for index in range(self.n_realizations):
            noise = (
                rng.normal(0.0, rel_noise, size=values.shape) * values
                + rng.normal(0.0, abs_noise, size=values.shape)
            )
            perturbed = pd.DataFrame(values + noise, columns=X.columns, index=X.index)
            realizations[index] = np.asarray(self.model.predict(perturbed), dtype=np.float64)
        mean_depth = np.nanmean(realizations, axis=0)
        standard_deviation = np.nanstd(realizations, axis=0)
        lower = np.nanpercentile(realizations, 2.5, axis=0)
        upper = np.nanpercentile(realizations, 97.5, axis=0)
        cv = 100.0 * standard_deviation / np.clip(np.abs(mean_depth), 0.1, None)
        return pd.DataFrame(
            {
                "Depth_Mean": mean_depth,
                "Uncertainty_Std": standard_deviation,
                "CI_Lower_95": lower,
                "CI_Upper_95": upper,
                "CV_Percent": cv,
            },
            index=X.index,
        )


class ErrorPropagator:
    """Depth-dependent and spatially correlated uncertainty propagation."""

    def __init__(self, cfg: SDBConfig) -> None:
        self.cfg = cfg
        self.error_model: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self._error_x: Optional[np.ndarray] = None
        self._error_y: Optional[np.ndarray] = None

    def fit_error_model(
        self, y_true: Sequence[float], y_pred: Sequence[float]
    ) -> None:
        true, pred = _finite_pair(y_true, y_pred)
        edges = get_adaptive_depth_bins(true, self.cfg.depth_bins)
        centers: List[float] = []
        rmses: List[float] = []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (true >= low) & (true < high)
            if int(mask.sum()) >= 3:
                centers.append(float((low + high) / 2.0))
                rmses.append(float(np.sqrt(np.mean((true[mask] - pred[mask]) ** 2))))
        global_rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
        if len(centers) < 2:
            min_depth = float(np.min(true))
            max_depth = float(np.max(true))
            if max_depth <= min_depth:
                max_depth = min_depth + 1e-6
            centers = [min_depth, max_depth]
            rmses = [global_rmse, global_rmse]
        order = np.argsort(centers)
        x = np.asarray(centers, dtype=np.float64)[order]
        y = np.maximum(np.asarray(rmses, dtype=np.float64)[order], 1e-8)
        unique_x, unique_idx = np.unique(x, return_index=True)
        unique_y = y[unique_idx]
        self._error_x = unique_x
        self._error_y = unique_y
        if len(unique_x) >= 3:
            interpolator = PchipInterpolator(unique_x, unique_y, extrapolate=True)
            self.error_model = lambda values: np.asarray(interpolator(values), dtype=np.float64)
        else:
            self.error_model = lambda values: np.interp(
                np.asarray(values, dtype=np.float64), unique_x, unique_y
            )

    @staticmethod
    def generate_gaussian_random_field(
        shape: Tuple[int, int],
        scale_px: float,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        height, width = shape
        generator = rng or np.random.default_rng()
        white = generator.normal(0.0, 1.0, size=(height, width))
        if scale_px <= 0.0:
            result = white
        else:
            spectrum = np.fft.fft2(white)
            ky = np.fft.fftfreq(height).reshape(-1, 1)
            kx = np.fft.fftfreq(width).reshape(1, -1)
            filter_kernel = np.exp(
                -2.0 * np.pi**2 * (ky**2 + kx**2) * float(scale_px) ** 2
            )
            result = np.real(np.fft.ifft2(spectrum * filter_kernel))
        result = result - np.mean(result)
        return result / max(float(np.std(result)), 1e-12)

    def _sigma(self, depth: np.ndarray) -> np.ndarray:
        if self.error_model is None:
            raise RuntimeError("Call fit_error_model before propagating uncertainty")
        return np.maximum(np.abs(self.error_model(depth)), 0.0)

    def propagate_spatial_error(
        self,
        depth_map: np.ndarray,
        n_simulations: int = 50,
    ) -> Dict[str, np.ndarray]:
        if n_simulations < 1:
            raise ValueError("n_simulations must be >= 1")
        depth = np.asarray(depth_map, dtype=np.float64)
        if depth.ndim != 2:
            raise ValueError("depth_map must be a two-dimensional array")
        valid = np.isfinite(depth)
        safe_depth = np.where(valid, depth, 0.0)
        sigma = self._sigma(safe_depth)
        accumulator = np.zeros_like(safe_depth, dtype=np.float64)
        rng = np.random.default_rng(self.cfg.random_seed)
        for _ in range(n_simulations):
            field = self.generate_gaussian_random_field(
                depth.shape,
                self.cfg.spatial_correlation_length_px,
                rng=rng,
            )
            accumulator += (field * sigma) ** 2
        variance = accumulator / float(n_simulations)
        standard_deviation = np.sqrt(variance)
        standard_deviation[~valid] = np.nan
        variance[~valid] = np.nan
        sigma[~valid] = np.nan
        return {
            "Spatial_Uncertainty_Std": standard_deviation,
            "Total_Error_Variance": variance,
            "Sigma_Z_Map": sigma,
        }

    def write_spatial_uncertainty_raster(
        self,
        depth_raster: Union[str, Path],
        output_raster: Union[str, Path],
        n_simulations: int = 50,
    ) -> Path:
        """Write uncertainty tile-by-tile so a large scene is not loaded at once."""
        _require_rasterio()
        if self.error_model is None:
            raise RuntimeError("Call fit_error_model before writing uncertainty raster")
        source_path = Path(depth_raster)
        target_path = Path(output_raster)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(source_path) as src:  # type: ignore[union-attr]
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                count=1,
                dtype="float32",
                nodata=float(self.cfg.nodata_value),
                compress="deflate",
                predictor=2,
                BIGTIFF="IF_SAFER",
            )
            with rasterio.open(target_path, "w", **profile) as dst:  # type: ignore[union-attr]
                windows = _iter_raster_windows(src, self.cfg.tile_size)
                for index, window in enumerate(windows, start=1):
                    values = src.read(1, window=window).astype(np.float64)
                    valid = np.isfinite(values) & (values != self.cfg.nodata_value)
                    safe = np.where(valid, values, 0.0)
                    sigma = self._sigma(safe)
                    accumulator = np.zeros_like(safe, dtype=np.float64)
                    rng = np.random.default_rng(self.cfg.random_seed + index)
                    for _ in range(n_simulations):
                        field = self.generate_gaussian_random_field(
                            safe.shape,
                            self.cfg.spatial_correlation_length_px,
                            rng=rng,
                        )
                        accumulator += (field * sigma) ** 2
                    output = np.sqrt(accumulator / n_simulations).astype(np.float32)
                    output[~valid] = self.cfg.nodata_value
                    dst.write(output, 1, window=window)
        logger.info("Spatial uncertainty raster written: %s", target_path)
        return target_path

# =============================================================================
# Tiled raster inference
# =============================================================================

def _iter_raster_windows(src: Any, tile_size: int) -> Iterable[Any]:
    """Yield source block windows or regular windows for non-tiled rasters."""
    if getattr(src, "is_tiled", False):
        for _, window in src.block_windows(1):
            yield window
        return
    for row_off in range(0, src.height, tile_size):
        for col_off in range(0, src.width, tile_size):
            yield Window(
                col_off,
                row_off,
                min(tile_size, src.width - col_off),
                min(tile_size, src.height - row_off),
            )


class RasterInferenceEngine:
    """Memory-efficient, scale-fixed tiled inference over a GeoTIFF scene."""

    def __init__(
        self,
        model: Any,
        feature_engineer: FeatureEngineer,
        cfg: SDBConfig,
        reflectance_scale: Optional[float] = None,
    ) -> None:
        self.model = model
        self.feature_engineer = feature_engineer
        self.cfg = cfg
        self.tile_size = int(cfg.tile_size)
        self.reflectance_scale = float(
            reflectance_scale
            or feature_engineer.reflectance_scale_
            or cfg.reflectance_scale
            or 10000.0
        )
        if self.reflectance_scale <= 0.0:
            raise ValueError("reflectance_scale must be positive")

    def _clip_predictions(self, prediction: np.ndarray) -> np.ndarray:
        output = np.asarray(prediction, dtype=np.float64)
        if self.cfg.prediction_min_depth is not None:
            output = np.maximum(output, self.cfg.prediction_min_depth)
        if self.cfg.prediction_max_depth is not None:
            output = np.minimum(output, self.cfg.prediction_max_depth)
        return output

    def predict_scene(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> Path:
        _require_rasterio()
        input_path = Path(input_path)
        output_path = Path(output_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input raster not found: {input_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(input_path) as src:  # type: ignore[union-attr]
            if src.count < len(self.feature_engineer.raw_bands):
                raise ValueError(
                    f"Input raster has {src.count} bands but "
                    f"{len(self.feature_engineer.raw_bands)} are required"
                )
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                count=1,
                dtype="float32",
                nodata=float(self.cfg.nodata_value),
                compress="deflate",
                predictor=2,
                BIGTIFF="IF_SAFER",
            )
            windows = _iter_raster_windows(src, self.tile_size)
            with rasterio.open(output_path, "w", **profile) as dst:  # type: ignore[union-attr]
                for index, window in enumerate(windows, start=1):
                    tile = src.read(window=window)
                    valid = np.all(np.isfinite(tile), axis=0)
                    if src.nodata is not None and np.isfinite(src.nodata):
                        valid &= ~np.any(np.isclose(tile, src.nodata), axis=0)
                    if self.cfg.zero_is_nodata:
                        valid &= np.all(tile > 0.0, axis=0)
                    output = np.full(
                        (int(window.height), int(window.width)),
                        self.cfg.nodata_value,
                        dtype=np.float32,
                    )
                    if np.any(valid):
                        scaled = tile.astype(np.float64) / self.reflectance_scale
                        features = self.feature_engineer.transform_array(
                            scaled, self.cfg.band_order
                        )
                        feature_names = self.feature_engineer.feature_names
                        matrix = np.column_stack(
                            [features[name].reshape(-1) for name in feature_names]
                        )
                        feature_frame = pd.DataFrame(matrix, columns=feature_names)
                        invalid_feature = ~np.isfinite(feature_frame.to_numpy()).all(axis=1)
                        feature_frame = feature_frame.replace(
                            [np.inf, -np.inf], np.nan
                        ).fillna(0.0)
                        prediction = self._clip_predictions(self.model.predict(feature_frame))
                        invalid = (
                            invalid_feature
                            | (~valid.reshape(-1))
                            | (~np.isfinite(prediction))
                        )
                        prediction[invalid] = self.cfg.nodata_value
                        output = prediction.reshape(
                            int(window.height), int(window.width)
                        ).astype(np.float32)
                    dst.write(output, 1, window=window)
        logger.info("Bathymetry raster written: %s", output_path)
        return output_path

# =============================================================================
# Explainability
# =============================================================================

class SDBExplainer:
    def __init__(self, model: Any, feature_names: Sequence[str]) -> None:
        self.model = model
        self.feature_names = list(feature_names)

    def compute_permutation_importance(
        self,
        X: pd.DataFrame,
        y: Sequence[float],
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> pd.DataFrame:
        from sklearn.inspection import permutation_importance
        result = permutation_importance(
            self.model,
            X,
            np.asarray(y),
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
            scoring="neg_root_mean_squared_error",
        )
        return pd.DataFrame(
            {
                "Feature": self.feature_names,
                "Importance_Mean": result.importances_mean,
                "Importance_Std": result.importances_std,
            }
        ).sort_values("Importance_Mean", ascending=False).reset_index(drop=True)

# =============================================================================
# Mixture models
# =============================================================================

class GMMSoftMixtureSDB(BaseSDBModel):
    """GMM responsibilities used as soft weights for local physical experts."""

    def __init__(
        self,
        n_components: int = 4,
        covariance_type: str = "full",
        cluster_cols: Sequence[str] = ("B2", "B3", "B4", "B8"),
        local_model: str = "stumpf",
        stumpf_bands: Tuple[str, str] = ("B2", "B3"),
        lyzenga_bands: Sequence[str] = ("B2", "B3", "B4"),
        min_effective_weight: float = 5.0,
        random_state: int = 42,
        n_init: int = 5,
    ) -> None:
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.cluster_cols = list(cluster_cols)
        self.local_model = str(local_model).lower()
        self.stumpf_bands = tuple(stumpf_bands)
        self.lyzenga_bands = list(lyzenga_bands)
        self.min_effective_weight = float(min_effective_weight)
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        if self.local_model not in {"stumpf", "lyzenga"}:
            raise ValueError("local_model must be 'stumpf' or 'lyzenga'")
        self.scaler = StandardScaler()
        self.gmm: Optional[GaussianMixture] = None
        self.experts_: Dict[int, BaseSDBModel] = {}
        self.active_components_: List[int] = []
        self.is_fitted_ = False

    def _cluster_features(self, X: pd.DataFrame) -> np.ndarray:
        _require_columns(X, self.cluster_cols)
        return X[self.cluster_cols].to_numpy(dtype=np.float64)

    def _new_local_expert(self) -> BaseSDBModel:
        if self.local_model == "lyzenga":
            return LyzengaModel(self.lyzenga_bands)
        return StumpfBandRatioModel(
            self.stumpf_bands[0], self.stumpf_bands[1]
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[float],
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "GMMSoftMixtureSDB":
        X, y = _validate_xy(X, y)
        weights = _validate_sample_weight(sample_weight, len(y))
        scaled = self.scaler.fit_transform(self._cluster_features(X))
        n_unique = len(np.unique(scaled, axis=0))
        components = max(1, min(self.n_components, n_unique, len(X)))
        self.gmm = GaussianMixture(
            n_components=components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=self.n_init,
            reg_covar=1e-6,
        )
        self.gmm.fit(scaled)
        responsibilities = self.gmm.predict_proba(scaled)
        self.experts_ = {}
        self.active_components_ = []
        for component in range(components):
            component_weights = responsibilities[:, component]
            if component_weights.sum() < self.min_effective_weight:
                continue
            try:
                expert = self._new_local_expert().fit(
                    X, y, sample_weight=component_weights
                )
                self.experts_[component] = expert
                self.active_components_.append(component)
            except Exception as exc:
                logger.debug("GMM component %d skipped: %s", component, exc)
        if not self.experts_:
            self.experts_[0] = self._new_local_expert().fit(X, y, sample_weight=weights)
            self.active_components_ = [0]
        self.is_fitted_ = True
        return self

    def _responsibilities(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_ or self.gmm is None:
            raise RuntimeError("Call fit() before predict()")
        full = self.gmm.predict_proba(
            self.scaler.transform(self._cluster_features(X))
        )
        active = full[:, self.active_components_]
        return active / np.clip(active.sum(axis=1, keepdims=True), 1e-12, None)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        responsibilities = self._responsibilities(X)
        prediction = np.zeros(len(X), dtype=np.float64)
        for index, component in enumerate(self.active_components_):
            prediction += responsibilities[:, index] * self.experts_[component].predict(X)
        return prediction

    def responsibility_entropy(self, X: pd.DataFrame) -> np.ndarray:
        responsibilities = self._responsibilities(X)
        if responsibilities.shape[1] <= 1:
            return np.zeros(len(X), dtype=np.float64)
        entropy = -np.sum(
            responsibilities * np.log(responsibilities + 1e-12), axis=1
        )
        return entropy / np.log(responsibilities.shape[1])


@dataclass
class ExpertSpec:
    """Declarative description of a physical or ModelFactory expert."""

    name: str
    kind: str
    model_type: str
    feature_cols: Optional[List[str]] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


class _ExpertWrapper:
    def __init__(self, spec: ExpertSpec) -> None:
        self.spec = spec
        self.model = self._build_model()

    def _build_model(self) -> BaseSDBModel:
        kind = self.spec.kind.lower()
        model_type = self.spec.model_type.lower()
        if kind == "physical":
            if model_type == "stumpf":
                bands = self.spec.model_kwargs.get("bands", ("B2", "B3"))
                return StumpfBandRatioModel(bands[0], bands[1])
            if model_type == "lyzenga":
                bands = self.spec.model_kwargs.get("bands", ("B2", "B3", "B4"))
                return LyzengaModel(bands)
            raise ValueError(f"Unknown physical expert type: {model_type}")
        if kind == "ml":
            return ModelFactory.create(model_type, **self.spec.model_kwargs)
        raise ValueError(f"Unknown expert kind: {self.spec.kind}")

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.spec.feature_cols] if self.spec.feature_cols else X

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[float],
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "_ExpertWrapper":
        selected = self._select(X)
        self.model.fit(selected, y, sample_weight=sample_weight)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(self._select(X)), dtype=np.float64)


class BaseGatingStrategy(ABC):
    def __init__(
        self,
        feature_cols: Sequence[str],
        random_state: int = 42,
        **_ : Any,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.random_state = int(random_state)

    def _features(self, X: pd.DataFrame) -> np.ndarray:
        _require_columns(X, self.feature_cols)
        return X[self.feature_cols].to_numpy(dtype=np.float64)

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        expert_oof_predictions: np.ndarray,
        y: np.ndarray,
    ) -> "BaseGatingStrategy":
        ...

    @abstractmethod
    def predict_weights(self, X: pd.DataFrame) -> np.ndarray:
        ...


_GATING_REGISTRY: Dict[str, Type[BaseGatingStrategy]] = {}


def register_gating_strategy(
    *aliases: str,
) -> Callable[[Type[BaseGatingStrategy]], Type[BaseGatingStrategy]]:
    def decorator(cls: Type[BaseGatingStrategy]) -> Type[BaseGatingStrategy]:
        for alias in aliases:
            _GATING_REGISTRY[alias.lower()] = cls
        return cls
    return decorator


@register_gating_strategy("equal", "uniform")
class EqualWeightGatingStrategy(BaseGatingStrategy):
    def fit(
        self,
        X: pd.DataFrame,
        expert_oof_predictions: np.ndarray,
        y: np.ndarray,
    ) -> "EqualWeightGatingStrategy":
        self.n_experts_ = int(expert_oof_predictions.shape[1])
        return self

    def predict_weights(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(
            (len(X), self.n_experts_),
            1.0 / max(self.n_experts_, 1),
            dtype=np.float64,
        )


@register_gating_strategy("gmm", "gmm_gate")
class GMMGatingStrategy(BaseGatingStrategy):
    def __init__(
        self,
        feature_cols: Sequence[str],
        random_state: int = 42,
        n_components: Optional[int] = None,
        covariance_type: str = "full",
        n_init: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(feature_cols, random_state, **kwargs)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_init = int(n_init)
        self.scaler = StandardScaler()
        self.gmm: Optional[GaussianMixture] = None

    def fit(
        self,
        X: pd.DataFrame,
        expert_oof_predictions: np.ndarray,
        y: np.ndarray,
    ) -> "GMMGatingStrategy":
        self.n_experts_ = int(expert_oof_predictions.shape[1])
        scaled = self.scaler.fit_transform(self._features(X))
        components = max(
            1,
            min(self.n_components or self.n_experts_, len(np.unique(scaled, axis=0)), len(X)),
        )
        self.gmm = GaussianMixture(
            n_components=components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=self.n_init,
            reg_covar=1e-6,
        ).fit(scaled)
        return self

    def predict_weights(self, X: pd.DataFrame) -> np.ndarray:
        if self.gmm is None:
            raise RuntimeError("GMM gate has not been fitted")
        responsibilities = self.gmm.predict_proba(
            self.scaler.transform(self._features(X))
        )
        if responsibilities.shape[1] == self.n_experts_:
            return responsibilities
        output = np.zeros((len(X), self.n_experts_), dtype=np.float64)
        count = min(responsibilities.shape[1], self.n_experts_)
        output[:, :count] = responsibilities[:, :count]
        return output / np.clip(output.sum(axis=1, keepdims=True), 1e-12, None)


@register_gating_strategy("learned", "supervised", "auto")
class LearnedGatingStrategy(BaseGatingStrategy):
    """Regularized RF gate trained on soft OOF expert-error targets."""

    def __init__(
        self,
        feature_cols: Sequence[str],
        random_state: int = 42,
        temperature: float = 1.0,
        n_estimators: int = 150,
        max_depth: Optional[int] = 4,
        min_samples_leaf: Optional[int] = None,
        shrinkage: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(feature_cols, random_state, **kwargs)
        self.temperature = max(float(temperature), 1e-6)
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
        self.regressor: Optional[RandomForestRegressor] = None

    @staticmethod
    def _soft_targets(errors: np.ndarray, temperature: float) -> np.ndarray:
        logits = -errors / max(temperature, 1e-6)
        logits -= np.max(logits, axis=1, keepdims=True)
        exponential = np.exp(np.clip(logits, -700.0, 700.0))
        return exponential / np.clip(exponential.sum(axis=1, keepdims=True), 1e-12, None)

    def fit(
        self,
        X: pd.DataFrame,
        expert_oof_predictions: np.ndarray,
        y: np.ndarray,
    ) -> "LearnedGatingStrategy":
        self.n_experts_ = int(expert_oof_predictions.shape[1])
        min_leaf = self.min_samples_leaf or max(5, len(X) // 20)
        errors = np.abs(expert_oof_predictions - y.reshape(-1, 1))
        targets = self._soft_targets(errors, self.temperature)
        self.regressor = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=min_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(self._features(X), targets)
        return self

    def predict_weights(self, X: pd.DataFrame) -> np.ndarray:
        if self.regressor is None:
            raise RuntimeError("Learned gate has not been fitted")
        learned = np.clip(self.regressor.predict(self._features(X)), 0.0, None)
        learned /= np.clip(learned.sum(axis=1, keepdims=True), 1e-12, None)
        uniform = np.full_like(learned, 1.0 / self.n_experts_)
        return (1.0 - self.shrinkage) * learned + self.shrinkage * uniform

class HeterogeneousMoESDB(BaseSDBModel):
    """Heterogeneous mixture of physical and ML experts with OOF gating."""

    def __init__(
        self,
        expert_specs: Optional[Sequence[ExpertSpec]] = None,
        gating: str = "learned",
        gating_kwargs: Optional[Mapping[str, Any]] = None,
        gate_features: Optional[Sequence[str]] = None,
        cv_folds: int = 5,
        auto_select: bool = True,
        min_expert_weight: float = 0.05,
        n_gate_repeats: int = 3,
        random_state: int = 42,
    ) -> None:
        self.expert_specs = list(expert_specs) if expert_specs else self._default_specs()
        self.gating = str(gating).lower()
        self.gating_kwargs = dict(gating_kwargs or {})
        self.gate_features = list(gate_features) if gate_features else None
        self.cv_folds = int(cv_folds)
        self.auto_select = bool(auto_select)
        self.min_expert_weight = float(min_expert_weight)
        self.n_gate_repeats = max(1, int(n_gate_repeats))
        self.random_state = int(random_state)
        if self.gating not in _GATING_REGISTRY:
            raise ValueError(
                f"Unknown gating strategy {gating!r}; available={sorted(_GATING_REGISTRY)}"
            )
        names = [spec.name for spec in self.expert_specs]
        if not names or len(names) != len(set(names)):
            raise ValueError("expert_specs must contain at least one unique expert name")
        self._experts: Dict[str, _ExpertWrapper] = {}
        self._expert_names: List[str] = []
        self._gate: Optional[BaseGatingStrategy] = None
        self._gate_cols: List[str] = []
        self._selection_report: Optional[pd.DataFrame] = None
        self.is_fitted_ = False

    @staticmethod
    def _default_specs() -> List[ExpertSpec]:
        return [
            ExpertSpec(
                "stumpf_b2b3",
                "physical",
                "stumpf",
                model_kwargs={"bands": ("B2", "B3")},
            ),
            ExpertSpec(
                "lyzenga",
                "physical",
                "lyzenga",
                model_kwargs={"bands": ("B2", "B3", "B4")},
            ),
            ExpertSpec("random_forest", "ml", "random_forest"),
            ExpertSpec("extra_trees", "ml", "extra_trees"),
            ExpertSpec("svr", "ml", "svr"),
        ]

    @staticmethod
    def _new_wrapper(spec: ExpertSpec) -> _ExpertWrapper:
        return _ExpertWrapper(spec)

    def _make_oof_predictions(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        names: Sequence[str],
        seed: int,
        sample_weight: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = len(X)
        output = np.full((n, len(names)), np.nan, dtype=np.float64)
        folds = min(max(2, self.cv_folds), n)
        if folds < 2:
            return output
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        specs = {spec.name: spec for spec in self.expert_specs}
        for train_idx, valid_idx in splitter.split(X):
            for expert_index, name in enumerate(names):
                try:
                    wrapper = self._new_wrapper(specs[name])
                    wrapper.fit(
                        X.iloc[train_idx],
                        y[train_idx],
                        sample_weight=(
                            sample_weight[train_idx]
                            if sample_weight is not None
                            else None
                        ),
                    )
                    prediction = wrapper.predict(X.iloc[valid_idx])
                    if np.all(np.isfinite(prediction)):
                        output[valid_idx, expert_index] = prediction
                except Exception as exc:
                    logger.debug("MoE OOF expert=%s failed: %s", name, exc)
        return output

    def _fit_gate(
        self,
        X: pd.DataFrame,
        oof: np.ndarray,
        y: np.ndarray,
        seed: int,
    ) -> Tuple[BaseGatingStrategy, np.ndarray]:
        valid = np.isfinite(oof).all(axis=1)
        if valid.sum() < max(10, self.cv_folds):
            raise RuntimeError(
                f"Only {int(valid.sum())} complete OOF rows are available for MoE gating"
            )
        gate_class = _GATING_REGISTRY[self.gating]
        kwargs = dict(self.gating_kwargs)
        kwargs.setdefault("random_state", seed)
        gate = gate_class(feature_cols=self._gate_cols, **kwargs)
        gate.fit(
            X.loc[valid].reset_index(drop=True),
            oof[valid],
            y[valid],
        )
        return gate, valid

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[float],
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "HeterogeneousMoESDB":
        X, y = _validate_xy(X, y)
        weights = _validate_sample_weight(sample_weight, len(y))
        self._gate_cols = self.gate_features or [
            column for column in X.columns if column not in {"ID", "Lon", "Lat", "Depth"}
        ]
        _require_columns(X, self._gate_cols)
        self._experts = {
            spec.name: self._new_wrapper(spec) for spec in self.expert_specs
        }
        self._expert_names = list(self._experts)
        if self.auto_select and len(self._expert_names) > 1:
            repeat_weights: List[np.ndarray] = []
            for repetition in range(self.n_gate_repeats):
                seed = self.random_state + repetition
                oof = self._make_oof_predictions(
                    X, y, self._expert_names, seed, sample_weight=weights
                )
                try:
                    gate, valid = self._fit_gate(X, oof, y, seed)
                    repeat_weights.append(
                        gate.predict_weights(X.loc[valid].reset_index(drop=True)).mean(axis=0)
                    )
                except Exception as exc:
                    logger.debug("MoE gate-repeat %d skipped: %s", repetition + 1, exc)
            if repeat_weights:
                matrix = np.vstack(repeat_weights)
                mean_weight = matrix.mean(axis=0)
                std_weight = matrix.std(axis=0)
                keep = mean_weight >= self.min_expert_weight
                if not np.any(keep):
                    keep[np.argmax(mean_weight)] = True
                self._selection_report = pd.DataFrame(
                    {
                        "Expert": self._expert_names,
                        "Mean_Gate_Weight": mean_weight,
                        "Std_Gate_Weight": std_weight,
                        "Kept": keep,
                    }
                ).sort_values("Mean_Gate_Weight", ascending=False)
                dropped = [
                    name
                    for index, name in enumerate(self._expert_names)
                    if not keep[index]
                ]
                if dropped:
                    logger.info("MoE auto-selection dropped experts: %s", dropped)
                    self._expert_names = [
                        name
                        for index, name in enumerate(self._expert_names)
                        if keep[index]
                    ]
                    self._experts = {
                        name: self._experts[name] for name in self._expert_names
                    }
        for wrapper in self._experts.values():
            wrapper.fit(X, y, sample_weight=weights)
        final_oof = self._make_oof_predictions(
            X, y, self._expert_names, self.random_state, sample_weight=weights
        )
        try:
            self._gate, _ = self._fit_gate(X, final_oof, y, self.random_state)
        except Exception as exc:
            logger.warning("MoE learned gate failed; using equal weights: %s", exc)
            self._gate = EqualWeightGatingStrategy(self._gate_cols).fit(
                X, np.nan_to_num(final_oof, nan=np.nanmean(final_oof, axis=0)), y
            )
        self.is_fitted_ = True
        logger.info(
            "HeterogeneousMoESDB fitted | gating=%s | experts=%s",
            self.gating,
            self._expert_names,
        )
        return self

    @property
    def selection_report(self) -> Optional[pd.DataFrame]:
        return self._selection_report

    def expert_weights(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_ or self._gate is None:
            raise RuntimeError("Call fit() before expert_weights()")
        return self._gate.predict_weights(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        weights = self.expert_weights(X)
        predictions = np.column_stack(
            [self._experts[name].predict(X) for name in self._expert_names]
        )
        return np.sum(weights * predictions, axis=1)

    def gate_entropy(self, X: pd.DataFrame) -> np.ndarray:
        weights = self.expert_weights(X)
        if weights.shape[1] <= 1:
            return np.zeros(len(X), dtype=np.float64)
        return -np.sum(weights * np.log(weights + 1e-12), axis=1) / np.log(weights.shape[1])


class MixtureOfExpertsSDB(BaseSDBModel):
    """Optional differentiable neural MoE; requires PyTorch."""

    def __init__(
        self,
        n_experts: int = 4,
        expert_type: str = "stumpf",
        stumpf_band_pairs: Optional[Sequence[Tuple[str, str]]] = None,
        default_stumpf_pair: Tuple[str, str] = ("B2", "B3"),
        gate_hidden: Sequence[int] = (32, 16),
        gate_features: Optional[Sequence[str]] = None,
        n_const: float = 1000.0,
        eps: float = 1e-6,
        lambda_sharp: float = 0.05,
        lambda_balance: float = 0.05,
        epochs: int = 300,
        lr: float = 1e-2,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        patience: int = 25,
        val_fraction: float = 0.15,
        device: str = "cpu",
        random_state: int = 42,
        verbose_every: int = 50,
    ) -> None:
        self.n_experts = int(n_experts)
        self.expert_type = str(expert_type).lower()
        self.stumpf_band_pairs = list(stumpf_band_pairs) if stumpf_band_pairs else None
        self.default_stumpf_pair = tuple(default_stumpf_pair)
        self.gate_hidden = list(gate_hidden)
        self.gate_features = list(gate_features) if gate_features else None
        self.n_const = float(n_const)
        self.eps = float(eps)
        self.lambda_sharp = float(lambda_sharp)
        self.lambda_balance = float(lambda_balance)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.val_fraction = float(val_fraction)
        self.device = str(device)
        self.random_state = int(random_state)
        self.verbose_every = int(verbose_every)
        if self.n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        if self.expert_type not in {"stumpf", "linear"}:
            raise ValueError("expert_type must be 'stumpf' or 'linear'")
        self._available = detect_torch_available()
        self._net: Any = None
        self._gate_cols: List[str] = []
        self._input_cols: List[str] = []
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self._fitted = False
        self.history_: Dict[str, List[float]] = {}

    def _pairs(self) -> List[Tuple[str, str]]:
        return self.stumpf_band_pairs or [self.default_stumpf_pair] * self.n_experts

    def _expert_inputs(self, X: pd.DataFrame) -> List[np.ndarray]:
        if self.expert_type == "linear":
            matrix = X[self._input_cols].to_numpy(dtype=np.float32)
            return [matrix for _ in range(self.n_experts)]
        outputs = []
        for band_i, band_j in self._pairs():
            ratio = _safe_stumpf_ratio(
                X[band_i].to_numpy(dtype=np.float64),
                X[band_j].to_numpy(dtype=np.float64),
                n_const=self.n_const,
                eps=self.eps,
            )
            outputs.append(ratio.reshape(-1, 1).astype(np.float32))
        return outputs

    def _gate_inputs(self, X: pd.DataFrame) -> np.ndarray:
        raw = X[self._gate_cols].to_numpy(dtype=np.float64)
        return ((raw - self._x_mean) / self._x_std).astype(np.float32)

    def _build_network(self, n_gate: int, expert_dims: List[int]) -> Any:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore

        class GatedNetwork(nn.Module):
            def __init__(self, gate_dim: int, dims: List[int], hidden: List[int], count: int) -> None:
                super().__init__()
                layers: List[nn.Module] = []
                previous = gate_dim
                for width in hidden:
                    layers.extend([nn.Linear(previous, width), nn.Tanh()])
                    previous = width
                layers.append(nn.Linear(previous, count))
                self.gate = nn.Sequential(*layers)
                self.experts = nn.ModuleList([nn.Linear(dim, 1) for dim in dims])

            def forward(self, gate_input: Any, expert_inputs: List[Any]) -> Tuple[Any, Any]:
                weights = torch.softmax(self.gate(gate_input), dim=1)
                expert_outputs = torch.cat(
                    [self.experts[i](expert_inputs[i]) for i in range(len(self.experts))],
                    dim=1,
                )
                return torch.sum(weights * expert_outputs, dim=1), weights

        torch.manual_seed(self.random_state)
        return GatedNetwork(n_gate, expert_dims, self.gate_hidden, self.n_experts).to(self.device)

    def fit(
        self,
        X: pd.DataFrame,
        y: Sequence[float],
        sample_weight: Optional[Sequence[float]] = None,
    ) -> "MixtureOfExpertsSDB":
        if not self._available:
            raise ImportError("Neural MoE requires `pip install torch`")
        import torch  # type: ignore
        import torch.nn.functional as functional  # type: ignore

        X, y = _validate_xy(X, y)
        set_global_seed(self.random_state)
        self._gate_cols = self.gate_features or [
            column for column in X.columns if column not in {"ID", "Lon", "Lat", "Depth"}
        ]
        if self.expert_type == "linear":
            self._input_cols = list(self._gate_cols)
        raw_gate = X[self._gate_cols].to_numpy(dtype=np.float64)
        self._x_mean = np.nanmean(raw_gate, axis=0)
        self._x_std = np.nanstd(raw_gate, axis=0) + 1e-8
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) + 1e-8
        rng = np.random.default_rng(self.random_state)
        permutation = rng.permutation(len(X))
        validation_size = max(1, min(len(X) - 1, int(round(len(X) * self.val_fraction))))
        valid_idx = permutation[:validation_size]
        train_idx = permutation[validation_size:]
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_valid = X.iloc[valid_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_valid = y[valid_idx]
        gate_train = torch.tensor(self._gate_inputs(X_train), device=self.device)
        gate_valid = torch.tensor(self._gate_inputs(X_valid), device=self.device)
        expert_train = [torch.tensor(a, device=self.device) for a in self._expert_inputs(X_train)]
        expert_valid = [torch.tensor(a, device=self.device) for a in self._expert_inputs(X_valid)]
        target_train = torch.tensor(
            ((y_train - self._y_mean) / self._y_std).astype(np.float32),
            device=self.device,
        )
        dimensions = [array.shape[1] for array in self._expert_inputs(X_train)]
        self._net = self._build_network(gate_train.shape[1], dimensions)
        optimizer = torch.optim.Adam(
            self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.history_ = {"train_loss": [], "val_rmse": [], "gate_entropy": []}
        best_state = None
        best_score = float("inf")
        no_improvement = 0
        for epoch in range(1, self.epochs + 1):
            self._net.train()
            order = torch.randperm(len(X_train))
            batch_losses: List[float] = []
            batch_entropy: List[float] = []
            for start in range(0, len(X_train), self.batch_size):
                idx = order[start:start + self.batch_size]
                estimate, weights = self._net(
                    gate_train[idx], [values[idx] for values in expert_train]
                )
                fit_loss = functional.mse_loss(estimate, target_train[idx])
                sample_entropy = -torch.sum(
                    weights * torch.log(weights + 1e-8), dim=1
                ).mean()
                mean_weights = weights.mean(dim=0)
                batch_entropy_value = -torch.sum(
                    mean_weights * torch.log(mean_weights + 1e-8)
                )
                loss = (
                    fit_loss
                    + self.lambda_sharp * sample_entropy
                    - self.lambda_balance * batch_entropy_value
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))
                batch_entropy.append(float(sample_entropy.detach().cpu()))
            self._net.eval()
            with torch.no_grad():
                valid_estimate, valid_weights = self._net(gate_valid, expert_valid)
                valid_depth = valid_estimate.cpu().numpy() * self._y_std + self._y_mean
                valid_rmse = float(np.sqrt(np.mean((valid_depth - y_valid) ** 2)))
                entropy = float(
                    (-torch.sum(
                        valid_weights * torch.log(valid_weights + 1e-8), dim=1
                    )).mean().cpu()
                )
            self.history_["train_loss"].append(float(np.mean(batch_losses)))
            self.history_["val_rmse"].append(valid_rmse)
            self.history_["gate_entropy"].append(entropy)
            if valid_rmse < best_score - 1e-6:
                best_score = valid_rmse
                best_state = {
                    key: value.detach().clone()
                    for key, value in self._net.state_dict().items()
                }
                no_improvement = 0
            else:
                no_improvement += 1
            if no_improvement >= self.patience:
                break
        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._fitted = True
        logger.info("MixtureOfExpertsSDB fitted | validation RMSE=%.4f m", best_score)
        return self

    def _forward(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted or self._net is None:
            raise RuntimeError("Call fit() before prediction")
        import torch  # type: ignore
        self._net.eval()
        with torch.no_grad():
            estimate, weights = self._net(
                torch.tensor(self._gate_inputs(X), device=self.device),
                [torch.tensor(a, device=self.device) for a in self._expert_inputs(X)],
            )
        return (
            estimate.cpu().numpy() * self._y_std + self._y_mean,
            weights.cpu().numpy(),
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._forward(X)[0]

    def gate_entropy(self, X: pd.DataFrame) -> np.ndarray:
        weights = self._forward(X)[1]
        if weights.shape[1] <= 1:
            return np.zeros(len(X), dtype=np.float64)
        return -np.sum(weights * np.log(weights + 1e-12), axis=1) / np.log(weights.shape[1])


# Register optional mixture models after their definitions.
ModelFactory._REGISTRY.update(
    {
        "gmm_soft_mixture": GMMSoftMixtureSDB,
        "hetero_moe": HeterogeneousMoESDB,
        "moe": MixtureOfExpertsSDB,
    }
)

ModelFactory._ALIASES.update(
    {
        "gmm_mixture": "gmm_soft_mixture",
        "gmmsoftmixture": "gmm_soft_mixture",
        "heterogeneous_moe": "hetero_moe",
        "heterogeneousmoe": "hetero_moe",
        "mixture_of_experts": "moe",
        "mixtureofexperts": "moe",
    }
)


class StructuralUncertaintyDecomposer:
    """Combine reflectance-noise uncertainty with gate structural uncertainty."""

    def __init__(self, aleatoric_quantifier: UncertaintyQuantifier, entropy_fn: Callable[[pd.DataFrame], np.ndarray]) -> None:
        self.uq = aleatoric_quantifier
        self.entropy_fn = entropy_fn
        self.kappa_: Optional[float] = None

    @staticmethod
    def _calibrate_kappa(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        entropy: np.ndarray,
        high_quantile: float = 0.75,
    ) -> float:
        residual = np.abs(y_pred - y_true)
        threshold = float(np.quantile(entropy, high_quantile))
        high = entropy >= threshold
        low = ~high
        if high.sum() < 3 or low.sum() < 3:
            return 1.0
        high_rmse = float(np.sqrt(np.mean(residual[high] ** 2)))
        low_rmse = float(np.sqrt(np.mean(residual[low] ** 2)))
        return max(high_rmse - low_rmse, 1e-3) / max(float(np.mean(entropy[high])), 1e-3)

    def decompose(
        self,
        X: pd.DataFrame,
        model: Any,
        y_true: Optional[Sequence[float]] = None,
        rel_noise: float = 0.05,
        abs_noise: float = 0.0005,
    ) -> pd.DataFrame:
        mc = self.uq.compute_mc_uncertainty(X, rel_noise, abs_noise)
        entropy = np.asarray(self.entropy_fn(X), dtype=np.float64)
        if y_true is not None:
            self.kappa_ = self._calibrate_kappa(
                np.asarray(y_true, dtype=np.float64),
                np.asarray(model.predict(X), dtype=np.float64),
                entropy,
            )
        if self.kappa_ is None:
            self.kappa_ = 1.0
        structural = entropy * self.kappa_
        aleatoric = mc["Uncertainty_Std"].to_numpy(dtype=np.float64)
        total = np.sqrt(aleatoric**2 + structural**2)
        mean_depth = mc["Depth_Mean"].to_numpy(dtype=np.float64)
        return pd.DataFrame(
            {
                "Depth_Mean": mean_depth,
                "Sigma_Aleatoric": aleatoric,
                "Entropy": entropy,
                "Sigma_Structural": structural,
                "Sigma_Total": total,
                "CV_Total_Percent": 100.0 * total / np.clip(np.abs(mean_depth), 0.1, None),
            },
            index=X.index,
        )

# =============================================================================
# Registry and model-complexity ladder
# =============================================================================

class ModelTier(str, Enum):
    PHYSICAL = "physical"
    CLUSTERING = "clustering"
    MACHINE_LEARNING = "machine_learning"
    HETEROGENEOUS_ENSEMBLE = "heterogeneous_ensemble"
    NEURAL = "neural"


@dataclass
class ModelRegistryEntry:
    tier: ModelTier
    default_kwargs: Dict[str, Any] = field(default_factory=dict)
    requires: Optional[str] = None
    description: str = ""


MODEL_REGISTRY_METADATA: Dict[str, ModelRegistryEntry] = {
    "stumpf": ModelRegistryEntry(ModelTier.PHYSICAL, description="Stumpf band-ratio regression"),
    "lyzenga": ModelRegistryEntry(ModelTier.PHYSICAL, description="Lyzenga log-linear regression"),
    "caballero": ModelRegistryEntry(ModelTier.PHYSICAL, description="Adaptive Caballero/Stumpf blend"),
    "cbr": ModelRegistryEntry(ModelTier.CLUSTERING, {"n_clusters": 4}, description="Cluster-based local Stumpf models"),
    "gmm_soft_mixture": ModelRegistryEntry(ModelTier.CLUSTERING, {"n_components": 4}, description="Soft GMM physical mixture"),
    "random_forest": ModelRegistryEntry(ModelTier.MACHINE_LEARNING, description="Random forest regressor"),
    "extra_trees": ModelRegistryEntry(ModelTier.MACHINE_LEARNING, description="Extremely randomized trees"),
    "svr": ModelRegistryEntry(ModelTier.MACHINE_LEARNING, description="RBF support-vector regressor"),
    "mlp": ModelRegistryEntry(ModelTier.MACHINE_LEARNING, description="Multi-layer perceptron"),
    "xgboost": ModelRegistryEntry(ModelTier.MACHINE_LEARNING, requires="xgboost", description="XGBoost regressor"),
    "lightgbm": ModelRegistryEntry(ModelTier.MACHINE_LEARNING, requires="lightgbm", description="LightGBM regressor"),
    "catboost": ModelRegistryEntry(ModelTier.MACHINE_LEARNING, requires="catboost", description="CatBoost regressor"),
    "hetero_moe": ModelRegistryEntry(
        ModelTier.HETEROGENEOUS_ENSEMBLE,
        {"gating": "learned", "auto_select": True, "n_gate_repeats": 3, "gating_kwargs": {"shrinkage": 0.3}},
        description="Heterogeneous physical/ML mixture with OOF gate",
    ),
    "moe": ModelRegistryEntry(
        ModelTier.NEURAL,
        {"n_experts": 4, "epochs": 300, "patience": 25},
        requires="torch",
        description="Differentiable neural mixture of experts",
    ),
}

DEFAULT_MODEL_CANDIDATES: List[str] = list(MODEL_REGISTRY_METADATA)


class ModelRegistry:
    @staticmethod
    def is_registered(name: str) -> bool:
        return ModelFactory.is_registered(name)

    @staticmethod
    def canonical(name: str) -> str:
        return ModelFactory.canonical_name(name)

    @staticmethod
    def validate(candidates: Sequence[str]) -> Tuple[List[str], List[str]]:
        valid: List[str] = []
        invalid: List[str] = []
        seen: set = set()
        for candidate in candidates:
            canonical = ModelRegistry.canonical(candidate)
            if not ModelRegistry.is_registered(canonical):
                invalid.append(str(candidate))
            elif canonical not in seen:
                valid.append(canonical)
                seen.add(canonical)
        return valid, invalid

    @staticmethod
    def tier_of(name: str) -> ModelTier:
        canonical = ModelRegistry.canonical(name)
        entry = MODEL_REGISTRY_METADATA.get(canonical)
        return entry.tier if entry else ModelTier.MACHINE_LEARNING

    @staticmethod
    def default_kwargs(name: str) -> Dict[str, Any]:
        entry = MODEL_REGISTRY_METADATA.get(ModelRegistry.canonical(name))
        return dict(entry.default_kwargs) if entry else {}

    @staticmethod
    def describe() -> pd.DataFrame:
        rows = []
        for name, entry in MODEL_REGISTRY_METADATA.items():
            rows.append(
                {
                    "Model": name,
                    "Tier": entry.tier.value,
                    "Registered": ModelRegistry.is_registered(name),
                    "Requires": entry.requires or "-",
                    "Description": entry.description,
                }
            )
        return pd.DataFrame(rows).sort_values(["Tier", "Model"]).reset_index(drop=True)


class ModelComplexityLadder:
    """Run the same outer CV for every configured model and compare tiers."""

    def __init__(
        self,
        validator: SpatialValidator,
        evaluator: Optional[ModelEvaluator] = None,
        rung_configs: Optional[Mapping[str, Mapping[str, Any]]] = None,
        tiers: Optional[Sequence[ModelTier]] = None,
        alpha: float = 0.05,
    ) -> None:
        self.validator = validator
        self.evaluator = evaluator or ModelEvaluator()
        self.rung_configs = {
            name: ModelRegistry.default_kwargs(name)
            for name in DEFAULT_MODEL_CANDIDATES
        } if rung_configs is None else {
            ModelRegistry.canonical(name): dict(kwargs)
            for name, kwargs in rung_configs.items()
        }
        self.tiers = list(tiers) if tiers is not None else list(ModelTier)
        self.alpha = float(alpha)
        self._oof_by_model: Dict[str, np.ndarray] = {}
        self._y: Optional[np.ndarray] = None

    def run(
        self,
        frame: pd.DataFrame,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        self._y = frame["Depth"].to_numpy(dtype=np.float64)
        rows: Dict[str, Dict[str, Any]] = {}
        for name, params in self.rung_configs.items():
            try:
                result = self.validator.validate_model_full(
                    ModelFactory,
                    frame,
                    name,
                    params,
                    feature_cols=feature_cols,
                    evaluator=self.evaluator,
                )
                summary = result["summary"]
                self._oof_by_model[name] = result["first_run_oof"]
                rows[name] = {
                    "Tier": ModelRegistry.tier_of(name).value,
                    "RMSE_mean": summary["Repeated_RMSE_mean"],
                    "RMSE_std": summary["Repeated_RMSE_std"],
                    "R2_mean": summary["Repeated_R2_mean"],
                    "N_Repeats_Executed": summary["N_Repeats_Executed"],
                    "Stability_Status": summary["Stability_Status"],
                }
            except Exception as exc:
                logger.warning("Complexity rung %s failed: %s", name, exc)
                rows[name] = {
                    "Tier": ModelRegistry.tier_of(name).value,
                    "RMSE_mean": np.nan,
                    "RMSE_std": np.nan,
                    "R2_mean": np.nan,
                    "N_Repeats_Executed": 0,
                    "Stability_Status": "FAILED",
                }
        report = pd.DataFrame(rows).T
        if not report.empty:
            report = report.sort_values("RMSE_mean", na_position="last")
            report = self._add_significance_columns(report)
        return report

    def _add_significance_columns(self, report: pd.DataFrame) -> pd.DataFrame:
        report["p_value_vs_best"] = np.nan
        report["Indistinguishable_From_Best"] = None
        if self._y is None or wilcoxon is None or report.empty:
            return report
        valid_report = report[report["RMSE_mean"].notna()]
        if valid_report.empty:
            return report
        best_name = str(valid_report.index[0])
        best_prediction = self._oof_by_model.get(best_name)
        if best_prediction is None:
            return report
        for name in report.index:
            if name == best_name:
                report.loc[name, "Indistinguishable_From_Best"] = "--"
                continue
            prediction = self._oof_by_model.get(name)
            if prediction is None:
                continue
            mask = np.isfinite(prediction) & np.isfinite(best_prediction) & np.isfinite(self._y)
            if int(mask.sum()) < 8:
                continue
            difference = np.abs(prediction[mask] - self._y[mask]) - np.abs(best_prediction[mask] - self._y[mask])
            if np.allclose(difference, 0.0):
                p_value = 1.0
            else:
                try:
                    _, p_value = wilcoxon(difference)
                    p_value = float(p_value)
                except ValueError:
                    continue
            report.loc[name, "p_value_vs_best"] = p_value
            report.loc[name, "Indistinguishable_From_Best"] = bool(p_value >= self.alpha)
        return report

    def run_multi_rung(
        self,
        frame: pd.DataFrame,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        flat = self.run(frame, feature_cols=feature_cols)
        tier_reports: Dict[str, pd.DataFrame] = {}
        tier_champions: Dict[str, str] = {}
        for tier in self.tiers:
            subset = flat[flat["Tier"] == tier.value]
            if subset.empty:
                continue
            subset = subset.sort_values("RMSE_mean", na_position="last")
            tier_reports[tier.value] = subset
            valid = subset[subset["RMSE_mean"].notna()]
            if not valid.empty:
                tier_champions[tier.value] = str(valid.index[0])
        valid_flat = flat[flat["RMSE_mean"].notna()]
        overall = str(valid_flat.index[0]) if not valid_flat.empty else None
        return {
            "flat_report": flat,
            "tier_reports": tier_reports,
            "tier_champions": tier_champions,
            "overall_champion": overall,
        }

# =============================================================================
# Article-results workbook
# =============================================================================

def _style_article_workbook(workbook_path: Union[str, Path]) -> None:
    """Apply publication-friendly formatting to the article workbook."""
    try:
        from openpyxl import load_workbook  # type: ignore
        from openpyxl.formatting.rule import ColorScaleRule  # type: ignore
        from openpyxl.styles import Alignment, Font, PatternFill  # type: ignore
        from openpyxl.utils import get_column_letter  # type: ignore
    except ImportError:
        logger.warning("openpyxl is not installed; workbook styling was skipped")
        return
    path = Path(workbook_path)
    workbook = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    selected_fill = PatternFill("solid", fgColor="C6EFCE")
    selected_font = Font(color="006100", bold=True)
    for sheet in workbook.worksheets:
        sheet.sheet_view.showGridLines = False
        if sheet.max_row < 1 or sheet.max_column < 1:
            continue
        headers = [sheet.cell(row=1, column=column).value for column in range(1, sheet.max_column + 1)]
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        sheet.row_dimensions[1].height = 32
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="center", wrap_text=False)
                if isinstance(cell.value, (float, np.floating)) and np.isfinite(float(cell.value)):
                    cell.number_format = "0.0000"
        if "Selected_Final_Model" in headers:
            selected_column = headers.index("Selected_Final_Model") + 1
            for row in range(2, sheet.max_row + 1):
                value = sheet.cell(row=row, column=selected_column).value
                if value is True or str(value).lower() == "true":
                    for column in range(1, sheet.max_column + 1):
                        sheet.cell(row=row, column=column).fill = selected_fill
                        sheet.cell(row=row, column=column).font = selected_font
        # Apply a red-to-green scale to common error/stability columns.
        for column_index, header in enumerate(headers, start=1):
            if not isinstance(header, str):
                continue
            header_lower = header.lower()
            if any(token in header_lower for token in ("rmse", "mae", "mape", "cov")):
                color_rule = ColorScaleRule(
                    start_type="min",
                    start_color="F8696B",
                    mid_type="percentile",
                    mid_value=50,
                    mid_color="FFEB84",
                    end_type="max",
                    end_color="63BE7B",
                )
                letter = get_column_letter(column_index)
                if sheet.max_row >= 2:
                    sheet.conditional_formatting.add(
                        f"{letter}2:{letter}{sheet.max_row}", color_rule
                    )
        for column_index in range(1, sheet.max_column + 1):
            letter = get_column_letter(column_index)
            maximum = 0
            for cell in sheet[letter]:
                value = "" if cell.value is None else str(cell.value)
                maximum = max(maximum, len(value))
            sheet.column_dimensions[letter].width = min(max(maximum + 2, 12), 38)
    workbook.save(path)


def _write_article_results_workbook(
    workbook_path: Union[str, Path],
    cfg: SDBConfig,
    sampled_raw: pd.DataFrame,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_cols: Sequence[str],
    cv_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    comparison_frame: pd.DataFrame,
    seed_records: Sequence[Mapping[str, Any]],
    fold_records: Sequence[Mapping[str, Any]],
    oof_predictions: Mapping[str, np.ndarray],
    holdout_predictions_by_model: Mapping[str, np.ndarray],
    holdout_prediction_frame: pd.DataFrame,
    depth_bins_cv_best: pd.DataFrame,
    depth_bins_holdout_best: pd.DataFrame,
    uncertainty_frame: pd.DataFrame,
    best_name: str,
    evaluator: ModelEvaluator,
    ladder_frame: Optional[pd.DataFrame] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Create one workbook containing every table needed for the Results section.

    Workbook filename: ``Results of Article.xlsx``.

    The sheet order mirrors the recommended manuscript structure, especially
    Section 4.3 (comparative model performance) and the corresponding
    stability/holdout/depth/uncertainty analyses.
    """
    path = Path(workbook_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 01 - one-row-per-model comparative table for Section 4.3.
    display_names = {
        "stumpf": "Stumpf band-ratio",
        "lyzenga": "Lyzenga log-linear",
        "caballero": "Caballero-Stumpf adaptive",
        "cbr": "Cluster-based regression",
        "gmm_soft_mixture": "GMM soft mixture",
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
        "svr": "Support Vector Regression",
        "mlp": "Multi-Layer Perceptron",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "hetero_moe": "Heterogeneous Mixture of Experts",
        "moe": "Neural Mixture of Experts",
    }

    def method_label(model_name: str) -> str:
        return display_names.get(model_name, model_name)

    metric_rows: List[Dict[str, Any]] = []
    cv_rank = cv_frame["Repeated_RMSE_mean"].rank(method="min", na_option="keep")
    holdout_rank = (
        holdout_frame["Holdout_RMSE_Point"].rank(method="min", na_option="keep")
        if "Holdout_RMSE_Point" in holdout_frame.columns
        else pd.Series(dtype=float)
    )
    holdout_columns = [
        "Holdout_RMSE_Point",
        "Holdout_RMSE_BootMean",
        "Holdout_RMSE_BootSE",
        "Holdout_RMSE_95CI_Lower",
        "Holdout_RMSE_95CI_Upper",
        "Holdout_MAE_Point",
        "Holdout_MAE_BootMean",
        "Holdout_MAE_BootSE",
        "Holdout_MAE_95CI_Lower",
        "Holdout_MAE_95CI_Upper",
        "Holdout_R2_Point",
        "Holdout_R2_BootMean",
        "Holdout_R2_BootSE",
        "Holdout_R2_95CI_Lower",
        "Holdout_R2_95CI_Upper",
        "Holdout_Bias_Point",
        "Holdout_NRMSE_Point",
        "Holdout_CCC_Point",
        "Holdout_MAPE_Point",
    ]
    for model_name in cv_frame.index:
        cv_row = cv_frame.loc[model_name]
        holdout_row = holdout_frame.loc[model_name] if model_name in holdout_frame.index else pd.Series(dtype=object)
        row: Dict[str, Any] = {
            "Model": model_name,
            "Method": method_label(model_name),
            "Tier": ModelRegistry.tier_of(model_name).value,
            "Selected_Final_Model": bool(model_name == best_name),
            "CV_Rank": cv_rank.get(model_name, np.nan),
            "Holdout_Rank": holdout_rank.get(model_name, np.nan),
            "CV_RMSE_Mean": cv_row.get("Repeated_RMSE_mean", np.nan),
            "CV_RMSE_Std": cv_row.get("Repeated_RMSE_std", np.nan),
            "CV_MAE_Mean": cv_row.get("Repeated_MAE_mean", np.nan),
            "CV_MAE_Std": cv_row.get("Repeated_MAE_std", np.nan),
            "CV_R2_Mean": cv_row.get("Repeated_R2_mean", np.nan),
            "CV_R2_Std": cv_row.get("Repeated_R2_std", np.nan),
            "CV_Bias_Mean": cv_row.get("Repeated_Bias_mean", np.nan),
            "CV_NRMSE_Mean": cv_row.get("Repeated_NRMSE_mean", np.nan),
            "CV_CCC_Mean": cv_row.get("Repeated_CCC_mean", np.nan),
            "CV_MAPE_Mean": cv_row.get("Repeated_MAPE_mean", np.nan),
            "CV_Stability_COV_Percent": cv_row.get("RMSE_Stability_COV_Percent", np.nan),
            "Stability_Status": cv_row.get("Stability_Status", ""),
            "N_Repeats_Requested": cv_row.get("N_Repeats_Requested", np.nan),
            "N_Repeats_Executed": cv_row.get("N_Repeats_Executed", np.nan),
        }
        for column in holdout_columns:
            row[column] = holdout_row.get(column, np.nan)
        row["Holdout_RMSE_95CI"] = holdout_row.get("Holdout_RMSE_CI_str", "N/A")
        row["Holdout_R2_95CI"] = holdout_row.get("Holdout_R2_CI_str", "N/A")
        metric_rows.append(row)
    model_metrics = pd.DataFrame(metric_rows)

    # 02 - stability table, explicitly separated from raw performance.
    stability_rows = []
    for model_name in cv_frame.index:
        row = cv_frame.loc[model_name]
        executed_value = row.get("N_Repeats_Executed", 0)
        try:
            executed_count = int(executed_value) if np.isfinite(float(executed_value)) else 0
        except (TypeError, ValueError):
            executed_count = 0
        stability_rows.append(
            {
                "Model": model_name,
                "Method": method_label(model_name),
                "Tier": ModelRegistry.tier_of(model_name).value,
                "CV_Strategy": row.get("CV_Strategy", cfg.cv_strategy.upper()),
                "N_Repeats_Requested": row.get("N_Repeats_Requested", np.nan),
                "N_Repeats_Executed": row.get("N_Repeats_Executed", np.nan),
                "ARI_Mean": row.get("Clustering_ARI_Mean", np.nan),
                "ARI_Min": row.get("Clustering_ARI_Min", np.nan),
                "Blocks_Verified_Deterministic": row.get(
                    "Clustering_Verified_Deterministic", np.nan
                ),
                "RMSE_Mean": row.get("Repeated_RMSE_mean", np.nan),
                "RMSE_Std": row.get("Repeated_RMSE_std", np.nan),
                "RMSE_COV_Percent": row.get("RMSE_Stability_COV_Percent", np.nan),
                "R2_Mean": row.get("Repeated_R2_mean", np.nan),
                "R2_Std": row.get("Repeated_R2_std", np.nan),
                "MAE_Mean": row.get("Repeated_MAE_mean", np.nan),
                "MAE_Std": row.get("Repeated_MAE_std", np.nan),
                "CCC_Mean": row.get("Repeated_CCC_mean", np.nan),
                "CCC_Std": row.get("Repeated_CCC_std", np.nan),
                "Stability_Status": row.get("Stability_Status", ""),
                "Interpretation": (
                    "10 model-seed repetitions on fixed spatial blocks"
                    if executed_count > 1
                    and bool(row.get("Clustering_Verified_Deterministic", False))
                    else "Repeated spatial/model-seed assessment"
                ),
            }
        )
    model_stability = pd.DataFrame(stability_rows)

    # 03 - complete bootstrap table.
    holdout_bootstrap = holdout_frame.reset_index().rename(columns={"index": "Model"})
    if "Model" not in holdout_bootstrap.columns:
        holdout_bootstrap.insert(0, "Model", holdout_frame.index)
    tier_values = holdout_bootstrap["Model"].map(
        lambda value: ModelRegistry.tier_of(value).value
    )
    if "Tier" in holdout_bootstrap.columns:
        holdout_bootstrap["Tier"] = tier_values
    else:
        holdout_bootstrap.insert(1, "Tier", tier_values)
    method_values = holdout_bootstrap["Model"].map(method_label)
    if "Method" in holdout_bootstrap.columns:
        holdout_bootstrap["Method"] = method_values
    else:
        holdout_bootstrap.insert(2, "Method", method_values)
    selected_values = holdout_bootstrap["Model"].eq(best_name)
    if "Selected_Final_Model" in holdout_bootstrap.columns:
        holdout_bootstrap["Selected_Final_Model"] = selected_values
    else:
        holdout_bootstrap.insert(3, "Selected_Final_Model", selected_values)

    # 04/05 - depth-stratified results for every successful model.
    depth_cv_tables: List[pd.DataFrame] = []
    for model_name, prediction in oof_predictions.items():
        valid = np.isfinite(prediction) & np.isfinite(
            train_frame["Depth"].to_numpy(dtype=np.float64)
        )
        if not np.any(valid):
            continue
        table = evaluator.depth_bin_analysis(
            train_frame.loc[valid, "Depth"].to_numpy(dtype=np.float64),
            prediction[valid],
            bins=cfg.depth_bins,
        ).reset_index()
        table.insert(0, "Model", model_name)
        table.insert(1, "Method", method_label(model_name))
        table.insert(2, "Tier", ModelRegistry.tier_of(model_name).value)
        table.insert(3, "Evaluation", "Spatial CV OOF")
        depth_cv_tables.append(table)
    depth_cv_all = (
        pd.concat(depth_cv_tables, ignore_index=True)
        if depth_cv_tables
        else depth_bins_cv_best.copy()
    )
    if not depth_bins_cv_best.empty:
        depth_cv_best = depth_bins_cv_best.reset_index()
        depth_cv_best.insert(0, "Model", best_name)
        depth_cv_best.insert(1, "Method", method_label(best_name))
        depth_cv_best.insert(2, "Tier", ModelRegistry.tier_of(best_name).value)
        depth_cv_best.insert(3, "Evaluation", "Selected model")
    else:
        depth_cv_best = pd.DataFrame()
    depth_holdout_tables: List[pd.DataFrame] = []
    true_holdout = test_frame["Depth"].to_numpy(dtype=np.float64)
    for model_name, prediction in holdout_predictions_by_model.items():
        valid = np.isfinite(prediction) & np.isfinite(true_holdout)
        if not np.any(valid):
            continue
        table = evaluator.depth_bin_analysis(
            true_holdout[valid], prediction[valid], bins=cfg.depth_bins
        ).reset_index()
        table.insert(0, "Model", model_name)
        table.insert(1, "Method", method_label(model_name))
        table.insert(2, "Tier", ModelRegistry.tier_of(model_name).value)
        table.insert(3, "Evaluation", "Independent holdout")
        depth_holdout_tables.append(table)
    depth_holdout_all = (
        pd.concat(depth_holdout_tables, ignore_index=True)
        if depth_holdout_tables
        else depth_bins_holdout_best.copy()
    )

    # 06 - residuals for the selected model; 07 - predictions for all models.
    predictions_all = test_frame[["ID", "Lon", "Lat", "Depth"]].copy()
    predictions_all = predictions_all.rename(columns={"Depth": "True_Depth"})
    for model_name, prediction in holdout_predictions_by_model.items():
        predictions_all[f"{model_name}_Pred_Depth"] = prediction
        predictions_all[f"{model_name}_Residual"] = prediction - predictions_all["True_Depth"]
        predictions_all[f"{model_name}_Absolute_Error"] = np.abs(
            predictions_all[f"{model_name}_Residual"]
        )

    # 08 - validation design summary.
    first_cv_row = cv_frame.loc[best_name] if best_name in cv_frame.index else pd.Series(dtype=object)
    validation_summary = pd.DataFrame(
        [
            {"Parameter": "Framework_Version", "Value": __version__},
            {"Parameter": "Total_Valid_Samples", "Value": len(sampled_raw)},
            {"Parameter": "Training_Samples", "Value": len(train_frame)},
            {"Parameter": "Holdout_Samples", "Value": len(test_frame)},
            {"Parameter": "Actual_Holdout_Percent", "Value": 100.0 * len(test_frame) / max(len(sampled_raw), 1)},
            {"Parameter": "CV_Strategy", "Value": cfg.cv_strategy.upper()},
            {"Parameter": "Spatial_Blocks", "Value": cfg.n_spatial_blocks},
            {"Parameter": "Spatial_CV_Folds", "Value": cfg.spatial_cv_folds},
            {"Parameter": "Repeats_Requested", "Value": cfg.n_repeats},
            {"Parameter": "Repeats_Executed_Selected_Model", "Value": first_cv_row.get("N_Repeats_Executed", np.nan)},
            {"Parameter": "ARI_Min_Selected_Model", "Value": first_cv_row.get("Clustering_ARI_Min", np.nan)},
            {"Parameter": "Number_of_Features", "Value": len(feature_cols)},
            {"Parameter": "Reflectance_Scale", "Value": getattr(cfg, "reflectance_scale", None) or "Auto-detected"},
            {"Parameter": "Design_Max_Depth_m", "Value": 3.0},
            {"Parameter": "Depth_Bins_m", "Value": list(cfg.depth_bins)},
            {"Parameter": "Selected_Final_Model", "Value": best_name},
            {"Parameter": "Primary_Selection_Rule", "Value": "Lowest repeated spatial-CV RMSE; R2/CCC tie-break"},
        ]
    )

    # 09 - all valid sampled points and their reflectance values.
    field_samples = sampled_raw.copy()

    # 10 - MC uncertainty for the independent holdout.
    uncertainty_holdout = uncertainty_frame.copy()
    if not uncertainty_holdout.empty:
        metadata = test_frame[["ID", "Lon", "Lat", "Depth"]].reset_index(drop=True)
        uncertainty_holdout = pd.concat(
            [metadata.rename(columns={"Depth": "True_Depth"}), uncertainty_holdout.reset_index(drop=True)],
            axis=1,
        )

    # 11 - complexity ladder, if requested.
    if ladder_frame is None or ladder_frame.empty:
        ladder_frame = pd.DataFrame(
            [{"Status": "Not requested; run with run_complexity_ladder=True"}]
        )

    # 12 - useful output paths and artifact summary.
    artifact_rows = []
    for key, value in cfg.to_dict().items():
        artifact_rows.append({"Category": "Configuration", "Name": key, "Value": _jsonable(value)})
    artifact_rows.append({"Category": "Selection", "Name": "Best_Model", "Value": best_name})
    for key, value in (artifacts or {}).items():
        if key == "saved_metrics_files" and isinstance(value, Mapping):
            for file_key, file_value in value.items():
                artifact_rows.append(
                    {
                        "Category": "Output_File",
                        "Name": file_key,
                        "Value": _jsonable(file_value),
                    }
                )
        elif key not in {"model_selection_cv", "complexity_ladder"}:
            artifact_rows.append(
                {
                    "Category": "Artifact",
                    "Name": key,
                    "Value": _jsonable(value),
                }
            )
    readme = pd.DataFrame(
        [
            {"Article_Section": "4.1 Data quality and sampling", "Workbook_Sheet": "09_Field_Samples", "Purpose": "Valid field observations and sampled reflectance"},
            {"Article_Section": "4.2 Spatial validation", "Workbook_Sheet": "08_Validation", "Purpose": "Train/holdout design, CV strategy and ARI"},
            {"Article_Section": "4.3 Comparative model performance", "Workbook_Sheet": "01_Model_Metrics", "Purpose": "One row per method with CV and holdout metrics"},
            {"Article_Section": "4.4 Bootstrap holdout evaluation", "Workbook_Sheet": "03_Holdout_Bootstrap", "Purpose": "Point estimates, SE and 95% confidence intervals"},
            {"Article_Section": "4.5 Depth-stratified performance", "Workbook_Sheet": "04_Depth_CV_OOF / 05_Depth_Holdout", "Purpose": "Metrics by depth interval"},
            {"Article_Section": "4.6 Residual analysis", "Workbook_Sheet": "06_Residuals_Best / 07_Predictions_All", "Purpose": "Residuals, absolute errors and outlier flags"},
            {"Article_Section": "4.7 Model stability", "Workbook_Sheet": "02_Model_Stability", "Purpose": "Repeated RMSE/R2/MAE/CCC, COV and stability status"},
            {"Article_Section": "4.8 Uncertainty and final product", "Workbook_Sheet": "10_Uncertainty_Holdout / raster outputs", "Purpose": "Monte-Carlo uncertainty and final map support"},
            {"Article_Section": "4.9 Sensitivity analysis", "Workbook_Sheet": "plots/ + metrics/*_sensitivity_*", "Purpose": "OAT, noise-robustness, permutation importance figures (Palatino Linotype, 10 pt)"},
        ]
    )
    path.unlink(missing_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="00_Readme", index=False)
        model_metrics.to_excel(writer, sheet_name="01_Model_Metrics", index=False)
        model_stability.to_excel(writer, sheet_name="02_Model_Stability", index=False)
        holdout_bootstrap.to_excel(writer, sheet_name="03_Holdout_Bootstrap", index=False)
        depth_cv_all.to_excel(writer, sheet_name="04_Depth_CV_OOF", index=False)
        depth_holdout_all.to_excel(writer, sheet_name="05_Depth_Holdout", index=False)
        holdout_prediction_frame.to_excel(writer, sheet_name="06_Residuals_Best", index=False)
        predictions_all.to_excel(writer, sheet_name="07_Predictions_All", index=False)
        validation_summary.to_excel(writer, sheet_name="08_Validation", index=False)
        field_samples.to_excel(writer, sheet_name="09_Field_Samples", index=False)
        uncertainty_holdout.to_excel(writer, sheet_name="10_Uncertainty_Holdout", index=False)
        ladder_frame.to_excel(writer, sheet_name="11_Complexity_Ladder", index=False)
        pd.DataFrame(artifact_rows).to_excel(writer, sheet_name="12_Configuration", index=False)
        pd.DataFrame(seed_records).to_excel(writer, sheet_name="13_Repeated_Seeds", index=False)
        pd.DataFrame(fold_records).to_excel(writer, sheet_name="14_CV_Folds", index=False)
        comparison_frame.reset_index().to_excel(writer, sheet_name="15_CV_vs_Holdout", index=False)
    _style_article_workbook(path)
    logger.info("Article results workbook written: %s", path)
    return path

# =============================================================================
# RSE publication figures: sensitivity and uncertainty analysis
# =============================================================================

def _configure_publication_style() -> None:
    """Apply the RSE journal figure style: Palatino Linotype, 10 pt text.

    The font is used silently whenever it is installed on the machine;
    otherwise the closest available serif font is selected.  Matplotlib
    font-matching warnings (``findfont``) are suppressed so the console
    stays clean.
    """
    import matplotlib  # type: ignore
    try:
        matplotlib.use("Agg")
    except Exception:  # pragma: no cover - backend already set
        pass
    from matplotlib import font_manager  # type: ignore
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_candidates = [
        "Palatino Linotype",
        "Palatino",
        "Book Antiqua",
        "URW Palladio L",
    ]
    font_family = [name for name in font_candidates if name in available_fonts]
    if not font_family:
        font_family = ["DejaVu Serif"]
    font_family = list(font_family) + ["serif"]
    # Never let matplotlib print "findfont: Font family ... not found".
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    matplotlib.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.1,
            "lines.markersize": 4,
            "figure.dpi": 110,
            "savefig.dpi": 1000,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "mathtext.fontset": "stix",
        }
    )


class RSEPublicationPlotter:
    """Generate every sensitivity/uncertainty figure required by an RSE paper.

    All figures are written to ``cfg.output_dir / "plots"``, each as a
    300-dpi PNG and a PDF, using the Palatino Linotype 10-pt style.
    """

    def __init__(self, cfg: SDBConfig, top_n_features: int = 5) -> None:
        self.cfg = cfg
        # Number of most-influential features shown in the sensitivity and
        # importance bar charts; the full tables are still saved as CSV.
        self.top_n_features = int(top_n_features)
        self.plots_dir = Path(cfg.output_dir) / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        try:
            _configure_publication_style()
            import matplotlib.pyplot as plt  # type: ignore
            self._plt = plt
        except Exception as exc:  # pragma: no cover
            raise ImportError("RSE publication plots require matplotlib") from exc

    # ------------------------------------------------------------------ save
    def _save(self, figure: Any, stem: str) -> Tuple[Path, Path]:
        png_path = self.plots_dir / f"{stem}.png"
        pdf_path = self.plots_dir / f"{stem}.pdf"
        figure.savefig(png_path, dpi=1000)
        figure.savefig(pdf_path)
        self._plt.close(figure)
        logger.info("Figure saved: %s (+ .pdf)", png_path.name)
        return png_path, pdf_path

    # ---------------------------------------------- 01 OAT sensitivity
    def plot_oat_sensitivity(self, sensitivity_frame: pd.DataFrame) -> Optional[Tuple[Path, Path]]:
        if sensitivity_frame is None or sensitivity_frame.empty:
            return None
        plt = self._plt
        # Plot only the most influential features so the figure stays
        # readable; the full table is still written as CSV.
        frame = sensitivity_frame.sort_values(
            "Absolute_Impact", ascending=False
        ).head(self.top_n_features)
        frame = frame.sort_values("Sensitivity_Index", ascending=True)
        figure, axis = plt.subplots(
            figsize=(3.6, max(2.2, 0.30 * len(frame) + 0.9))
        )
        colors = ["#C0392B" if value < 0 else "#1F618D" for value in frame["Sensitivity_Index"]]
        axis.barh(frame["Feature"], frame["Sensitivity_Index"], color=colors,
                  edgecolor="black", linewidth=0.4)
        axis.axvline(0.0, color="black", linewidth=0.7)
        axis.set_xlabel("Sensitivity index (relative output change per 5% input change)")
        axis.set_title("One-at-a-time sensitivity of the final model")
        figure.tight_layout()
        return self._save(figure, "01_sensitivity_oat")

    # ---------------------------------------------- 02 noise robustness
    def plot_noise_robustness(self, noise_frame: pd.DataFrame) -> Optional[Tuple[Path, Path]]:
        if noise_frame is None or noise_frame.empty:
            return None
        plt = self._plt
        figure, axis = plt.subplots(figsize=(3.6, 2.7))
        bars = axis.bar(
            noise_frame["Noise_Level"],
            noise_frame["RMSE_Drift"],
            color="#1F618D",
            edgecolor="black",
            linewidth=0.4,
        )
        axis.set_xlabel("Input noise level (% of feature s.d.)")
        axis.set_ylabel("RMSE drift (m)")
        for bar, value in zip(bars, noise_frame["RMSE_Drift"]):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3f}",
                ha="center", va="bottom", fontsize=10,
            )
        axis.set_title("Noise robustness of the final model")
        figure.tight_layout()
        return self._save(figure, "02_sensitivity_noise_robustness")

    # ---------------------------------------------- 03 permutation importance
    def plot_permutation_importance(self, importance_frame: pd.DataFrame) -> Optional[Tuple[Path, Path]]:
        if importance_frame is None or importance_frame.empty:
            return None
        plt = self._plt
        # Plot only the most important features so the figure stays readable.
        frame = importance_frame.sort_values(
            "Importance_Mean", ascending=False
        ).head(self.top_n_features).sort_values("Importance_Mean", ascending=True)
        figure, axis = plt.subplots(
            figsize=(3.6, max(2.2, 0.30 * len(frame) + 0.9))
        )
        axis.barh(
            frame["Feature"],
            frame["Importance_Mean"],
            xerr=frame["Importance_Std"],
            color="#16A085",
            edgecolor="black",
            linewidth=0.4,
            capsize=2.0,
        )
        axis.set_xlabel("Permutation importance (RMSE increase, m)")
        axis.set_title("Feature importance of the final model")
        figure.tight_layout()
        return self._save(figure, "03_sensitivity_permutation_importance")

    # ---------------------------------------------- 04 MC uncertainty
    def plot_mc_uncertainty(
        self,
        uncertainty_frame: pd.DataFrame,
        y_true: Optional[Sequence[float]] = None,
    ) -> Optional[Tuple[Path, Path]]:
        if uncertainty_frame is None or uncertainty_frame.empty:
            return None
        plt = self._plt
        figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
        standard_deviation = uncertainty_frame["Uncertainty_Std"].to_numpy(dtype=np.float64)
        finite_sigma = standard_deviation[np.isfinite(standard_deviation)]
        if finite_sigma.size == 0:
            self._plt.close(figure)
            return None
        axes[0].hist(finite_sigma, bins=20, color="#1F618D",
                     edgecolor="white", linewidth=0.4)
        axes[0].set_xlabel("MC uncertainty std (m)")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Aleatoric uncertainty distribution")
        mean_depth = uncertainty_frame["Depth_Mean"].to_numpy(dtype=np.float64)
        valid = np.isfinite(mean_depth) & np.isfinite(standard_deviation)
        if np.any(valid):
            axes[1].scatter(mean_depth[valid], standard_deviation[valid],
                            s=9, color="#16A085", edgecolor="black", linewidth=0.2)
        axes[1].set_xlabel("MC mean predicted depth (m)")
        axes[1].set_ylabel("MC uncertainty std (m)")
        axes[1].set_title("Uncertainty versus predicted depth")
        figure.tight_layout()
        return self._save(figure, "04_uncertainty_monte_carlo")

    # ---------------------------------------------- 05 observed vs predicted
    def plot_observed_vs_predicted(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        metrics: Optional[Mapping[str, float]] = None,
    ) -> Optional[Tuple[Path, Path]]:
        true, pred = _finite_pair(y_true, y_pred)
        if true.size == 0:
            return None
        plt = self._plt
        figure, axis = plt.subplots(figsize=(3.6, 3.4))
        axis.scatter(true, pred, s=14, color="#1F618D",
                     edgecolor="black", linewidth=0.3, alpha=0.85)
        lower = min(float(np.min(true)), float(np.min(pred)), 0.0)
        upper = max(float(np.max(true)), float(np.max(pred)))
        axis.plot([lower, upper], [lower, upper], color="#C0392B",
                  linestyle="--", linewidth=1.1, label="1:1 line")
        axis.set_xlabel("Observed depth (m)")
        axis.set_ylabel("Predicted depth (m)")
        axis.set_title("Observed versus predicted depth")
        annotation = []
        if metrics is not None:
            if np.isfinite(metrics.get("R2", np.nan)):
                annotation.append(f"$R^2$ = {metrics['R2']:.3f}")
            if np.isfinite(metrics.get("RMSE", np.nan)):
                annotation.append(f"RMSE = {metrics['RMSE']:.3f} m")
            if np.isfinite(metrics.get("CCC", np.nan)):
                annotation.append(f"CCC = {metrics['CCC']:.3f}")
        annotation.append(f"n = {true.size}")
        axis.text(
            0.05, 0.95, "\n".join(annotation), transform=axis.transAxes,
            va="top", ha="left", fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                  "edgecolor": "0.6", "linewidth": 0.5},
        )
        figure.tight_layout()
        return self._save(figure, "05_observed_vs_predicted")

    # ---------------------------------------------- 06 residuals
    def plot_residuals(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
    ) -> Optional[Tuple[Path, Path]]:
        true, pred = _finite_pair(y_true, y_pred)
        if true.size == 0:
            return None
        plt = self._plt
        figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.0),
                                    gridspec_kw={"width_ratios": [2.0, 1.0]})
        residual = pred - true
        axes[0].scatter(true, residual, s=14, color="#16A085",
                        edgecolor="black", linewidth=0.3, alpha=0.85)
        axes[0].axhline(0.0, color="#C0392B", linestyle="--", linewidth=1.1)
        axes[0].set_xlabel("Observed depth (m)")
        axes[0].set_ylabel("Residual (predicted - observed, m)")
        axes[0].set_title("Residuals versus depth")
        axes[1].hist(residual, bins=20, orientation="horizontal",
                     color="#1F618D", edgecolor="white", linewidth=0.4)
        axes[1].axhline(0.0, color="#C0392B", linestyle="--", linewidth=1.1)
        axes[1].set_xlabel("Frequency")
        axes[1].set_title("Residual distribution")
        figure.tight_layout()
        return self._save(figure, "06_residuals")

    # ---------------------------------------------- 07 depth-bin error profile
    def plot_depth_bin_metrics(self, depth_bin_frame: pd.DataFrame) -> Optional[Tuple[Path, Path]]:
        if depth_bin_frame is None or depth_bin_frame.empty:
            return None
        plt = self._plt
        frame = depth_bin_frame.reset_index()
        bin_column = [column for column in frame.columns if column == "Depth_Range"] or [frame.columns[0]]
        labels = frame[bin_column[0]].astype(str)
        positions = np.arange(len(frame))
        width = 0.38
        figure, axis = plt.subplots(figsize=(4.6, 3.0))
        axis.bar(positions - width / 2.0, frame["RMSE"], width,
                 color="#1F618D", edgecolor="black", linewidth=0.4, label="RMSE")
        axis.bar(positions + width / 2.0, frame["MAE"], width,
                 color="#16A085", edgecolor="black", linewidth=0.4, label="MAE")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=30, ha="right")
        axis.set_xlabel("Depth bin (m)")
        axis.set_ylabel("Error (m)")
        axis.set_title("Depth-stratified error profile")
        axis.legend()
        figure.tight_layout()
        return self._save(figure, "07_depth_bin_error_profile")

    # ---------------------------------------------- all figures at once
    def plot_all(
        self,
        sensitivity_frame: Optional[pd.DataFrame] = None,
        noise_frame: Optional[pd.DataFrame] = None,
        importance_frame: Optional[pd.DataFrame] = None,
        uncertainty_frame: Optional[pd.DataFrame] = None,
        y_true: Optional[Sequence[float]] = None,
        y_pred: Optional[Sequence[float]] = None,
        metrics: Optional[Mapping[str, float]] = None,
        depth_bin_frame: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Tuple[Path, Path]]:
        """Produce the complete RSE figure set and return path manifests."""
        saved: Dict[str, Tuple[Path, Path]] = {}
        candidates = {
            "sensitivity_oat": self.plot_oat_sensitivity(sensitivity_frame),
            "noise_robustness": self.plot_noise_robustness(noise_frame),
            "permutation_importance": self.plot_permutation_importance(importance_frame),
            "monte_carlo_uncertainty": (
                self.plot_mc_uncertainty(uncertainty_frame, y_true)
            ),
            "observed_vs_predicted": (
                self.plot_observed_vs_predicted(y_true, y_pred, metrics)
                if y_true is not None and y_pred is not None else None
            ),
            "residuals": (
                self.plot_residuals(y_true, y_pred)
                if y_true is not None and y_pred is not None else None
            ),
            "depth_bin_error_profile": self.plot_depth_bin_metrics(depth_bin_frame),
        }
        for name, paths in candidates.items():
            if paths is not None:
                saved[name] = paths
        return saved

# =============================================================================
# Operational pipeline
# =============================================================================

class SDBOperationalPipeline:
    """End-to-end leakage-safe SDB training, evaluation and deployment."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        cfg: Optional[SDBConfig] = None,
    ) -> None:
        self.start_time = time.time()
        if cfg is not None:
            self.cfg = cfg
        elif config_path is not None:
            self.cfg = SDBConfig.from_yaml(config_path)
        else:
            self.cfg = SDBConfig()
        self._setup_workspace()
        logger.info("SDB Operational Pipeline v%s ready", __version__)

    def _setup_workspace(self) -> None:
        for subdirectory in (
            "models",
            "rasters",
            "metrics",
            "plots",
            "logs",
            "reports",
        ):
            (self.cfg.output_dir / subdirectory).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _build_feature_dataset(
        frame: pd.DataFrame,
        feature_engineer: FeatureEngineer,
    ) -> pd.DataFrame:
        return feature_engineer.build_dataset(frame)

    def _model_params(self, model_name: str) -> Dict[str, Any]:
        """Return registry defaults plus pipeline-level reproducibility settings."""
        canonical = ModelRegistry.canonical(model_name)
        params = ModelRegistry.default_kwargs(canonical)
        # All stochastic model wrappers in this module expose random_state.
        if canonical in {
            "cbr",
            "gmm_soft_mixture",
            "hetero_moe",
            "moe",
            "random_forest",
            "extra_trees",
            "mlp",
            "xgboost",
            "lightgbm",
            "catboost",
        }:
            params.setdefault("random_state", self.cfg.random_seed)
        if canonical in {"random_forest", "extra_trees", "xgboost", "lightgbm"}:
            params.setdefault("n_jobs", self.cfg.n_jobs)
        return params

    @staticmethod
    def _select_best_model(cv_frame: pd.DataFrame) -> str:
        valid = cv_frame[
            cv_frame["Repeated_RMSE_mean"].notna()
            & cv_frame["Repeated_R2_mean"].notna()
        ].copy()
        if valid.empty:
            valid = cv_frame[cv_frame["Repeated_RMSE_mean"].notna()].copy()
        if valid.empty:
            raise RuntimeError("No model has a finite cross-validation RMSE")
        best_rmse = float(valid["Repeated_RMSE_mean"].min())
        tolerance = max(abs(best_rmse) * 0.02, 1e-9)
        candidates = valid[valid["Repeated_RMSE_mean"] <= best_rmse + tolerance]
        candidates = candidates.sort_values(
            by=["Repeated_R2_mean", "Repeated_CCC_mean", "Repeated_RMSE_mean"],
            ascending=[False, False, True],
            na_position="last",
        )
        return str(candidates.index[0])

    def run_full_pipeline(
        self,
        model_candidates: Optional[Sequence[str]] = None,
        run_hpo: bool = False,
        compute_uncertainty_raster: bool = True,
        run_complexity_ladder: bool = False,
    ) -> Dict[str, Any]:
        """Execute all phases and return a JSON-serializable artifact manifest."""
        cfg = self.cfg
        reports_dir = cfg.output_dir / "reports"
        metrics_dir = cfg.output_dir / "metrics"
        rasters_dir = cfg.output_dir / "rasters"
        models_dir = cfg.output_dir / "models"
        artifacts: Dict[str, Any] = {}
        logger.info("=" * 80)
        logger.info("SDB FULL PIPELINE START | v%s", __version__)
        logger.info("=" * 80)

        # ------------------------------------------------------------------
        # Phase 1: ingest raw field observations and sample the scene.
        # ------------------------------------------------------------------
        sampler = RasterSampler(cfg.raster_path, cfg.band_order, cfg.points_crs)
        field_frame = sampler.read_field_data(cfg.excel_path, cfg)
        sampled_raw = sampler.sample_at_points(
            field_frame,
            scale=cfg.reflectance_scale if cfg.reflectance_scale is not None else "auto",
        )
        if len(sampled_raw) < cfg.min_valid_samples:
            raise RuntimeError(
                f"Only {len(sampled_raw)} valid samples remain; at least "
                f"{cfg.min_valid_samples} are required. Check CRS, raster extent "
                "and masks."
            )
        sampled_raw.to_csv(reports_dir / "sampled_points.csv", index=False)
        artifacts.update(
            {
                "n_field_points": int(len(field_frame)),
                "n_valid_samples": int(len(sampled_raw)),
                "reflectance_scale": sampler.reflectance_scale_,
            }
        )

        # ------------------------------------------------------------------
        # Phase 2: spatial holdout BEFORE fitting any batch-dependent feature
        # statistics. This is the central leakage fix in the rewrite.
        # ------------------------------------------------------------------
        validator = SpatialValidator(
            n_blocks=cfg.n_spatial_blocks,
            n_folds=cfg.spatial_cv_folds,
            random_seed=cfg.random_seed,
            cv_strategy=cfg.cv_strategy,
            n_repeats=cfg.n_repeats,
        )
        train_raw, test_raw = validator.split_holdout(
            sampled_raw, cfg.holdout_fraction
        )
        train_fe = FeatureEngineer(
            cfg.band_order,
            cfg.n_const,
            cfg.deep_water_percentile,
            cfg.eps,
        ).fit(train_raw)
        train_fe.reflectance_scale_ = sampler.reflectance_scale_
        train_frame = self._build_feature_dataset(train_raw, train_fe)
        test_frame = self._build_feature_dataset(test_raw, train_fe)
        feature_cols = train_fe.feature_names
        train_frame.to_csv(reports_dir / "training_engineered_dataset.csv", index=False)
        test_frame.to_csv(reports_dir / "holdout_engineered_dataset.csv", index=False)
        artifacts["n_features"] = len(feature_cols)
        artifacts["n_train"] = int(len(train_frame))
        artifacts["n_holdout"] = int(len(test_frame))

        # ------------------------------------------------------------------
        # Phase 3: validate candidate registry and run the tournament.
        # ------------------------------------------------------------------
        requested = list(model_candidates) if model_candidates is not None else list(DEFAULT_MODEL_CANDIDATES)
        valid_models, invalid_models = ModelRegistry.validate(requested)
        if invalid_models:
            logger.warning("Unknown model candidates dropped: %s", invalid_models)
        if not valid_models:
            raise RuntimeError("No valid model candidates remain")
        artifacts["model_candidates_requested"] = requested
        artifacts["model_candidates_invalid"] = invalid_models
        artifacts["model_candidates_valid"] = valid_models
        evaluator = ModelEvaluator()
        bootstrap = BootstrapEvaluator(
            n_bootstrap=cfg.n_bootstrap_ci,
            random_seed=cfg.random_seed,
            evaluator=evaluator,
        )
        repeated_summaries: Dict[str, Dict[str, Any]] = {}
        seed_records: List[Dict[str, Any]] = []
        fold_records: List[Dict[str, Any]] = []
        oof_predictions: Dict[str, np.ndarray] = {}
        holdout_records: Dict[str, Dict[str, Any]] = {}
        holdout_distributions: Dict[str, Dict[str, np.ndarray]] = {}
        holdout_predictions_by_model: Dict[str, np.ndarray] = {}
        model_errors: Dict[str, str] = {}
        X_train = train_frame[feature_cols]
        y_train = train_frame["Depth"].to_numpy(dtype=np.float64)
        X_test = test_frame[feature_cols]
        y_test = test_frame["Depth"].to_numpy(dtype=np.float64)
        for name in valid_models:
            params = self._model_params(name)
            try:
                summary, seeds, folds, oof = validator.evaluate_repeated_cv(
                    ModelFactory,
                    train_frame,
                    name,
                    params=params,
                    feature_cols=feature_cols,
                    evaluator=evaluator,
                )
                candidate = ModelFactory.create(name, **params)
                candidate.fit(X_train, y_train)
                train_prediction = candidate.predict(X_train)
                test_prediction = candidate.predict(X_test)
                train_metrics = evaluator.calculate_metrics(y_train, train_prediction)
                holdout_summary, distribution = bootstrap.evaluate_bootstrap_ci(
                    y_test, test_prediction
                )
                repeated_summaries[name] = summary
                seed_records.extend(seeds)
                fold_records.extend(folds)
                oof_predictions[name] = oof
                record: Dict[str, Any] = {
                    "Model": name,
                    "Tier": ModelRegistry.tier_of(name).value,
                    "N_Train": len(y_train),
                    "N_Holdout": len(y_test),
                    "N_Bootstrap_Resamples": cfg.n_bootstrap_ci,
                    **holdout_summary,
                }
                record.update({f"Train_{key}": value for key, value in train_metrics.items()})
                holdout_records[name] = record
                holdout_distributions[name] = distribution
                holdout_predictions_by_model[name] = test_prediction.copy()
                logger.info(
                    "Model | %-18s | CV RMSE=%.4f +/- %.4f | Holdout RMSE=%s",
                    name.upper(),
                    summary["Repeated_RMSE_mean"],
                    summary["Repeated_RMSE_std"],
                    holdout_summary["Holdout_RMSE_CI_str"],
                )
            except Exception as exc:
                model_errors[name] = f"{type(exc).__name__}: {exc}"
                logger.warning("Model %s failed and was skipped: %s", name, exc)
        if not repeated_summaries or not holdout_records:
            raise RuntimeError(
                "All candidate models failed. Errors: " + json.dumps(model_errors, default=str)
            )
        cv_frame = pd.DataFrame(repeated_summaries).T
        cv_frame.index.name = "Model"
        cv_frame = cv_frame.sort_values("Repeated_RMSE_mean", na_position="last")
        holdout_frame = pd.DataFrame(holdout_records.values()).set_index("Model")
        holdout_frame = holdout_frame.reindex(cv_frame.index)
        # Select only from models that completed BOTH the outer CV and the
        # independent holdout fit.  A model that passed CV but failed its
        # holdout fit must never become the deployment winner by accident.
        common_models = cv_frame.index.intersection(holdout_records.keys())
        selection_frame = cv_frame.loc[common_models]
        best_name = self._select_best_model(selection_frame)
        comparison_rows: List[Dict[str, Any]] = []
        for name in cv_frame.index:
            if name not in holdout_frame.index or pd.isna(holdout_frame.loc[name, "Holdout_RMSE_Point"]):
                continue
            cv_row = cv_frame.loc[name]
            holdout_row = holdout_frame.loc[name]
            comparison_rows.append(
                {
                    "Model": name,
                    "Tier": ModelRegistry.tier_of(name).value,
                    "Repeated_CV_RMSE_mean": cv_row["Repeated_RMSE_mean"],
                    "Repeated_CV_RMSE_std": cv_row["Repeated_RMSE_std"],
                    "Holdout_RMSE_Point": holdout_row["Holdout_RMSE_Point"],
                    "Holdout_RMSE_95CI": holdout_row["Holdout_RMSE_CI_str"],
                    "Delta_RMSE_Holdout_minus_CV": holdout_row["Holdout_RMSE_Point"] - cv_row["Repeated_RMSE_mean"],
                    "Repeated_CV_MAE_mean": cv_row["Repeated_MAE_mean"],
                    "Holdout_MAE_Point": holdout_row["Holdout_MAE_Point"],
                    "Repeated_CV_R2_mean": cv_row["Repeated_R2_mean"],
                    "Holdout_R2_Point": holdout_row["Holdout_R2_Point"],
                    "Repeated_CV_CCC_mean": cv_row["Repeated_CCC_mean"],
                    "Holdout_CCC_Point": holdout_row["Holdout_CCC_Point"],
                    "CV_Stability_Status": cv_row["Stability_Status"],
                }
            )
        comparison_frame = pd.DataFrame(comparison_rows).set_index("Model")

        # Persist core tables early so a later optional phase cannot erase the
        # most important results.
        cv_summary_path = metrics_dir / "1_all_models_repeated_cv_summary.csv"
        seeds_path = metrics_dir / "2_all_models_repeated_cv_seeds.csv"
        folds_path = metrics_dir / "3_all_models_cv_folds.csv"
        holdout_path = metrics_dir / "4_all_models_holdout_bootstrap.csv"
        comparison_path = metrics_dir / "5_all_models_cv_vs_holdout.csv"
        cv_frame.to_csv(cv_summary_path)
        pd.DataFrame(seed_records).to_csv(seeds_path, index=False)
        pd.DataFrame(fold_records).to_csv(folds_path, index=False)
        holdout_frame.to_csv(holdout_path)
        comparison_frame.to_csv(comparison_path)
        cv_frame.to_csv(reports_dir / "model_selection_cv.csv")
        comparison_frame.to_csv(reports_dir / "model_cv_vs_holdout_metrics.csv")
        artifacts["best_model"] = best_name
        artifacts["model_errors"] = model_errors
        artifacts["model_selection_cv"] = _jsonable(cv_frame.round(6).to_dict(orient="index"))
        artifacts["saved_metrics_files"] = {
            "repeated_cv_summary": str(cv_summary_path),
            "repeated_cv_seeds": str(seeds_path),
            "cv_folds": str(folds_path),
            "holdout_bootstrap": str(holdout_path),
            "cv_vs_holdout": str(comparison_path),
        }

        # ------------------------------------------------------------------
        # Optional HPO on the CV winner.
        # ------------------------------------------------------------------
        best_params: Dict[str, Any] = {}
        if run_hpo:
            try:
                optimizer = BayesianOptimizer(cfg)
                spatial_splits = list(
                    validator.get_spatial_cv(train_frame, seed=cfg.random_seed)
                )
                hpo_result = optimizer.optimize(
                    best_name,
                    X_train,
                    y_train,
                    n_trials=cfg.n_trials,
                    cv_folds=cfg.spatial_cv_folds,
                    splits=spatial_splits,
                )
                best_params = hpo_result.best_params
                artifacts["hpo_best_params"] = best_params
                artifacts["hpo_best_rmse"] = hpo_result.best_score
            except Exception as exc:
                logger.warning("HPO skipped/failed; defaults retained: %s", exc)

        # ------------------------------------------------------------------
        # Phase 4: final train/holdout analysis using the selected parameters.
        # ------------------------------------------------------------------
        final_params = {**self._model_params(best_name), **best_params}
        final_model = ModelFactory.create(best_name, **final_params)
        final_model.fit(X_train, y_train)
        final_train_prediction = final_model.predict(X_train)
        final_test_prediction = final_model.predict(X_test)
        train_metrics = evaluator.calculate_metrics(y_train, final_train_prediction)
        holdout_metrics = evaluator.calculate_metrics(y_test, final_test_prediction)
        artifacts["train_metrics"] = train_metrics
        artifacts["holdout_metrics"] = holdout_metrics
        depth_bins_cv = pd.DataFrame()
        if best_name in oof_predictions:
            valid = np.isfinite(oof_predictions[best_name])
            if np.any(valid):
                depth_bins_cv = evaluator.depth_bin_analysis(
                    y_train[valid],
                    oof_predictions[best_name][valid],
                    bins=cfg.depth_bins,
                )
                depth_bins_cv_path = metrics_dir / "6_best_model_depth_bins_cv.csv"
                depth_bins_cv.to_csv(depth_bins_cv_path)
                artifacts["saved_metrics_files"]["depth_bins_cv"] = str(depth_bins_cv_path)
        depth_bins_holdout = evaluator.depth_bin_analysis(
            y_test, final_test_prediction, bins=cfg.depth_bins
        )
        depth_bins_holdout_path = metrics_dir / "7_best_model_depth_bins_holdout.csv"
        depth_bins_holdout.to_csv(depth_bins_holdout_path)
        depth_bins_holdout.to_csv(reports_dir / "depth_bin_metrics.csv")
        artifacts["saved_metrics_files"]["depth_bins_holdout"] = str(depth_bins_holdout_path)
        holdout_prediction_frame = test_frame[["ID", "Lon", "Lat", "Depth"]].copy()
        holdout_prediction_frame = holdout_prediction_frame.rename(columns={"Depth": "True_Depth"})
        holdout_prediction_frame["Pred_Depth"] = final_test_prediction
        holdout_prediction_frame["Residual"] = final_test_prediction - holdout_prediction_frame["True_Depth"]
        holdout_prediction_frame["Absolute_Error"] = np.abs(holdout_prediction_frame["Residual"])
        holdout_prediction_frame["Relative_Error_Percent"] = 100.0 * holdout_prediction_frame["Absolute_Error"] / np.clip(
            np.abs(holdout_prediction_frame["True_Depth"]), 0.1, None
        )
        residual_std = float(np.std(holdout_prediction_frame["Residual"]))
        holdout_prediction_frame["Outlier_Flag_2SD"] = np.abs(
            holdout_prediction_frame["Residual"]
        ) > 2.0 * max(residual_std, 1e-12)
        prediction_path = metrics_dir / "8_best_model_holdout_predictions.csv"
        holdout_prediction_frame.to_csv(prediction_path, index=False)
        artifacts["saved_metrics_files"]["holdout_predictions"] = str(prediction_path)
        raw_bootstrap = pd.DataFrame(
            {
                f"{model}_{metric}": values
                for model, distribution in holdout_distributions.items()
                for metric, values in distribution.items()
            }
        )
        raw_bootstrap_path = metrics_dir / "9_holdout_bootstrap_raw_distributions.csv"
        raw_bootstrap.to_csv(raw_bootstrap_path, index=False)
        artifacts["saved_metrics_files"]["bootstrap_raw"] = str(raw_bootstrap_path)

        # ------------------------------------------------------------------
        # Optional model-complexity ladder on the training set.
        # ------------------------------------------------------------------
        complexity_ladder_frame = pd.DataFrame()
        if run_complexity_ladder:
            try:
                ladder = ModelComplexityLadder(validator, evaluator=evaluator)
                ladder_result = ladder.run_multi_rung(train_frame, feature_cols=feature_cols)
                complexity_ladder_frame = ladder_result["flat_report"].copy()
                ladder_path = metrics_dir / "0_model_complexity_ladder.csv"
                ladder_result["flat_report"].to_csv(ladder_path)
                artifacts["complexity_ladder"] = {
                    "tier_champions": ladder_result["tier_champions"],
                    "overall_champion": ladder_result["overall_champion"],
                    "flat_report": _jsonable(
                        ladder_result["flat_report"].round(6).to_dict(orient="index")
                    ),
                }
                artifacts["saved_metrics_files"]["complexity_ladder"] = str(ladder_path)
            except Exception as exc:
                logger.warning("Complexity ladder skipped: %s", exc)

        # ------------------------------------------------------------------
        # Excel workbook.
        # ------------------------------------------------------------------
        excel_path = metrics_dir / "SDB_All_Models_Professional_Evaluation.xlsx"
        try:
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                cv_frame.to_excel(writer, sheet_name="1_Repeated_CV_Summary")
                holdout_frame.to_excel(writer, sheet_name="2_Holdout_Bootstrap_CI")
                comparison_frame.to_excel(writer, sheet_name="3_CV_vs_Holdout")
                pd.DataFrame(seed_records).to_excel(writer, sheet_name="4_Seeds", index=False)
                pd.DataFrame(fold_records).to_excel(writer, sheet_name="5_Folds", index=False)
                depth_bins_holdout.to_excel(writer, sheet_name="6_Depth_Bins_Holdout")
                holdout_prediction_frame.to_excel(writer, sheet_name="7_Holdout_Predictions", index=False)
                if run_complexity_ladder and "complexity_ladder" in artifacts:
                    ladder_frame = pd.DataFrame(artifacts["complexity_ladder"]["flat_report"]).T
                    ladder_frame.to_excel(writer, sheet_name="8_Complexity_Ladder")
            artifacts["saved_metrics_files"]["excel_workbook"] = str(excel_path)
        except Exception as exc:
            logger.warning("Excel export skipped: %s", exc)

        # ------------------------------------------------------------------
        # Phase 5: refit feature engineering and model on all observations for
        # deployment. Holdout metrics above remain strictly independent.
        # ------------------------------------------------------------------
        deployment_fe = FeatureEngineer(
            cfg.band_order,
            cfg.n_const,
            cfg.deep_water_percentile,
            cfg.eps,
        ).fit(sampled_raw)
        deployment_fe.reflectance_scale_ = sampler.reflectance_scale_
        all_frame = self._build_feature_dataset(sampled_raw, deployment_fe)
        deployment_model = ModelFactory.create(best_name, **final_params)
        deployment_model.fit(all_frame[feature_cols], all_frame["Depth"].to_numpy(dtype=np.float64))
        model_path = models_dir / f"sdb_{best_name}_final.joblib"
        try:
            import joblib  # type: ignore
            joblib.dump(
                {
                    "framework_version": __version__,
                    "model": deployment_model,
                    "feature_engineer": deployment_fe,
                    "feature_cols": feature_cols,
                    "model_name": best_name,
                    "model_params": final_params,
                    "config": cfg.to_dict(),
                },
                model_path,
            )
            artifacts["model_path"] = str(model_path)
        except ImportError:
            logger.warning("joblib is not installed; deployment model was not serialized")

        # ------------------------------------------------------------------
        # Phase 6: uncertainty and full-scene inference.
        # ------------------------------------------------------------------
        try:
            error_propagator = ErrorPropagator(cfg)
            error_propagator.fit_error_model(y_test, final_test_prediction)
            artifacts["error_model_fitted"] = True
        except Exception as exc:
            error_propagator = None
            logger.warning("Error model fitting failed: %s", exc)
        uncertainty_frame = pd.DataFrame()
        try:
            uq = UncertaintyQuantifier(
                final_model,
                n_realizations=cfg.mc_realizations,
                random_seed=cfg.random_seed,
            )
            uncertainty_frame = uq.compute_mc_uncertainty(
                X_test,
                rel_noise=cfg.reflectance_relative_uncertainty,
                abs_noise=cfg.reflectance_absolute_uncertainty,
            )
            uncertainty_path = reports_dir / "holdout_mc_uncertainty.csv"
            uncertainty_frame.to_csv(uncertainty_path, index=False)
            artifacts["holdout_mean_uncertainty_std"] = float(
                uncertainty_frame["Uncertainty_Std"].mean()
            )
        except Exception as exc:
            logger.warning("Monte-Carlo uncertainty skipped: %s", exc)

        # ------------------------------------------------------------------
        # Phase 6b: RSE sensitivity analysis and publication figures.
        # All figures are written into the ``plots`` folder with Palatino
        # Linotype 10-pt typography (PNG 300-dpi + PDF).  Nothing else in the
        # pipeline is altered by this optional phase.
        # ------------------------------------------------------------------
        sensitivity_plots: Dict[str, str] = {}
        try:
            import matplotlib  # type: ignore
            matplotlib_available = True
        except Exception:
            matplotlib_available = False
        if matplotlib_available and len(X_test) > 1:
            try:
                plotter = RSEPublicationPlotter(cfg)
                analyzer = SensitivityAnalyzer(final_model, feature_cols)
                sensitivity_frame = analyzer.run_oat_sensitivity(
                    X_test, perturbation=0.05
                )
                noise_frame = analyzer.evaluate_noise_robustness(
                    X_test,
                    noise_levels=(0.01, 0.02, 0.05, 0.10),
                    random_state=cfg.random_seed,
                )
                importance_frame = SDBExplainer(
                    final_model, feature_cols
                ).compute_permutation_importance(
                    X_test, y_test, n_repeats=10, random_state=cfg.random_seed
                )
                saved_figures = plotter.plot_all(
                    sensitivity_frame=sensitivity_frame,
                    noise_frame=noise_frame,
                    importance_frame=importance_frame,
                    uncertainty_frame=uncertainty_frame,
                    y_true=y_test,
                    y_pred=final_test_prediction,
                    metrics=holdout_metrics,
                    depth_bin_frame=depth_bins_holdout,
                )
                for name, (png_path, _pdf_path) in saved_figures.items():
                    sensitivity_plots[name] = str(png_path)
                artifacts["sensitivity_plots"] = sensitivity_plots
                # CSV copies of the sensitivity tables for the article.
                sensitivity_path = metrics_dir / "10_sensitivity_oat.csv"
                sensitivity_frame.to_csv(sensitivity_path, index=False)
                noise_path = metrics_dir / "11_sensitivity_noise_robustness.csv"
                noise_frame.to_csv(noise_path, index=False)
                importance_path = metrics_dir / "12_permutation_importance.csv"
                importance_frame.to_csv(importance_path, index=False)
                artifacts["saved_metrics_files"]["sensitivity_oat"] = str(sensitivity_path)
                artifacts["saved_metrics_files"]["sensitivity_noise"] = str(noise_path)
                artifacts["saved_metrics_files"]["permutation_importance"] = str(importance_path)
                logger.info(
                    "RSE sensitivity figures written to %s (%d figures)",
                    plotter.plots_dir,
                    len(sensitivity_plots),
                )
            except Exception as exc:
                logger.warning("Sensitivity analysis / publication plots skipped: %s", exc)
        else:
            logger.warning(
                "RSE sensitivity plots skipped: matplotlib=%s, holdout=%d",
                matplotlib_available,
                len(X_test),
            )

        scene_stem = Path(cfg.raster_path).stem
        bathymetry_path = rasters_dir / f"{scene_stem}_bathymetry_{best_name}.tif"
        inference = RasterInferenceEngine(
            deployment_model,
            deployment_fe,
            cfg,
            reflectance_scale=sampler.reflectance_scale_,
        )
        inference.predict_scene(cfg.raster_path, bathymetry_path)
        artifacts["bathymetry_raster"] = str(bathymetry_path)
        if compute_uncertainty_raster and error_propagator is not None:
            try:
                uncertainty_raster_path = rasters_dir / f"{scene_stem}_uncertainty_{best_name}.tif"
                error_propagator.write_spatial_uncertainty_raster(
                    bathymetry_path,
                    uncertainty_raster_path,
                    n_simulations=50,
                )
                artifacts["uncertainty_raster"] = str(uncertainty_raster_path)
            except Exception as exc:
                logger.warning("Spatial uncertainty raster skipped: %s", exc)

        # ------------------------------------------------------------------
        # Publication workbook: one Excel file for the complete Results
        # section of the article.
        # ------------------------------------------------------------------
        article_workbook_path = metrics_dir / "Results of Article.xlsx"
        try:
            _write_article_results_workbook(
                workbook_path=article_workbook_path,
                cfg=cfg,
                sampled_raw=sampled_raw,
                train_frame=train_frame,
                test_frame=test_frame,
                feature_cols=feature_cols,
                cv_frame=cv_frame,
                holdout_frame=holdout_frame,
                comparison_frame=comparison_frame,
                seed_records=seed_records,
                fold_records=fold_records,
                oof_predictions=oof_predictions,
                holdout_predictions_by_model=holdout_predictions_by_model,
                holdout_prediction_frame=holdout_prediction_frame,
                depth_bins_cv_best=depth_bins_cv,
                depth_bins_holdout_best=depth_bins_holdout,
                uncertainty_frame=uncertainty_frame,
                best_name=best_name,
                evaluator=evaluator,
                ladder_frame=complexity_ladder_frame,
                artifacts=artifacts,
            )
            artifacts["saved_metrics_files"]["article_results_workbook"] = str(
                article_workbook_path
            )
        except Exception as exc:
            logger.warning("Article-results workbook export skipped: %s", exc)

        # ------------------------------------------------------------------
        # Phase 7: final report and manifest.
        # ------------------------------------------------------------------
        final_report = evaluator.generate_report({best_name: holdout_metrics})
        final_report.to_csv(reports_dir / "final_model_report.csv")
        self._finalize_deployment(artifacts)
        return _jsonable(artifacts)

    def _finalize_deployment(self, artifacts: Optional[Mapping[str, Any]] = None) -> Path:
        elapsed_minutes = (time.time() - self.start_time) / 60.0
        manifest = {
            "framework_version": __version__,
            "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_time_minutes": round(elapsed_minutes, 2),
            "status": "SUCCESS",
            "output_directory": str(self.cfg.output_dir.resolve()),
            "config": self.cfg.to_dict(),
            "system": {
                "python": sys.version.split()[0],
                "gpu_available": detect_gpu(),
            },
            "artifacts": _jsonable(artifacts or {}),
        }
        manifest_path = self.cfg.output_dir / "deployment_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        try:
            self.cfg.save("config.yaml")
        except Exception as exc:
            logger.warning("Could not save final configuration: %s", exc)
        logger.info(
            "SDB pipeline completed successfully | %.2f min | manifest=%s",
            elapsed_minutes,
            manifest_path,
        )
        return manifest_path

# =============================================================================
# Convenience functions and CLI
# =============================================================================

def run_sdb_for_files(
    excel_path: Union[str, Path],
    raster_path: Union[str, Path],
    output_dir: Union[str, Path] = "sdb_outputs",
    id_col: str = "En_Name",
    lon_col: str = "Longitude",
    lat_col: str = "Latitude",
    depth_col: str = "Depth",
    band_order: Sequence[str] = ("B2", "B3", "B4", "B8", "B11"),
    model_candidates: Optional[Sequence[str]] = None,
    run_hpo: bool = False,
    run_complexity_ladder: bool = False,
    compute_uncertainty_raster: bool = True,
    **extra_cfg_kwargs: Any,
) -> Dict[str, Any]:
    """One-call wrapper for a complete SDB run."""
    cfg = SDBConfig(
        excel_path=Path(excel_path),
        raster_path=Path(raster_path),
        output_dir=Path(output_dir),
        id_col=id_col,
        lon_col=lon_col,
        lat_col=lat_col,
        depth_col=depth_col,
        band_order=tuple(band_order),
        **extra_cfg_kwargs,
    )
    return SDBOperationalPipeline(cfg=cfg).run_full_pipeline(
        model_candidates=model_candidates,
        run_hpo=run_hpo,
        run_complexity_ladder=run_complexity_ladder,
        compute_uncertainty_raster=compute_uncertainty_raster,
    )


def load_deployment_bundle(model_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a serialized deployment bundle."""
    try:
        import joblib  # type: ignore
    except ImportError as exc:
        raise ImportError("Loading a model bundle requires `pip install joblib`") from exc
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, Mapping) or "model" not in bundle or "feature_engineer" not in bundle:
        raise ValueError("Invalid deployment bundle: expected model and feature_engineer")
    return dict(bundle)


def predict_new_scene(
    model_path: Union[str, Path],
    raster_path: Union[str, Path],
    output_path: Union[str, Path],
) -> Path:
    """Apply a saved model bundle to a new GeoTIFF scene."""
    bundle = load_deployment_bundle(model_path)
    config_data = dict(bundle.get("config", {}))
    valid_keys = {
        item.name
        for item in fields(SDBConfig)
        if item.init and not item.name.startswith("_")
    }
    cfg_kwargs = {key: value for key, value in config_data.items() if key in valid_keys}
    cfg_kwargs["raster_path"] = Path(raster_path)
    # The output directory is only needed for the config object and should not
    # accidentally point to a deleted training location.
    cfg_kwargs["output_dir"] = Path(output_path).parent
    cfg = SDBConfig(**cfg_kwargs)
    engine = RasterInferenceEngine(
        bundle["model"],
        bundle["feature_engineer"],
        cfg,
        reflectance_scale=getattr(bundle["feature_engineer"], "reflectance_scale_", None),
    )
    return engine.predict_scene(raster_path, output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"SDB Framework v{__version__}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("full", "train", "predict", "list-models"),
        default="full",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config path")
    parser.add_argument("--excel-path", type=str, default=None)
    parser.add_argument("--raster-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--id-col", type=str, default=None)
    parser.add_argument("--lon-col", type=str, default=None)
    parser.add_argument("--lat-col", type=str, default=None)
    parser.add_argument("--depth-col", type=str, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument(
        "--cv-strategy",
        choices=("loo", "spatial_loo", "repeated_spatial_cv", "kfold"),
        default=None,
    )
    parser.add_argument("--n-repeats", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument("--hpo", action="store_true")
    parser.add_argument("--ladder", action="store_true")
    parser.add_argument("--no-uncertainty-raster", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _strip_ipykernel_arguments(argv: Sequence[str]) -> List[str]:
    """Remove arguments injected by Jupyter/IPython when a script is run there.

    IPython starts a script with an extra pair such as::

        -f C:\\Users\\...\\kernel-<id>.json

    That pair belongs to the kernel launcher, not to this application's CLI.
    Normal command-line arguments remain untouched and are still validated by
    argparse, so genuine user typos are not silently hidden.
    """
    raw = list(argv)
    cleaned: List[str] = []
    index = 0
    while index < len(raw):
        argument = raw[index]
        if argument == "-f":
            # Drop -f and its connection-file value, if present.
            index += 2
            continue
        if argument.startswith("-f="):
            index += 1
            continue
        cleaned.append(argument)
        index += 1
    return cleaned


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    # When argv is omitted, use the process arguments.  The cleaning step is
    # harmless in a regular terminal and fixes the standard ipykernel -f
    # connection-file argument when the file is launched from Jupyter.
    effective_argv = _strip_ipykernel_arguments(
        sys.argv[1:] if argv is None else argv
    )
    args = parser.parse_args(effective_argv)
    if args.mode == "list-models":
        print(ModelRegistry.describe().to_string(index=False))
        return
    if args.mode == "predict":
        missing = [
            option
            for option, value in (
                ("--model-path", args.model_path),
                ("--input", args.input),
                ("--output", args.output),
            )
            if not value
        ]
        if missing:
            parser.error(f"Predict mode requires: {', '.join(missing)}")
        predict_new_scene(args.model_path, args.input, args.output)
        return
    # Load config, then apply explicitly supplied CLI overrides.
    if args.config:
        cfg = SDBConfig.from_yaml(args.config)
    else:
        cfg = SDBConfig()
    overrides = {
        "excel_path": args.excel_path,
        "raster_path": args.raster_path,
        "output_dir": args.output_dir,
        "id_col": args.id_col,
        "lon_col": args.lon_col,
        "lat_col": args.lat_col,
        "depth_col": args.depth_col,
        "cv_strategy": args.cv_strategy,
        "n_repeats": args.n_repeats,
        "n_bootstrap_ci": args.n_bootstrap,
    }
    changed = {key: value for key, value in overrides.items() if value is not None}
    if changed:
        config_data = cfg.to_dict()
        config_data.update(changed)
        cfg = SDBConfig(**config_data)
    if not Path(cfg.excel_path).exists():
        parser.error(
            f"Field-data file not found: {cfg.excel_path}. "
            "Pass --excel-path or --config."
        )
    if not Path(cfg.raster_path).exists():
        parser.error(
            f"Raster file not found: {cfg.raster_path}. "
            "Pass --raster-path or --config."
        )
    artifacts = SDBOperationalPipeline(cfg=cfg).run_full_pipeline(
        model_candidates=args.models,
        run_hpo=args.hpo,
        run_complexity_ladder=args.ladder,
        compute_uncertainty_raster=not args.no_uncertainty_raster,
    )
    logger.info(
        "Best model=%s | Holdout RMSE=%.4f m | raster=%s",
        artifacts.get("best_model"),
        artifacts.get("holdout_metrics", {}).get("RMSE", float("nan")),
        artifacts.get("bathymetry_raster"),
    )


if __name__ == "__main__":
    main()


# In[ ]:




