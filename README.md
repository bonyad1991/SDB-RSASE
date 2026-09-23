# SDB-RSASE
Research code for satellite-derived bathymetry using a mixture-of-experts framework.
A single, coherent Python framework for estimating shallow-water depth (Satellite-Derived Bathymetry, SDB) from multispectral satellite imagery and in-situ field measurements. It combines classical physics-based models (Stumpf, Lyzenga, Caballero-Stumpf), classical ML and gradient-boosted ensembles, and optional deep mixture-of-experts models, wrapped in a leakage-safe validation, uncertainty-quantification and raster-inference pipeline.

Designed for shallow-water field data with a maximum depth of 3 m: the default depth bins, prediction ceiling and depth-dependent error model all follow this design envelope. Update SDBConfig if your study site is deeper.

Table of contents
Features
Repository structure
Installation
Input data requirements
Quick start (CLI)
Programmatic use
Configuration files
Output structure
Supported models
Validation & uncertainty
Reproducibility
Known limitations / notes
Citation
License
Features
Fit-on-training-only feature engineering — band ratios, indices and scalers are fit exclusively on the training fold to avoid data leakage.
14 registered models (as returned by --mode list-models), from interpretable physical models to gradient boosting and neural mixture-of-experts (full list below).
Small-sample-aware validation: spatial leave-one-out, repeated spatial cross-validation, k-fold and plain LOO.
Bootstrap confidence intervals, one-at-a-time sensitivity analysis, permutation importance, and Monte-Carlo depth-uncertainty propagation.
Optional Bayesian hyper-parameter optimization via Optuna.
Tiled GeoTIFF inference with an accompanying uncertainty raster, safe for scenes larger than memory.
Model-complexity ladder with a Wilcoxon signed-rank test to check whether a more complex model actually beats a simpler one.
Publication-ready figures and workbook (1000 dpi PNG + vector PDF, Palatino-Linotype 10 pt style, Excel report with a 00_Readme sheet) for direct use in a manuscript's supplementary material.
Deployable model bundles (joblib) that can be applied to a brand-new scene with a single command.
Lazy optional dependencies — importing the module never requires rasterio, xgboost, lightgbm, catboost, optuna, torch or matplotlib; each raises a clear, actionable error only when the corresponding feature is actually used.
Repository structure

Current contents of this repository:

sdb-framework/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
└── sdb_framework.py     # the module (rename from SDB__RSASE_.py)

Recommended layout before the public release — not created yet, add these incrementally:

sdb-framework/
├── README.md
├── LICENSE                       # TODO: choose and add (see License section)
├── CITATION.cff                  # TODO: add once paper details are final
├── requirements.txt
├── environment.yml
├── .gitignore
├── sdb_framework.py
├── configs/
│   └── example_config.yaml       # sample SDBConfig (no real paths committed)
├── examples/
│   ├── README.md                 # how to obtain / structure sample data
│   └── field_points_sample.csv   # small, anonymized, optional
├── docs/
│   └── figures/                  # optional: architecture / workflow diagrams
└── tests/
    └── test_sdb_framework.py     # optional: unit tests (not yet included)

Internally, the module is already organized into clear sections you can follow top-to-bottom or use as a map when splitting it into a package later:

Section (as commented in the source)	Line range	Contents
Standard library / third-party imports	56–180	stdlib, numpy/pandas/scipy/sklearn, lazy rasterio/optuna/torch
Infrastructure and utilities	182–436	set_global_seed, detect_gpu, setup_logger, small numeric helpers
Configuration	436–637	SDBConfig (dataclass: I/O paths, schema, physics, CV, uncertainty settings)
Raster sampling and field-data ingestion	637–890	RasterSampler — reads Excel/CSV, samples raster bands at field points
Feature engineering	890–1142	FeatureEngineer — band ratios/indices, fit-on-train-only scaling
Model base classes and physical models	1142–1436	BaseSDBModel, StumpfBandRatioModel, LyzengaModel, CaballeroStumpfModel, ClusterBasedRegression
Tabular ML model wrappers	1436–1940	RF, ExtraTrees, SVR, MLP, XGBoost, LightGBM, CatBoost wrappers, ModelFactory
Bayesian optimization	1940–2338	Optuna search spaces + BayesianOptimizer
Evaluation and bootstrap CIs	2338–2486	ModelEvaluator, BootstrapEvaluator
Spatial validation	2486–2989	SpatialValidator (spatial LOO, repeated spatial CV, k-fold, plain LOO)
Sensitivity, Monte-Carlo uncertainty, error propagation	2989–3269	SensitivityAnalyzer, UncertaintyQuantifier, ErrorPropagator
Tiled raster inference	3269–3389	RasterInferenceEngine — memory-safe GeoTIFF prediction
Explainability	3389–3423	SDBExplainer (permutation importance)
Mixture models	3423–4297	GMMSoftMixtureSDB, HeterogeneousMoESDB, MixtureOfExpertsSDB, gating strategies
Registry and model-complexity ladder	4297–4525	ModelRegistry, ModelComplexityLadder (with Wilcoxon test)
Article-results workbook	4525–4939	Styled multi-sheet Excel report generator
RSE publication figures	4939–5258	RSEPublicationPlotter — sensitivity/uncertainty figures
Operational pipeline	5258–5899	SDBOperationalPipeline.run_full_pipeline() — orchestrates everything
Convenience functions and CLI	5899–6118	run_sdb_for_files, predict_new_scene, argparse CLI, main()
Installation
Option A — Conda (recommended, especially on Windows)

Rasterio depends on GDAL, which is far easier to get right through conda-forge than through pip.

bash
conda env create -f environment.yml
conda activate sdb-framework
Option B — pip
bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

Then uncomment/install any optional extras you need (XGBoost, LightGBM, CatBoost, Optuna, Matplotlib, PyTorch) — see the comments inside requirements.txt / environment.yml.

Python: 3.10+ recommended.

Input data requirements
1. Field measurements (Excel or CSV)

A table with one row per in-situ depth measurement. Column names are flexible — the loader recognizes common aliases (Longitude/Long/Lon/x, Latitude/Lat/y, Depth/depth_m/z, ID/Station/En_Name), or you can set exact names via --id-col/--lon-col/--lat-col/--depth-col or SDBConfig.

Column	Required	Description
ID	optional	Station/sample identifier
Longitude	required	Decimal degrees, CRS configurable (points_crs, default EPSG:4326)
Latitude	required	Decimal degrees
Depth	required	Measured water depth in metres (positive downward)
2. Satellite scene (GeoTIFF)

A multi-band GeoTIFF readable by rasterio. Default expected band order is B2, B3, B4, B8, B11 (Sentinel-2-style: blue, green, red, NIR, SWIR) — configurable via --band-order/SDBConfig.band_order. Any CRS is fine; field points are reprojected as needed.

Quick start (CLI)
bash
# Full pipeline: train, validate, select the best model, generate a bathymetry
# raster + uncertainty raster + publication figures + Excel report.
python sdb_framework.py \
    --mode full \
    --excel-path "field_points.xlsx" \
    --raster-path "sentinel2_scene.tif" \
    --output-dir "sdb_outputs" \
    --models catboost lightgbm extra_trees random_forest lyzenga

# List every model available in the registry
python sdb_framework.py --mode list-models

# Apply a previously trained, saved model bundle to a brand-new scene
python sdb_framework.py \
    --mode predict \
    --model-path "sdb_outputs/models/sdb_catboost_final.joblib" \
    --input "new_scene.tif" \
    --output "new_scene_bathymetry.tif"

Useful flags: --hpo (Bayesian hyper-parameter search via Optuna), --ladder (run the model-complexity ladder with a Wilcoxon test), --cv-strategy {loo,spatial_loo,repeated_spatial_cv,kfold}, --no-uncertainty-raster (skip the Monte-Carlo uncertainty raster to save time).

Note: --mode train currently runs the exact same code path as --mode full — there is no reduced "train-only" behaviour yet. Treat the two as equivalent until this is implemented.

Programmatic use
python
from sdb_framework import run_sdb_for_files

artifacts = run_sdb_for_files(
    excel_path="field_points.xlsx",
    raster_path="sentinel2_scene.tif",
    output_dir="sdb_outputs",
    model_candidates=["random_forest", "extra_trees", "lyzenga"],
    run_hpo=False,
)
print(artifacts["best_model"], artifacts["holdout_metrics"])
Configuration files

Instead of (or in addition to) CLI flags, pass a YAML/JSON file matching the SDBConfig dataclass fields:

bash
python sdb_framework.py --config configs/example_config.yaml

SDBConfig.save() writes the exact configuration used for a run into output_dir/config.yaml, making every run reproducible.

Output structure

Each run creates the following subfolders under --output-dir:

sdb_outputs/
├── models/       # deployable .joblib bundles (model + feature engineer + config)
├── rasters/      # predicted bathymetry GeoTIFF + uncertainty GeoTIFF
├── metrics/      # CSV metrics (CV, holdout, bootstrap CI, sensitivity, ladder)
├── plots/        # 1000-dpi PNG + vector PDF publication figures
├── reports/      # multi-sheet Excel "article results" workbook
├── logs/         # run logs
├── config.yaml               # exact configuration used
└── deployment_manifest.json  # summary manifest of the completed run
Supported models

14 models are registered in MODEL_REGISTRY_METADATA / ModelFactory, grouped by the framework's own ModelTier enum:

Tier (ModelTier)	Count	Models
Physical	3	Stumpf band-ratio (stumpf), Lyzenga log-linear (lyzenga), Caballero/Stumpf blend (caballero)
Clustering	2	Cluster-based local Stumpf regression (cbr), soft GMM physical mixture (gmm_soft_mixture)
Machine learning	7	Random Forest, Extra Trees, SVR, MLP (no extra deps); XGBoost, LightGBM, CatBoost (each requires its own optional package)
Heterogeneous ensemble	1	Learned-gate mixture of physical + ML experts (hetero_moe)
Neural	1	Differentiable neural mixture of experts (moe, requires torch)

Run python sdb_framework.py --mode list-models for the exact, always-up-to-date registry (default hyper-parameters, required extra package, tier).

Validation & uncertainty
Cross-validation: plain LOO, spatial LOO, repeated spatial CV, or k-fold — chosen to stay valid on small, spatially clustered field datasets.
Bootstrap confidence intervals on holdout metrics (RMSE, MAE, R², CCC).
Sensitivity analysis: one-at-a-time perturbation + permutation importance.
Uncertainty quantification: Monte-Carlo propagation of reflectance uncertainty into per-pixel depth uncertainty, exported as a raster.
Model-complexity ladder: statistically tests (Wilcoxon signed-rank) whether a more complex model significantly outperforms a simpler baseline, to avoid over-engineering.
Reproducibility
Exact configuration is saved. Every full run writes the resolved SDBConfig to output_dir/config.yaml (SDBConfig.save()), so the precise settings used for a given result are always recoverable.
Random seeds are controlled. SDBConfig.__post_init__ calls set_global_seed(), which seeds Python's random, NumPy, PYTHONHASHSEED, and PyTorch (CPU + CUDA, with deterministic cuDNN flags) when it is installed.
Trained model bundles are serialized. --mode predict / load_deployment_bundle() load a single .joblib file containing the fitted model, the fitted FeatureEngineer (with its training-only statistics), feature column order, model hyper-parameters, the framework version, and the full configuration used to train it.
CV strategy is explicitly recorded. cv_strategy, n_repeats, n_spatial_blocks, spatial_cv_folds and holdout_fraction are all part of SDBConfig and therefore part of the saved config.yaml, in addition to being logged per-fold in the Excel report.
Dependencies are pinned/documented. requirements.txt and environment.yml define the exact package set a run depends on, including which optional packages are needed for which optional model/feature.
Known limitations / notes
3 m design envelope. Depth bins, prediction ceiling and the error model assume a maximum depth of 3 m. Using the framework beyond that (it will warn, not fail) needs careful review of depth_bins / prediction_max_depth.
--mode train vs --mode full are currently identical (see Quick start).
Single-file module. The framework is currently distributed as a single, clearly sectioned module (~6,000 lines, see the table above), which can be used as-is. A future package split (sdb_framework/config.py, sampling.py, features.py, models.py, validation.py, inference.py, pipeline.py, cli.py) is recommended for long-term maintenance and unit testing, not required for the code to function.
No automated tests included yet. Consider adding pytest coverage for FeatureEngineer, SpatialValidator and RasterInferenceEngine before accepting external contributions.
Fonts: publication figures request "Palatino Linotype" and fall back silently to "DejaVu Serif" if it isn't installed — no action needed, but figures will look different across machines without that font.
Citation

If you use this framework in academic work, please cite the associated paper (add the full reference / DOI here once published) and this repository. A CITATION.cff file will be added before the public release so GitHub can generate citation snippets automatically — it needs the author list, title, DOI and year, which aren't final yet.

License

No license has been chosen yet. Add a LICENSE file (e.g. MIT, BSD-3-Clause, or Apache-2.0 for permissive reuse; GPL-3.0 if you want derivative works to stay open) before making the repository public — without one, others have no legal right to reuse the code even though it's visible on GitHub. Check whether Remote Sensing of Environment / Elsevier's code-sharing policy for your article recommends a specific license.
