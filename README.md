# SDB Framework

**Professional, leakage-safe Satellite-Derived Bathymetry (SDB) for shallow water — designed for a 0–3 m depth envelope.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![rasterio](https://img.shields.io/badge/GeoTIFF-rasterio-brightgreen)](https://rasterio.readthedocs.io/)

A single, coherent pipeline covering field-data ingestion, fit-on-training-only spectral feature engineering, physics-based and machine-learning models, spatial holdout and small-sample cross-validation, bootstrap confidence intervals, Monte-Carlo uncertainty, tiled GeoTIFF inference, and RSE-format sensitivity figures.

---

## Key features

* **Leakage-safe by design** — batch-dependent feature statistics (Lyzenga offsets, composite-index normalisation) are fitted on the training split only, then reused unchanged for holdout points and raster tiles.
* **Physics + ML model tiers** — Stumpf, Lyzenga, Caballero–Stumpf, cluster-based regression, GMM soft mixtures, Random Forest, Extra Trees, SVR, MLP, XGBoost, LightGBM, CatBoost, heterogeneous and neural Mixtures-of-Experts — all behind one registry.
* **Small-sample spatial validation** — spatial leave-one-out, repeated spatial K-fold, ARI-verified block stability, per-seed repeats.
* **Uncertainty quantification** — 2000-resample bootstrap CIs, Monte-Carlo reflectance perturbation, depth-dependent spatial error propagation, uncertainty GeoTIFFs.
* **Reviewer-friendly reproduction** — one command runs the entire pipeline on the bundled dataset.
* **Publication-ready outputs** — every manuscript table in one Excel workbook, every sensitivity figure as 1000-dpi PNG + PDF in Palatino Linotype 10 pt.

## Repository structure

```
SDB-Framework/
├── sdb_framework_final.py       # Main executable module (single-file framework)
├── requirements.txt             # Dependencies (pip install -r requirements.txt)
├── run_example.sh               # One-command run — Linux / macOS
├── run_example.bat              # One-command run — Windows
├── REVIEWER_QUICKSTART.md       # Step-by-step reproduction guide for reviewers
├── config.yaml                  # Portable paths — ALWAYS pass with --config
├── .gitattributes               # Keeps the paper source byte-identical
├── data/
│   ├── README.md                # Data dictionary and provenance
│   ├── SDBNew Latlong.xlsx      # Field observations (ID, Lon, Lat, Depth)
│   └── 20220213_SDB_5bands.tif  # Sentinel-2 5-band scene (B2,B3,B4,B8,B11)
├── outputs/                     # All results (auto-created; not versioned)
└── .gitignore
```

---

## Quick start (reviewers)

```bash
git clone https://github.com/YOUR_USERNAME/SDB-Framework.git
cd SDB-Framework
bash run_example.sh        # Windows: run_example.bat
```

That single command creates an isolated environment, installs every
dependency, and runs the complete pipeline on the data contained in
`data/`. Full details, expected outputs and troubleshooting:
**[REVIEWER_QUICKSTART.md](REVIEWER_QUICKSTART.md)**.

### Manual equivalent

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python sdb_framework_final.py --mode full --config config.yaml
```

> **Code integrity:** `sdb_framework_final.py` is published **byte-for-byte
> unchanged** from the version used to produce the paper. All portability
> comes from `config.yaml` — always run with `--config config.yaml`; never
> without it.

## Input data

| File | Content | Schema |
|---|---|---|
| `data/SDBNew Latlong.xlsx` | Field observations | `En_Name`, `Longitude`, `Latitude`, `Depth` (EPSG:4326, metres) |
| `data/20220213_SDB_5bands.tif` | Sentinel-2 MSI scene | 5 bands: `B2`, `B3`, `B4`, `B8`, `B11` (reflectance auto-scaled) |

Common column aliases (`Lon`/`Lat`, `water_depth`, …) and integer
reflectance scaling (÷10000) are detected automatically. See
[data/README.md](data/README.md) for the full data dictionary.

## CLI modes

| Mode | Purpose |
|---|---|
| `--mode full` | Complete pipeline: training → spatial CV → bootstrap → uncertainty → GeoTIFF prediction |
| `--mode predict` | Apply a saved model bundle to a new scene |
| `--mode list-models` | Print every registered model and its tier |
| `--hpo` | Add Bayesian hyper-parameter optimisation (Optuna) |
| `--ladder` | Add the model-complexity ladder with significance testing |

Example — deploy the final model on a new scene:

```bash
python sdb_framework_final.py --mode predict \
    --model-path "outputs/full_run/models/sdb_<name>_final.joblib" \
    --input  "new_scene.tif" \
    --output "new_scene_bathymetry.tif"
```

## Outputs

| Folder | Contents |
|---|---|
| `reports/` | Sampled points, engineered datasets, final report |
| `metrics/` | Repeated-CV summaries, bootstrap CIs, depth-bin tables, **`Results of Article.xlsx`** (all manuscript tables) |
| `rasters/` | Bathymetry GeoTIFF + spatial-uncertainty GeoTIFF |
| `plots/` | Sensitivity & uncertainty figures (PNG + PDF, Palatino Linotype 10 pt) |
| `models/` | Serialised final model bundle (`*.joblib`) |
| `deployment_manifest.json` | Machine-readable run summary |

## Methodology highlights

1. **Quality control** of field points, global reflectance scaling, spatial holdout (default 20 %) **before** any batch statistics are learned.
2. **Fit-on-training-only feature engineering** — 60+ spectral features (Stumpf ratios, Lyzenga-corrected logs, water/turbidity indices) frozen at training time.
3. **Repeated spatial cross-validation** — block-verified (ARI), every repetition reseeds stochastic models so reported mean ± sd is genuine.
4. **Model tournament** — lowest repeated-CV RMSE wins (R²/CCC tie-break); only models completing *both* CV and holdout are eligible.
5. **Independent holdout + bootstrap** — point estimates with 95 % CIs; depth-stratified metrics over the 0–3 m bins.
6. **Uncertainty & sensitivity** — Monte-Carlo reflectance noise, spatially correlated error fields, OAT sensitivity, noise robustness, permutation importance.
7. **Deployment** — final refit on all observations, tiled inference, uncertainty raster, complete Results workbook.

## Reproducibility

* Fixed seed (`random_seed = 42`) seeds Python, NumPy, scikit-learn models and (if installed) PyTorch.
* All CV, bootstrap and Monte-Carlo draws derive from that seed.
* Exact dependency bounds in [`requirements.txt`](requirements.txt).
* Configuration is snapshotted into `deployment_manifest.json` for every run.

## Copyright

© 2026 the authors. **All rights reserved.** No permission is granted to
copy, modify, distribute, or publish this code or data (in whole or in
part) without explicit written permission from the authors.

Contains modified Copernicus Sentinel data © ESA / Copernicus.

## Acknowledgements

Field campaigns, Copernicus Sentinel-2 imagery, and the open-source
scientific Python ecosystem (NumPy, SciPy, scikit-learn, rasterio,
Matplotlib, Optuna).
