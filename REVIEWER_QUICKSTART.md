# Reviewer Quick Start

Everything needed to reproduce the results of this study is contained in
this repository together with the two input datasets in `data/`.

## Requirements

* Python 3.10 or newer
* ~2 GB free disk space (raster + outputs)
* No GPU required

## Input data (included)

| File | Description |
|---|---|
| `data/SDBNew Latlong.xlsx` | Field observations: `En_Name, Longitude, Latitude, Depth` (EPSG:4326, depth in m) |
| `data/20220213_SDB_5bands.tif` | Sentinel-2 5-band scene: `B2, B3, B4, B8, B11` |

## Run — option A (one command, recommended)

Linux / macOS:

```bash
bash run_example.sh
```

Windows (Command Prompt):

```bat
run_example.bat
```

The script creates an isolated virtual environment, installs
`requirements.txt`, and executes the complete pipeline
(training → spatial CV → bootstrap → uncertainty → GeoTIFF prediction).

## Run — option B (manual)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python sdb_framework_final.py --mode full --config config.yaml --output-dir "outputs/reviewer_run"
```

> **Always pass `--config config.yaml`.** The source file is published
> unchanged (original placeholder defaults are retained for the paper record);
> `config.yaml` supplies the portable relative paths used on your machine.

## Expected outputs

After ~10–30 minutes (depending on the machine) the folder
`outputs/reviewer_run/` contains:

```
reports/   sampled points, engineered datasets, final report
metrics/   repeated-CV tables, bootstrap CIs, depth-bin metrics,
           "Results of Article.xlsx" (all manuscript tables)
rasters/   *_bathymetry_*.tif  and  *_uncertainty_*.tif
plots/     RSE-format sensitivity figures (PNG + PDF)
models/    serialized final model (*.joblib)
deployment_manifest.json
```

## Useful extra modes

```bash
# List every available model and its tier
python sdb_framework_final.py --mode list-models

# Apply a saved model to a new scene
python sdb_framework_final.py --mode predict \
    --model-path "outputs/reviewer_run/models/sdb_<name>_final.joblib" \
    --input  "new_scene.tif" \
    --output "new_scene_bathymetry.tif"

# Optional: Bayesian hyper-parameter optimization + complexity ladder
python sdb_framework_final.py --mode full --hpo --ladder \
    --config config.yaml --output-dir "outputs/reviewer_run"
```

## Reproducibility

* Fixed seed (`random_seed = 42`) seeds Python, NumPy, scikit-learn models
  and (if installed) PyTorch.
* All cross-validation, bootstrap and Monte-Carlo draws use that seed.
* Exact dependency versions are pinned in `requirements.txt`.
* Re-running the command above must reproduce every number in the
  manuscript within floating-point tolerance.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Raster operations require rasterio` | `pip install rasterio` (already in requirements) |
| `Field-data file not found` | Check the `--excel-path` spelling/spaces |
| `No field point was sampled successfully` | Raster and points must overlap; points are EPSG:4326 |
| Out-of-memory on large raster | Lower `--tile-size` effect via config `tile_size: 512` |
