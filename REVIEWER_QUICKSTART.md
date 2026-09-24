# Reviewer Quick Start

This document explains how to reproduce every result of the study from a clean
computer. Three run paths are provided — pick the one that matches your machine:

* **Option A** — one-command script (recommended): `run_example.bat` / `run_example.sh`
* **Option B** — manual commands in your own Python or conda environment
* **Option C** — Jupyter notebook (`run_in_jupyter.ipynb`)

All paths execute the published source file `sdb_framework_final.py`
**unchanged**; portability is provided entirely by the root `config.yaml`
(passed as `--config config.yaml`).

---

## 0. Requirements

| Item | Value |
|---|---|
| Python | **3.10 or newer** — python.org, Anaconda or Miniconda all work |
| Disk | ~2 GB free (raster + outputs) |
| Network | Required on the first run (pip packages + raster download) |
| GPU | Not required |

---

## 1. Get the repository

```bat
git clone https://github.com/bonyad1991/SDB-RSASE.git
cd SDB-RSASE
```

(Alternatively, download the repository ZIP from the green **Code** button and
extract it.)

---

## 2. Place the input data in `data/`

| File | How to obtain it |
|---|---|
| `data/SDBNew Latlong.xlsx` | Ships with the repository — already present after clone |
| `data/20220213_SDB_5bands.tif` (~300 MB) | **Too large for Git** — download it from the repository **Releases** page and save it inside `data/` with the exact same file name |

> Full instructions, direct link and a lossless compression recipe are in
> [`data/README.md`](data/README.md).

Verify that both files exist:

```bat
dir data            (Windows)
ls -lh data         (Linux / macOS)
```

---

## 3. Do you already have Python?

Open a terminal and run:

```bat
python --version
```

### ✔ Path A — it prints `Python 3.10` or newer

You are ready — go straight to **Option A** (or B / C) below.

> **Anaconda / Miniconda users:** if a plain Command Prompt does not see your
> Python, open **Anaconda Prompt** (or run `conda activate <env>`) first.
> Any environment with Python ≥ 3.10 works.

### ✖ Path B — "not recognized", the Microsoft Store opens, or version < 3.10

Install Python first (about 2 minutes):

1. Download the latest **Python 3.x** from <https://www.python.org/downloads/>.
2. Run the installer and **tick “Add python.exe to PATH”** at the bottom of the
   first screen.
3. Click **Install Now** and wait for completion.
4. **Open a brand-new terminal** (PATH changes apply only to new windows).
5. Confirm: `python --version` → `Python 3.x.y`.

*Anaconda / Miniconda alternative:* open **Anaconda Prompt** and run every
command below inside it — no separate python.org install is needed.

---

## Option A — one command (recommended)

**Windows** (Command Prompt or Git CMD):

```bat
run_example.bat
```

Optional smoke test first (2 models, ~3–10 min):

```bat
run_example.bat quick
```

**Linux / macOS:**

```bash
bash run_example.sh
```

What the script does: verifies the repository files **and both data inputs** →
finds a usable Python (≥ 3.10) → creates an isolated `.venv` → installs
`requirements.txt` → executes the complete pipeline (training → spatial CV →
bootstrap → uncertainty → GeoTIFF prediction). Any problem stops early with a
clear `[ERROR]` message instead of failing half-way through.

---

## Option B — manual, in your own environment

### B1 — standard Python (virtual environment)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python sdb_framework_final.py --mode full --config config.yaml --output-dir "outputs\reviewer_run"
```

Linux / macOS: activate with `source .venv/bin/activate` and use normal `/`
paths.

### B2 — conda

```bat
conda create -n sdb python=3.11 pip -y
conda activate sdb
pip install -r requirements.txt
python sdb_framework_final.py --mode full --config config.yaml --output-dir "outputs\reviewer_run"
```

Equivalent one-liner without activating the environment:

```bat
conda run -n sdb python sdb_framework_final.py --mode full --config config.yaml --output-dir "outputs\reviewer_run"
```

> **Always pass `--config config.yaml`.** The source file is published unchanged
> (original placeholder defaults are retained for the paper record);
> `config.yaml` supplies the portable relative paths used on your machine.

---

## Option C — Jupyter

Open `run_in_jupyter.ipynb` in Jupyter Notebook / JupyterLab / VS Code and run
the cells from top to bottom. The core cell is:

```python
SDBOperationalPipeline(config_path="config.yaml").run_full_pipeline()
```

Alternatives (`%run`, `!python ...`, `run_sdb_for_files(...)`,
`predict_new_scene(...)`) are shown inside the notebook itself.

---

## 4. Expected outputs

After ~10–60 minutes (depending on the machine) the folder
`outputs/reviewer_run/` (or `outputs/quick_run/` for the smoke test) contains:

```
reports/    sampled points, engineered datasets, final report
metrics/    repeated-CV tables, bootstrap CIs, depth-bin metrics,
            "Results of Article.xlsx" (all manuscript tables)
rasters/    *_bathymetry_*.tif  and  *_uncertainty_*.tif
plots/      RSE-format sensitivity figures (PNG + PDF)
models/     serialized final model (*.joblib)
deployment_manifest.json
```

**Success signals:**

* Console ends with
  `SDB pipeline completed successfully | ... min | manifest= ...`
  and `Best model = ... | Holdout RMSE = ...`
* `deployment_manifest.json` contains `"status": "SUCCESS"`
* `metrics/Results of Article.xlsx` exists (all manuscript tables)

---

## 5. Useful extra modes

```bat
:: List every available model and its tier
python sdb_framework_final.py --mode list-models

:: Apply a saved model to a new scene
python sdb_framework_final.py --mode predict --model-path "outputs\reviewer_run\models\sdb_<name>_final.joblib" --input "new_scene.tif" --output "new_scene_bathymetry.tif"

:: Optional: Bayesian hyper-parameter optimization + complexity ladder
python sdb_framework_final.py --mode full --hpo --ladder --config config.yaml --output-dir "outputs\reviewer_run"
```

---

## 6. Reproducibility

* Fixed seed (`random_seed = 42`) seeds Python, NumPy, scikit-learn models
  and (if installed) PyTorch.
* All cross-validation, bootstrap and Monte-Carlo draws use that seed.
* Exact dependency versions are pinned in `requirements.txt`.
* Re-running the command above must reproduce every number in the
  manuscript within floating-point tolerance.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Python was not found` or the Microsoft Store opens | No real Python on `PATH` → install from python.org with **Add to PATH** (Path B), or use *Anaconda Prompt* |
| `"python" ... was found but cannot run` | Microsoft Store stub — same fix as above |
| `Python X.Y is too old` | The framework needs **3.10+** → upgrade |
| `Missing input: data\20220213_SDB_5bands.tif` | Download the raster from the **Releases** page into `data/` (see `data/README.md`) |
| `Failed to install dependencies` | Check internet/proxy → run the script again; if it persists, delete `.venv` and rerun |
| `Raster operations require rasterio` | `pip install rasterio` (already listed in `requirements.txt`) |
| `Field-data file not found` | Keep the shipped file name exactly: `data/SDBNew Latlong.xlsx` (space included) |
| `No field point was sampled successfully` | Raster and points must overlap; points are EPSG:4326 |
| Out-of-memory on the large raster | Reduce `tile_size` in `config.yaml` (default 512) |

---

For data licensing, attribution and the large-raster policy see
[`data/README.md`](data/README.md).
Contains modified Copernicus Sentinel data © ESA / Copernicus.
