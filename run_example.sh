#!/usr/bin/env bash
# ============================================================================
# One-command demo run for reviewers (Linux / macOS)
# Usage:  bash run_example.sh
# ============================================================================
set -euo pipefail

echo "== SDB Framework: reviewer quick run =="

# 1. Create a clean virtual environment (first run only)
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Run the full pipeline on the provided data.
#    --config supplies portable relative paths; the published paper code
#    itself is executed completely unchanged.
python sdb_framework_final.py \
    --mode full \
    --config config.yaml \
    --output-dir "outputs/reviewer_run"

echo ""
echo "Done. Results are in outputs/reviewer_run/"
echo "  - metrics/  : CSV/Excel evaluation tables"
echo "  - rasters/  : bathymetry GeoTIFF"
echo "  - plots/    : sensitivity figures"
