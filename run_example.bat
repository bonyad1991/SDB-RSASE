@echo off
REM ============================================================================
REM One-command demo run for reviewers (Windows)
REM Usage:  run_example.bat
REM ============================================================================
setlocal

echo == SDB Framework: reviewer quick run ==

REM 1. Create a clean virtual environment (first run only)
if not exist ".venv" (
    python -m venv .venv
)
call .venv\Scripts\activate.bat

REM 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

REM 3. Run the full pipeline on the provided data.
REM    --config supplies portable relative paths; the published paper code
REM    itself is executed completely unchanged.
python sdb_framework_final.py ^
    --mode full ^
    --config config.yaml ^
    --output-dir "outputs\reviewer_run"

echo.
echo Done. Results are in outputs\reviewer_run\
endlocal
