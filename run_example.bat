@echo off
REM ============================================================================
REM SDB Framework — one-command demo run for reviewers (Windows)
REM ----------------------------------------------------------------------------
REM Usage:
REM   run_example.bat          Full pipeline  (all models, ~15-60 min)
REM   run_example.bat quick    Smoke test     (2 models,  ~3-10 min)
REM
REM What it does:
REM   0) Verifies repository files and the two datasets inside data\
REM   1) Finds a usable Python (py launcher or python), version >= 3.10
REM   2) Creates an isolated virtual environment (.venv)
REM   3) Installs dependencies from requirements.txt
REM   4) Runs:  python sdb_framework_final.py --mode full --config config.yaml
REM      (the published paper source is executed completely unchanged)
REM
REM If the ~300 MB raster is missing, download it from the repository's
REM GitHub "Releases" page into data\  — see data\README.md.
REM ============================================================================

setlocal
cd /d "%~dp0"
title SDB Framework - reviewer quick run

echo ================================================================
echo  SDB Framework - reviewer quick run
echo ================================================================
echo Working directory: %CD%
echo.

REM ---------------------------------------------------------------------------
REM 0. Repository integrity + input data — fail fast, BEFORE any install
REM ---------------------------------------------------------------------------
if not exist "sdb_framework_final.py" (
    echo [ERROR] sdb_framework_final.py not found.
    echo This folder must be the repository root - the folder that contains it.
    goto fail
)
if not exist "config.yaml" (
    echo [ERROR] config.yaml not found - it provides the portable paths.
    goto fail
)
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found - the repository is incomplete.
    goto fail
)
if not exist "data\SDBNew Latlong.xlsx" (
    echo [ERROR] Missing input: data\SDBNew Latlong.xlsx
    echo This small file ships with the repository - re-clone or re-download the ZIP.
    goto fail
)
if not exist "data\20220213_SDB_5bands.tif" (
    echo [ERROR] Missing input: data\20220213_SDB_5bands.tif
    echo.
    echo The ~300 MB raster is too large for Git. Get it from the repository
    echo GitHub "Releases" page and save it as:
    echo     data\20220213_SDB_5bands.tif
    echo Details: data\README.md
    goto fail
)
echo [OK] Repository files and data\ inputs found.
echo.

REM ---------------------------------------------------------------------------
REM 1. Select run mode: full by default, or quick smoke test
REM ---------------------------------------------------------------------------
set "EXTRA_ARGS="
set "OUT_DIR=outputs\reviewer_run"
if /I "%~1"=="quick" (
    set "EXTRA_ARGS=--models lyzenga random_forest --n-repeats 3 --n-bootstrap 200 --no-uncertainty-raster"
    set "OUT_DIR=outputs\quick_run"
    echo Mode: QUICK smoke test - 2 models, 3 CV repeats, a few minutes
) else (
    echo Mode: FULL pipeline - all models, about 15-60 minutes
    echo Tip: run "run_example.bat quick" first for a few-minute test
)
echo.

REM ---------------------------------------------------------------------------
REM 2. Find a usable Python installation - version >= 3.10
REM ---------------------------------------------------------------------------
set "PYTHON_CMD="

where py >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py"
    goto python_found
)

where python >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto python_found
)

echo [ERROR] Python was not found on this computer.
echo Install Python 3.10+ from https://www.python.org/downloads/
echo IMPORTANT: tick "Add python.exe to PATH" during installation,
echo then open a NEW Command Prompt and run this script again.
goto fail

:python_found
echo Python launcher : %PYTHON_CMD%

REM Guard against the Microsoft Store "python.exe" stub and broken installs
%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] "%PYTHON_CMD%" was found but cannot run.
    echo This is usually the Microsoft Store stub. Install the real Python
    echo from https://www.python.org/downloads/ and tick "Add to PATH".
    goto fail
)

for /f "tokens=2 delims= " %%V in ('%PYTHON_CMD% --version 2^>^&1') do set "PYVER=%%V"
echo Python version   : %PYVER%

for /f "tokens=1,2 delims=." %%A in ("%PYVER%") do (
    set "PYMAJ=%%A"
    set "PYMIN=%%B"
)
if not defined PYMAJ set "PYMAJ=3"
if not defined PYMIN set "PYMIN=0"
if %PYMAJ% LSS 3 (
    echo [ERROR] Python %PYVER% is too old. This framework needs Python 3.10+.
    goto fail
)
if %PYMAJ% EQU 3 if %PYMIN% LSS 10 (
    echo [ERROR] Python %PYVER% is too old. This framework needs Python 3.10+.
    goto fail
)
echo [OK] Python is usable.
echo.

REM ---------------------------------------------------------------------------
REM 3. Create an isolated virtual environment
REM ---------------------------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        goto fail
    )
)
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv exists but python.exe is missing inside it.
    echo Delete the .venv folder and run this script again.
    goto fail
)
set "PYTHON=.venv\Scripts\python.exe"
echo [OK] Virtual environment ready.
echo.

REM ---------------------------------------------------------------------------
REM 4. Install dependencies - needs internet on the first run
REM ---------------------------------------------------------------------------
echo Installing dependencies - first run only, internet required ...
"%PYTHON%" -m pip install --upgrade pip --disable-pip-version-check
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip - check your internet connection.
    goto fail
)

"%PYTHON%" -m pip install -r requirements.txt --disable-pip-version-check
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    echo Check your internet connection or proxy, then run this script again.
    goto fail
)
echo [OK] Dependencies installed.
echo.

REM ---------------------------------------------------------------------------
REM 5. Run the pipeline - paper source executed completely unchanged
REM ---------------------------------------------------------------------------
echo ================================================================
echo  Running SDB Framework ...
echo  Output folder : %OUT_DIR%
echo  Please wait - do not close this window ...
echo ================================================================
echo.

"%PYTHON%" sdb_framework_final.py ^
    --mode full ^
    --config config.yaml ^
    --output-dir "%OUT_DIR%" ^
    %EXTRA_ARGS%

if errorlevel 1 (
    echo.
    echo [ERROR] SDB Framework run failed - see the log messages above.
    goto fail
)

REM ---------------------------------------------------------------------------
REM 6. Success summary
REM ---------------------------------------------------------------------------
echo.
echo ================================================================
echo  SUCCESS
echo ================================================================
echo Results are in:
echo     %OUT_DIR%\
echo.
if exist "%OUT_DIR%\metrics\Results of Article.xlsx" (
    echo Key file: metrics\Results of Article.xlsx  - all manuscript tables
    echo.
)
if exist "%OUT_DIR%\deployment_manifest.json" (
    echo Summary from deployment_manifest.json:
    findstr /C:"best_model" /C:"execution_time_minutes" /C:"status" "%OUT_DIR%\deployment_manifest.json"
    echo.
)
echo Contents of %OUT_DIR%:
if exist "%OUT_DIR%" dir /b "%OUT_DIR%"
echo ================================================================
echo Press any key to close ...
pause >nul
endlocal
exit /b 0

:fail
echo.
echo ================================================================
echo  FAILED - read the message above and try again.
echo  Most common fix: download the raster from the GitHub Releases
echo  page into data\  - see data\README.md
echo ================================================================
echo Press any key to close ...
pause >nul
endlocal
exit /b 1
