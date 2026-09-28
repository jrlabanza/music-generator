@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
title Music Gen Studio setup
echo Setting up Music Gen Studio: YuE submodule, Python environment, PyTorch with CUDA, packages, model weights.
echo The first run downloads about 10 GB. Running it again only fills in what is missing.
echo.
set "PY="
for %%V in (3.12 3.11 3.13 3.10) do if not defined PY py -%%V -c "pass" >nul 2>&1 && set "PY=py -%%V"
if not defined PY python -c "import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 13))" >nul 2>&1 && set "PY=python"
if not defined PY goto nopython
%PY% initialize.py %*
if errorlevel 1 goto failed
echo.
echo All set. Double-click "Start Music Gen Studio.cmd" to open the app.
goto end

:nopython
echo Python 3.10 to 3.13 was not found. Install Python 3.12 from https://www.python.org/downloads/windows/
echo and run this again.
goto end

:failed
echo.
echo Setup stopped - see the message above. Fix that and run this again; finished steps are skipped.

:end
pause
