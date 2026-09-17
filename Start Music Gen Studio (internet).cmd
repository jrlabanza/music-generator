@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
title Music Gen Studio (internet)
if not exist password.txt (
  echo Create password.txt with one line, the password your phone will use, then run this again.
  pause
  exit /b 1
)
echo Music Gen Studio with a Cloudflare tunnel. The public https address is printed below and saved to
echo public_url.txt; the share pill on the page shows it as a QR code. Password: see password.txt
echo If the app ever exits it restarts by itself. To stop: press Ctrl+C, answer Y, then close this window.
:again
"YuE\.venv\Scripts\python.exe" webui.py --share --tunnel --password-file password.txt --open
echo.
echo The app stopped (exit code %ERRORLEVEL%). Restarting in 15 seconds...
timeout /t 15 /nobreak >nul
goto again
