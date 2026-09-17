@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
title Music Gen Studio (internet)
if not exist password.txt (
  echo Create password.txt with one line, the password your phone will use, then run this again.
  pause
  exit /b 1
)
echo Starting Music Gen Studio with a Cloudflare tunnel. The public https address is printed below
echo and shown as a QR code when you click the share pill on the page. Password: see password.txt
echo Keep this window open while you are out; press Ctrl+C or close it to stop.
"YuE\.venv\Scripts\python.exe" webui.py --share --tunnel --password-file password.txt --open
pause
