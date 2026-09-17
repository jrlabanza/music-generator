@echo off
cd /d "%~dp0"
title Music Gen Studio
echo Starting Music Gen Studio... the page opens in your browser when the server is ready.
echo Keep this window open while you use it; press Ctrl+C or close it to stop.
"YuE\.venv\Scripts\python.exe" webui.py --open
pause
