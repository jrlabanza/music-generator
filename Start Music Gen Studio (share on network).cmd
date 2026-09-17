@echo off
cd /d "%~dp0"
title Music Gen Studio (shared on your network)
echo Starting Music Gen Studio for everyone on your network.
echo The console prints the address to give your friends. Windows Firewall must allow TCP port 7860 (see README).
echo Keep this window open while people use it; press Ctrl+C or close it to stop.
"YuE\.venv\Scripts\python.exe" webui.py --share --open
pause
