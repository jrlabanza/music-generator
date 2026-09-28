@echo off
:: One-time setup on Windows - runs initialize.py via "Setup Music Gen Studio.bat"
:: (finds Python 3.10-3.13, YuE submodule, .venv, CUDA PyTorch, packages, model weights).
:: On Linux use linux\initialize.sh instead.
call "%~dp0Setup Music Gen Studio.bat" %*
