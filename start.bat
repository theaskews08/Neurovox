@echo off
REM NeuroVox Voice Assistant Startup Script
echo Starting NeuroVox Voice Assistant...
call venv\Scripts\activate.bat
cd src
python main.py
pause
