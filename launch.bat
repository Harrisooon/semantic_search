@echo off
cd /d "%~dp0"

if not exist venv\Scripts\activate.bat (
    echo Virtual environment not found. Please run install.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python server/app.py
pause
