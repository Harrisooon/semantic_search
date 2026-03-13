@echo off
cd /d "%~dp0"

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing PyTorch with CUDA 12.1 support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo Installing semantic-search and remaining dependencies...
pip install -e ".[server]"

echo.
echo Installation complete. The virtual environment is in .\venv\
echo Before launching, edit config.yaml to set your watched_folders path.
echo Then double-click launch.bat to start.
pause
