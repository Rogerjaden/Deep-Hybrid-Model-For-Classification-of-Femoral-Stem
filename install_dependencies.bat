@echo off
echo ======================================================
echo 🦴 MSFT-Net: Femoral Stem Classification Setup 🦴
echo ======================================================
echo.
echo [1/3] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b
)

echo [2/3] Upgrading pip...
python -m pip install --upgrade pip

echo [3/3] Installing required libraries from requirements.txt...
pip install -r requirements.txt

echo.
echo ======================================================
echo ✅ Installation Complete!
echo You can now run the project.
echo ======================================================
pause
