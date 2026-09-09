@echo off
echo ===================================================
echo Waste Vision System - Requirements Check and Setup
echo ===================================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Please install Python 3.8 or newer and ensure it is added to your system PATH.
    pause
    exit /b 1
)
echo [OK] Python is installed.
python --version
echo.

echo Checking pip installation...
pip --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] pip is not installed. Please install pip.
    pause
    exit /b 1
)
echo [OK] pip is installed.
echo.

echo Upgrading pip to the latest version...
python -m pip install --upgrade pip
echo.

echo Installing dependencies from requirements.txt...
IF NOT EXIST "requirements.txt" (
    echo [ERROR] requirements.txt not found in the current directory.
    pause
    exit /b 1
)

pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to install some dependencies. Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ===================================================
echo [SUCCESS] All requirements are installed and up to date!
echo You can now start the application using run_app.bat
echo ===================================================
pause
