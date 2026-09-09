@echo off
echo ===================================================
echo Starting Waste Vision System
echo ===================================================
echo.

echo Waiting for server to start and automatically opening browser...
start /b powershell -WindowStyle Hidden -Command "1..60 | ForEach-Object { try { (New-Object System.Net.Sockets.TcpClient('127.0.0.1', 8501)).Close(); Start-Process 'http://localhost:8501'; break } catch { Start-Sleep -Milliseconds 500 } }"

echo Launching the Streamlit dashboard...
python main.py

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] The application crashed or failed to start.
    echo Please check the error messages above. Make sure you have run check_requirements.bat first.
    pause
    exit /b 1
)

