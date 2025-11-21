@echo off
REM AVCS DNA-MATRIX SPIRIT v7.0 - Windows Demo Launcher

echo 🚀 AVCS DNA-MATRIX SPIRIT v7.0 - Demo Launcher
echo ==============================================

REM Check if Docker is running
docker version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)

echo ✅ Docker is running

REM Build and start services
echo.
echo 📦 Building and starting services...
docker-compose up -d --build

REM Wait for services to start
echo.
echo ⏳ Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Display access information
echo.
echo 🎯 ACCESS INFORMATION:
echo    • AVCS Dashboard:  http://localhost:8501
echo    • API Documentation: http://localhost:8501/docs
echo.
echo 🔧 MANAGEMENT COMMANDS:
echo    • View logs:       docker-compose logs -f
echo    • Stop services:   docker-compose down
echo    • Restart:         docker-compose restart
echo.
echo 💡 For technical support: engineering@avcs-systems.com
echo ==============================================
pause
