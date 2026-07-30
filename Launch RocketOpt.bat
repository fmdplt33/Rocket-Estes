@echo off
REM ===================================================================
REM  RocketOpt one-click launcher for Windows
REM
REM  Double-click this file. On the first run it creates a private
REM  virtual environment next to itself and installs everything
REM  RocketOpt needs; that takes a few minutes and needs an internet
REM  connection. Every run after that starts in a couple of seconds.
REM
REM  Nothing is installed system-wide and nothing outside this folder
REM  is touched, so deleting the folder removes RocketOpt completely.
REM ===================================================================

setlocal
cd /d "%~dp0"

set VENV_DIR=.venv
set STAMP=%VENV_DIR%\.rocketopt-installed

echo.
echo  ==============================================
echo    RocketOpt - model rocket design suite
echo  ==============================================
echo.

REM --- Find a usable Python -----------------------------------------
set PY_CMD=
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>nul
    if !ERRORLEVEL! EQU 0 set PY_CMD=py -3
)

if "%PY_CMD%"=="" (
    where python >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        python -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>nul
        if %ERRORLEVEL% EQU 0 set PY_CMD=python
    )
)

if "%PY_CMD%"=="" (
    echo  [X] Python 3.11 or newer was not found.
    echo.
    echo      Install it from:  https://www.python.org/downloads/
    echo.
    echo      IMPORTANT: on the first installer screen, tick
    echo      "Add python.exe to PATH" before clicking Install.
    echo.
    echo      Then close this window and double-click this file again.
    echo.
    pause
    exit /b 1
)

echo  [1/3] Found Python:
%PY_CMD% --version

REM --- Create the virtual environment on first run ------------------
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo.
    echo  [2/3] First run - creating a private environment...
    %PY_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo  [X] Could not create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo  [2/3] Environment already present.
)

set VENV_PY=%VENV_DIR%\Scripts\python.exe

REM --- Install RocketOpt and its dependencies -----------------------
if not exist "%STAMP%" (
    echo.
    echo  [3/3] Installing RocketOpt and its dependencies.
    echo        This takes a few minutes the first time and needs
    echo        an internet connection. Please leave it running.
    echo.
    "%VENV_PY%" -m pip install --upgrade pip --quiet
    "%VENV_PY%" -m pip install -e ".[ui,reports]"
    if errorlevel 1 (
        echo.
        echo  [X] Installation failed. The usual cause is no internet
        echo      connection. Check your connection and try again.
        echo.
        pause
        exit /b 1
    )
    echo installed > "%STAMP%"
    echo.
    echo  Installation complete.
) else (
    echo  [3/3] Dependencies already installed.
)

echo.
echo  Starting RocketOpt...
echo  (You can close this window once the application appears.)
echo.

start "" "%VENV_DIR%\Scripts\pythonw.exe" -m rocketopt.ui

REM Give the window a moment to appear so errors stay visible.
timeout /t 3 >nul
exit /b 0
