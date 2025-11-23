@echo off
title ZAYA BOT - Auto Launcher

echo ================================
echo   🚀 ZAYA Trading System Loader
echo ================================
echo.

REM ---- ENTER PROJECT FOLDER ----
cd /d C:\Users\ahmad\Desktop\trading-bot-project

REM ---- ACTIVATE ANACONDA ENV ----
echo 🔹 Activating environment: whale_env
call conda activate whale_env

echo.
echo --------------------------------
echo   STARTING WEBSOCKET ENGINE...
echo --------------------------------
start cmd /k "conda activate whale_env && python multi_ws_futures.py"

echo.
echo --------------------------------
echo      STARTING DASHBOARD...
echo --------------------------------
start cmd /k "conda activate whale_env && python server.py"

echo.
echo ======================================
echo  ✔️ All systems running!
echo  ✔️ WebSocket + Dashboard active
echo  ✔️ Leave windows open
echo ======================================
pause
