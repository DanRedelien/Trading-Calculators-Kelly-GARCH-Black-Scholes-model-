@echo off
title Streamlit Module Launcher

:menu
cls
echo =================================
echo   Streamlit Module Launcher
echo =================================
echo.
echo Select what module to run:
echo.
echo 1. Black Scholes model (BSmodel.py)
echo 2. Quantitative Volatility Analysis (Qnt_VolaAnalysis.py)
echo 3. Kelly Criterion calculator (Trading_Calc.py)
echo 4. Run All
echo.
echo Press Ctrl+C to exit.
echo.

set /p choice="Enter your choice (1-4): "

if not defined choice (
    goto menu
)

if "%choice%"=="1" goto run_bs
if "%choice%"=="2" goto run_qnt
if "%choice%"=="3" goto run_kelly
if "%choice%"=="4" goto run_all

echo Invalid choice. Press any key to try again.
pause > nul
goto menu

:run_bs
cls
echo Starting Black Scholes model...
echo You can close this window or press Ctrl+C here to stop the server.
streamlit run BSmodel.py
goto menu

:run_qnt
cls
echo Starting Quantitative Volatility Analysis...
echo You can close this window or press Ctrl+C here to stop the server.
streamlit run Qnt_VolaAnalysis.py
goto menu

:run_kelly
cls
echo Starting Kelly Criterion calculator...
echo You can close this window or press Ctrl+C here to stop the server.
streamlit run Trading_Calc.py
goto menu

:run_all
cls
echo Starting all modules in separate windows...
start "Black Scholes model" streamlit run BSmodel.py
start "Quantitative Volatility Analysis" streamlit run Qnt_VolaAnalysis.py
start "Kelly Criterion calculator" streamlit run Trading_Calc.py
echo.
echo Modules are starting. Press any key to return to the main menu.
pause > nul
goto menu
