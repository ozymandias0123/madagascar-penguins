@echo off
chcp 65001 >nul
title Penguin Squad — Installer
color 0A

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║                                                          ║
echo  ║      🐧  PENGUIN SQUAD  —  AUTO INSTALLER               ║
echo  ║                                                          ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

:: ── Check Python ──────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  ❌  Python not found!
    echo.
    echo  👉  Download Python from: https://www.python.org/downloads/
    echo      ⚠️  During install — check "Add Python to PATH"!
    echo.
    start https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  ✅  Python %PYVER% found.

:: ── Ask which exchange ────────────────────────────────────────────────────
echo.
echo  ══════════════════════════════════════════════════════════
echo  Which exchange will you use?
echo  ══════════════════════════════════════════════════════════
echo.
echo    [1]  MetaTrader 5  (Forex / CFDs — e.g. Exness, IC Markets)
echo    [2]  Bybit / Binance / OKX / Kraken  (Crypto)
echo    [3]  Both
echo.
set /p EXCHANGE_CHOICE="  Your choice (1/2/3): "

set USE_MT5=0
set USE_CRYPTO=0

if "%EXCHANGE_CHOICE%"=="1" set USE_MT5=1
if "%EXCHANGE_CHOICE%"=="2" set USE_CRYPTO=1
if "%EXCHANGE_CHOICE%"=="3" (
    set USE_MT5=1
    set USE_CRYPTO=1
)
if "%EXCHANGE_CHOICE%"=="" set USE_CRYPTO=1

:: ── MetaTrader 5 desktop app ──────────────────────────────────────────────
if "%USE_MT5%"=="1" (
    echo.
    echo  ══════════════════════════════════════════════════════════
    echo  MetaTrader 5 Setup
    echo  ══════════════════════════════════════════════════════════
    echo.

    :: Check if MT5 is already installed
    set MT5_FOUND=0
    if exist "%PROGRAMFILES%\MetaTrader 5\terminal64.exe"   set MT5_FOUND=1
    if exist "%PROGRAMFILES(x86)%\MetaTrader 5\terminal64.exe" set MT5_FOUND=1
    if exist "%APPDATA%\MetaQuotes\Terminal" set MT5_FOUND=1

    if "%MT5_FOUND%"=="1" (
        echo  ✅  MetaTrader 5 is already installed.
    ) else (
        echo  MetaTrader 5 desktop app is NOT installed.
        echo.
        echo  You need to:
        echo    1. Download MT5 from your broker  (e.g. Exness, IC Markets, Pepperstone)
        echo       OR from: https://www.metatrader5.com/en/download
        echo    2. Install it and log in with your broker account
        echo    3. Keep MT5 running when you use Penguin Squad
        echo.
        echo  Opening download page...
        start https://www.metatrader5.com/en/download
        echo.
        echo  ⏳  Install MetaTrader 5, log in, then press any key to continue...
        pause >nul
    )
)

:: ── Upgrade pip ────────────────────────────────────────────────────────────
echo.
echo  [1/4]  Upgrading pip...
python -m pip install --upgrade pip --quiet
echo  ✅  pip ready.

:: ── Install TA-Lib ─────────────────────────────────────────────────────────
echo.
echo  [2/4]  Installing TA-Lib (technical indicators)...
python -m pip install TA-Lib-prebuilt --quiet 2>nul
if errorlevel 1 (
    python -m pip install TA-Lib --quiet 2>nul
    if errorlevel 1 (
        echo  ⚠️  TA-Lib not installed — some indicators will use fallback.
    ) else (
        echo  ✅  TA-Lib installed.
    )
) else (
    echo  ✅  TA-Lib installed.
)

:: ── Install MT5 Python connector ───────────────────────────────────────────
if "%USE_MT5%"=="1" (
    echo.
    echo  [3/4]  Installing MetaTrader5 Python connector...
    python -m pip install MetaTrader5 --quiet
    if errorlevel 1 (
        echo  ⚠️  MT5 connector install failed.
    ) else (
        echo  ✅  MetaTrader5 connector ready.
    )
) else (
    echo.
    echo  [3/4]  Skipping MetaTrader5 (crypto exchange selected).
    echo  ✅  Skipped.
)

:: ── Install Penguin Squad ──────────────────────────────────────────────────
echo.
echo  [4/4]  Installing Penguin Squad...
python -m pip install git+https://github.com/YOUR_USERNAME/madagascar-penguins.git --quiet
if errorlevel 1 (
    echo  ❌  Install failed. Check internet connection and try again.
    pause
    exit /b 1
)
echo  ✅  Penguin Squad installed.

:: ── Create desktop shortcut ───────────────────────────────────────────────
echo.
echo  Creating desktop shortcut...
set SHORTCUT="%USERPROFILE%\Desktop\Penguin Squad.bat"
echo @echo off > %SHORTCUT%
echo chcp 65001 ^>nul >> %SHORTCUT%
echo title Penguin Squad >> %SHORTCUT%
echo madagascar-penguins >> %SHORTCUT%
echo pause >> %SHORTCUT%
echo  ✅  Shortcut created on Desktop.

:: ── Done ───────────────────────────────────────────────────────────────────
echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║                                                          ║
echo  ║   ✅  Installation complete!                             ║
echo  ║                                                          ║
echo  ║   ▶  Double-click "Penguin Squad" on your Desktop        ║
echo  ║      OR run:  madagascar-penguins                              ║
echo  ║                                                          ║
echo  ║   The setup wizard will ask for:                         ║
echo  ║     • Telegram Bot Token + Chat ID                       ║
echo  ║     • Exchange credentials (MT5 / API keys)              ║
echo  ║     • LLM API keys (optional)                            ║
echo  ║                                                          ║
echo  ║   🔒 Everything saved at:                                ║
echo  ║      C:\Users\%USERNAME%\.penguin_squad\.env             ║
echo  ║      (only on YOUR computer — never uploaded)            ║
echo  ║                                                          ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

set /p LAUNCH="  Launch Penguin Squad now? (y/n): "
if /i "%LAUNCH%"=="y" (
    madagascar-penguins
)

pause
