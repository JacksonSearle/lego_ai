@echo off
rem Double-click to start Lego AI on Windows. Keeps itself up to date.
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
  echo First start: installing uv, the tool that manages Python for Lego AI...
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
)
uv run --no-project --python 3.12 python launch.py
echo.
echo Lego AI has stopped.
pause
