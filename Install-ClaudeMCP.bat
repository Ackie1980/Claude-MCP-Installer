@echo off
:: ============================================================================
:: Claude Code & MCP Servers Installer Launcher
:: ============================================================================
:: Double-click this file to run the installer
:: ============================================================================

title Claude Code & MCP Installer

echo.
echo ============================================================
echo   Claude Code ^& MCP Servers Installer
echo ============================================================
echo.
echo This will install Claude Code CLI and configure MCP servers.
echo.
echo Starting PowerShell installer...
echo.

:: Run the PowerShell installer with execution policy bypass
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-ClaudeMCP.ps1"

:: Check for errors
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================
    echo   ERROR: Installation encountered a problem
    echo   Error code: %ERRORLEVEL%
    echo ============================================================
    echo.
)

echo.
echo Press any key to exit...
pause > nul
