@echo off
title Claude MCP Installer - GUI
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0Install-ClaudeMCP-GUI.ps1"
