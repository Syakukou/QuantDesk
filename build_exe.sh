#!/usr/bin/env bash
# QuantDesk 打包脚本：生成 onedir 文件夹版（dist/QuantDesk/QuantDesk.exe）
# onedir 不往临时目录解压，比 onefile 稳（规避杀软/权限导致的闪退）。
# 用法（Git Bash）：
#   cd QuantDesk
#   source .venv/Scripts/activate
#   bash build_exe.sh
set -e
cd "$(dirname "$0")"
pyinstaller --noconfirm --onedir --windowed --name QuantDesk \
  --collect-all akshare \
  --hidden-import PyQt6.sip \
  --hidden-import matplotlib.backends.backend_qtagg \
  main.py
# 附一个诊断启动脚本（显示退出码）
cat > dist/QuantDesk/run_diagnose.bat <<'BAT'
@echo off
cd /d "%~dp0"
"QuantDesk.exe"
echo.
echo ===== Exit code: %ERRORLEVEL% =====
pause
BAT
echo "构建完成：dist/QuantDesk/QuantDesk.exe"
