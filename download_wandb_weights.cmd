@echo off
setlocal EnableExtensions

REM 用法:
REM   download_wandb_weights.cmd 2026-06-14_11-48-48
REM       -> 在预设 W&B 项目中自动查找该 run，下载整个 run 文件夹
REM   download_wandb_weights.cmd pnd_adam_pro_velocity_v3_legs 2026-06-14_16-38-44
REM       -> 指定实验 + run
REM   download_wandb_weights.cmd --wandb-run entity/project/run_id
REM
REM 需先 wandb login；Python 环境需已安装 wandb（如 env_mjlab）。

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if "%~1"=="" (
  python scripts/download_wandb_weights.py --help
  exit /b 0
)

if "%~2"=="" (
  python scripts/download_wandb_weights.py "%~1"
) else (
  python scripts/download_wandb_weights.py "%~1" "%~2"
)
exit /b %ERRORLEVEL%
