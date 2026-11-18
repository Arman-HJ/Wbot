@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM 用法:
REM   download_latest_log.cmd           -> 自动下载下方全部预设实验中缺失的 run
REM   download_latest_log.cmd x2_amp    -> 仅指定实验
REM   download_latest_log.cmd x2_amp 500
REM   download_latest_log.cmd 500       -> 全部预设 + 模型间隔 500

REM 远程服务器配置
set "SERVER_USER=songhat"
set "SERVER_IP=neucrack.com"
set "SERVER_PORT=9900"
set "REPO_ROOT=/home/songhat/code_workspace/rl/unitree_rl_mjlab"

REM 默认要同步的实验（无参数时全部处理，不交互询问）
set "PRESET_COUNT=2"
set "PRESET_1_KEY=x2_velocity_v3"
set "PRESET_1_PATH=logs/rsl_rl/x2_velocity_v3"
set "PRESET_2_KEY=x2_amp"
set "PRESET_2_PATH=logs/rsl_rl/x2_amp"

set "MODEL_INTERVAL=0"
set "SINGLE_KEY="

if not "%~1"=="" (
  call :is_numeric "%~1" _IS_NUM
  if "!_IS_NUM!"=="1" (
    set "MODEL_INTERVAL=%~1"
  ) else (
    set "SINGLE_KEY=%~1"
    if not "%~2"=="" set "MODEL_INTERVAL=%~2"
  )
)

if not "%MODEL_INTERVAL%"=="0" (
  for /f "delims=0123456789" %%A in ("%MODEL_INTERVAL%") do (
    echo 错误: 模型编号间隔必须是非负整数
    exit /b 1
  )
)

echo ==========================================
echo 从远程服务器下载最新log ^(Windows CMD版^)
echo ==========================================
echo 服务器: %SERVER_USER%@%SERVER_IP%:%SERVER_PORT%
echo 模型编号间隔: %MODEL_INTERVAL% ^(0=全部^)
echo 过滤目录: videos/
if "%SINGLE_KEY%"=="" (
  echo 待处理: 全部预设 ^(x2_velocity_v3, x2_amp^)
) else (
  echo 待处理: %SINGLE_KEY%
)
echo ==========================================

if "%SINGLE_KEY%"=="" (
  for /L %%I in (1,1,%PRESET_COUNT%) do (
    call :sync_experiment !PRESET_%%I_KEY!
  )
) else (
  call :sync_experiment %SINGLE_KEY%
)

echo ==========================================
echo 所有下载完成!
echo ==========================================
exit /b 0

:is_numeric
set "_val=%~1"
set "_out=%~2"
set "%_out%=0"
for /f "delims=0123456789" %%A in ("%_val%") do exit /b 0
set "%_out%=1"
exit /b 0

:resolve_logs_path
set "_key=%~1"
set "_outvar=%~2"
for /L %%I in (1,1,%PRESET_COUNT%) do (
  if /i "%_key%"=="!PRESET_%%I_KEY!" (
    set "%_outvar%=!PRESET_%%I_PATH!"
    exit /b 0
  )
)
echo 错误: 未知实验 %_key%
exit /b 1

:sync_experiment
set "LOGS_KEY=%~1"
call :resolve_logs_path "%LOGS_KEY%" LOGS_PATH
if errorlevel 1 exit /b 1

set "LOCAL_LOGS_DIR=%CD%\%LOGS_PATH%"
set "REMOTE_LOGS_ABS=%REPO_ROOT%/%LOGS_PATH%"

echo.
echo ---------- %LOGS_KEY% ----------
echo 远程: %REMOTE_LOGS_ABS%
echo 本地: %LOCAL_LOGS_DIR%

if not exist "%LOCAL_LOGS_DIR%" mkdir "%LOCAL_LOGS_DIR%"

set "REMOTE_DIRS="
for /f "usebackq delims=" %%D in (`ssh -p %SERVER_PORT% %SERVER_USER%@%SERVER_IP% "cd '%REMOTE_LOGS_ABS%' && ls -d */ 2>/dev/null | sed 's#/##'"`) do (
  if not "%%D"=="" set "REMOTE_DIRS=!REMOTE_DIRS! %%D"
)

if "!REMOTE_DIRS!"=="" (
  echo 警告: 远程无 run 或无法访问，跳过 %LOGS_KEY%
  goto :eof
)

set "MISSING_DIRS="
for %%D in (!REMOTE_DIRS!) do (
  if not exist "%LOCAL_LOGS_DIR%\%%D\" set "MISSING_DIRS=!MISSING_DIRS! %%D"
)

if "!MISSING_DIRS!"=="" (
  echo 已全部存在本地，跳过 %LOGS_KEY%
  goto :eof
)

echo 自动下载:!MISSING_DIRS!
for %%D in (!MISSING_DIRS!) do call :download_one "%LOGS_KEY%" "%%D"
goto :eof

:download_one
set "LOGS_KEY=%~1"
set "RUN_DIR=%~2"
call :resolve_logs_path "%LOGS_KEY%" LOGS_PATH
set "REMOTE_RUN=%REPO_ROOT%/%LOGS_PATH%/%RUN_DIR%"
set "LOCAL_RUN=%CD%\%LOGS_PATH%\%RUN_DIR%"
set "DOWNLOAD_FAILED=0"

echo 开始下载 %LOGS_KEY%/%RUN_DIR%...
if not exist "%LOCAL_RUN%" mkdir "%LOCAL_RUN%"

if %MODEL_INTERVAL% GTR 0 (
  ssh -p %SERVER_PORT% %SERVER_USER%@%SERVER_IP% "cd '%REMOTE_RUN%' && tar --exclude='videos' --exclude='model_*.pt' -cf - ." | tar -xf - -C "%LOCAL_RUN%"
  if errorlevel 1 (
    echo 错误: %RUN_DIR% 基础文件下载失败
    goto :eof
  )
  for /f "usebackq delims=" %%M in (`ssh -p %SERVER_PORT% %SERVER_USER%@%SERVER_IP% "cd '%REMOTE_RUN%' && ls -1 model_*.pt 2>/dev/null"`) do (
    set "MODEL_FILE=%%M"
    set "MID="
    for /f "tokens=2 delims=_." %%I in ("%%M") do set "MID=%%I"
    if not "!MID!"=="" (
      set /a "_mod=!MID! %% %MODEL_INTERVAL%" >nul 2>nul
      if "!_mod!"=="0" (
        scp -P %SERVER_PORT% %SERVER_USER%@%SERVER_IP%:"%REMOTE_RUN%/!MODEL_FILE!" "%LOCAL_RUN%/"
        if errorlevel 1 set "DOWNLOAD_FAILED=1"
      )
    )
  )
) else (
  ssh -p %SERVER_PORT% %SERVER_USER%@%SERVER_IP% "cd '%REMOTE_RUN%' && tar --exclude='videos' -cf - ." | tar -xf - -C "%LOCAL_RUN%"
  if errorlevel 1 (
    echo 错误: %RUN_DIR% 下载失败
    goto :eof
  )
)

if "%DOWNLOAD_FAILED%"=="1" (
  echo 警告: %RUN_DIR% 部分模型下载失败
) else (
  echo %RUN_DIR% 下载成功
)
goto :eof
