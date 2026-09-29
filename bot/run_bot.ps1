# Telegram-бот на этом компьютере (Windows): запуск с перезапуском при сбое.
#
#   powershell -ExecutionPolicy Bypass -File bot\run_bot.ps1                  # в этом окне
#   powershell -ExecutionPolicy Bypass -File bot\run_bot.ps1 -Autostart       # при каждом входе в Windows
#   powershell -ExecutionPolicy Bypass -File bot\run_bot.ps1 -RemoveAutostart
#
# Токен — в файле .env в корне репозитория (шаблон .env.example) или в
# переменной DXA_QC_TG_TOKEN. Журнал: %LOCALAPPDATA%\DXA-QC\logs\bot.log.
# Автозапуск — задание планировщика для текущего пользователя, права
# администратора не нужны.
param(
    [switch]$Autostart,
    [switch]$RemoveAutostart,
    [string]$Python = ""
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Task = "DXA-QC Telegram bot"

if ($RemoveAutostart) {
    Unregister-ScheduledTask -TaskName $Task -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "автозапуск бота снят"
    exit 0
}

if ($Autostart) {
    $arg = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    if ($Python) { $arg += " -Python `"$Python`"" }
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $Root
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName $Task -Action $action -Trigger $trigger -Settings $settings `
        -Description "Telegram-бот сервиса контроля качества DXA" -Force | Out-Null
    Start-ScheduledTask -TaskName $Task
    Write-Host "бот запущен в фоне и будет запускаться при входе в Windows (задание «$Task»)"
    exit 0
}

if (-not $Python) {
    $venv = Join-Path $Root "build\py311\Scripts\python.exe"
    $Python = if (Test-Path $venv) { $venv } else { "python" }
}
$logs = Join-Path $env:LOCALAPPDATA "DXA-QC\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$log = Join-Path $logs "bot.log"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
Set-Location $Root
Write-Host "бот запускается, журнал: $log (остановить — Ctrl+C)"
$ErrorActionPreference = "Continue"
while ($true) {
    & $Python -m bot --log $log
    if ($LASTEXITCODE -eq 2) { Write-Host "нет токена или токен не принят — см. .env.example"; exit 2 }
    Write-Host "бот остановился (код $LASTEXITCODE), перезапуск через 10 с"
    Start-Sleep -Seconds 10
}
