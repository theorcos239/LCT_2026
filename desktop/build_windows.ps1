# Сборка настольного приложения и установщика для Windows.
#
#   powershell -ExecutionPolicy Bypass -File desktop\build_windows.ps1
#   powershell -ExecutionPolicy Bypass -File desktop\build_windows.ps1 -Version 1.0.1 -SkipInstaller
#
# Что делает (из корня репозитория):
#   1. окружение Python 3.11 с теми же версиями пакетов, что в Docker-образе
#      (requirements.lock), плюс сборочные (desktop\requirements-desktop.txt);
#   2. PyInstaller -> dist\DXA-QC\ (DXA-QC.exe — окно, dxa-qc-cli.exe — консоль);
#   3. проверка сборки: версия, загрузка моделей, отчёт на демо-архиве, если он есть;
#   4. Inno Setup -> dist\DXA-QC-Setup-<версия>.exe и переносной zip.
#
# Нужны: uv (https://docs.astral.sh/uv/) и Inno Setup 6 (ISCC.exe). Интернет —
# только для первой установки пакетов.
param(
    [string]$Version = "1.0.0",
    [switch]$SkipInstaller,
    [switch]$SkipPortable
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Venv = Join-Path $Root "build\py311"
$Py = Join-Path $Venv "Scripts\python.exe"

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }

Step "окружение Python 3.11"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "нужен uv: powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`""
}
if (-not (Test-Path $Py)) {
    uv python install 3.11
    uv venv $Venv --python 3.11
}
uv pip install --python $Py -r requirements.lock -r desktop\requirements-desktop.txt
if ($LASTEXITCODE -ne 0) { throw "не удалось установить пакеты" }

Step "иконки"
& $Py desktop\make_icons.py

Step "PyInstaller"
& $Py -m PyInstaller desktop\dxa_qc.spec --noconfirm --distpath dist --workpath build\pyinstaller
if ($LASTEXITCODE -ne 0) { throw "PyInstaller завершился с ошибкой" }
$Cli = Join-Path $Root "dist\DXA-QC\dxa-qc-cli.exe"

Step "проверка сборки"
& $Cli version
if ($LASTEXITCODE -ne 0) { throw "dxa-qc-cli не запускается" }
if (Test-Path "demo.zip") {
    $out = Join-Path $Root "build\smoke_report.csv"
    & $Cli batch demo.zip --out $out --quiet
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $out)) { throw "пакетная обработка demo.zip не прошла" }
    Write-Host "отчёт по demo.zip: $out"
} else {
    Write-Host "demo.zip нет — пакетная проверка пропущена (python docs\presentation\make_demo_zip.py)"
}

if (-not $SkipPortable) {
    Step "переносной zip"
    $zip = Join-Path $Root "dist\DXA-QC-$Version-portable.zip"
    if (Test-Path $zip) { Remove-Item $zip }
    Compress-Archive -Path "dist\DXA-QC" -DestinationPath $zip -CompressionLevel Optimal
    Write-Host $zip
}

if (-not $SkipInstaller) {
    Step "установщик Inno Setup"
    $iscc = @(
        (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source,
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
    if (-not $iscc) {
        throw "нет Inno Setup 6: https://jrsoftware.org/isdl.php (можно без прав администратора: /CURRENTUSER)"
    }
    & $iscc /Q "/DAppVersion=$Version" desktop\installer.iss
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup завершился с ошибкой" }
    Get-Item "dist\DXA-QC-Setup-$Version.exe" | Select-Object Name, @{n = "МБ"; e = { [math]::Round($_.Length / 1MB) } }
}
Write-Host "`nготово" -ForegroundColor Green
