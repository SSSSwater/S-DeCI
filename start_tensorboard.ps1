param(
    [string]$LogDir = "outputs\tensorboard",
    [int]$Port = 6006,
    [string]$HostName = "127.0.0.1",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$ResolvedLogDir = Join-Path $ProjectRoot $LogDir
if (-not (Test-Path $ResolvedLogDir)) {
    New-Item -ItemType Directory -Path $ResolvedLogDir | Out-Null
}

$TensorBoardExe = Join-Path $ProjectRoot ".venv\Scripts\tensorboard.exe"
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Url = "http://${HostName}:${Port}/"

Write-Host "TensorBoard logdir: $ResolvedLogDir"
Write-Host "TensorBoard URL: $Url"
Write-Host "Press Ctrl+C in this window to stop the service."

if (-not $NoBrowser) {
    Start-Process $Url
}

if (Test-Path $TensorBoardExe) {
    & $TensorBoardExe --logdir $ResolvedLogDir --host $HostName --port $Port
}
elseif (Test-Path $PythonExe) {
    & $PythonExe -m tensorboard.main --logdir $ResolvedLogDir --host $HostName --port $Port
}
else {
    python -m tensorboard.main --logdir $ResolvedLogDir --host $HostName --port $Port
}
