param(
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot),
    [int]$Port = 18201
)

$ErrorActionPreference = 'Stop'

# Stop only the process listening on the experiment service port.
$listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($listener in $listeners) {
    Stop-Process -Id $listener.OwningProcess -Force
}

Start-Sleep -Milliseconds 300

# The experiment uses a fresh embedded Milvus store for every service request.
$memoryPath = Join-Path $Workspace 'data\milvus_memory.db'
if (Test-Path -LiteralPath $memoryPath) {
    Remove-Item -LiteralPath $memoryPath -Recurse -Force
}

$python = Join-Path $Workspace '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python runtime not found: $python"
}

Start-Process -FilePath $python -ArgumentList 'app.py' -WorkingDirectory $Workspace -WindowStyle Hidden

for ($attempt = 1; $attempt -le 30; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -Method Get -TimeoutSec 2
        if ($health.status -eq 'ok') {
            exit 0
        }
    } catch {
        Start-Sleep -Milliseconds 500
    }
}

throw "Experiment service did not become healthy on port $Port."
