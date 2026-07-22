param(
    [string]$Profile = 'dataset/test_user.json',
    [string]$Persona = 'dataset/test_agent.json',
    [string]$OutputDir = 'data/profile_layer_ablation',
    [int]$Port = 18201
)

$ErrorActionPreference = 'Stop'

$workspace = Split-Path -Parent $PSScriptRoot
$python = Join-Path $workspace '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python runtime not found: $python"
}

$outputPath = Join-Path $workspace $OutputDir
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

$resultFiles = @(
    'profile_layer_ablation_results.json',
    'profile_layer_ablation_summary.json',
    'report.md'
)
foreach ($fileName in $resultFiles) {
    $target = Join-Path $outputPath $fileName
    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Force
    }
}

& $python -m src.experiments.profile_layer_ablation `
    --profile $Profile `
    --persona $Persona `
    --output-dir $OutputDir `
    --overwrite `
    --port $Port

exit $LASTEXITCODE
