<#
.SYNOPSIS
从双人对话 JSON 中生成五层粗粒度用户画像。

.DESCRIPTION
调用 src.experiments.coarse_user_profile_generation，清洗对话后请求大模型，
并生成只包含 core、regulation、cognition、identity、behavior 的中文 JSON。

.PARAMETER Realtalk
输入对话 JSON 路径。相对路径以项目根目录为基准。

.PARAMETER Output
可选的输出 JSON 路径。省略时保存到
user/{用户}_{智能体}_coarse_profile.json。

.PARAMETER Config
模型配置文件路径，默认 config.ini。

.PARAMETER MaxUtterances
最多发送给模型的对话消息数，默认 180。

.PARAMETER MaxChars
清洗后对话 JSON 的最大字符数，默认 24000。

.EXAMPLE
./scripts/run_coarse_user_profile_generation.ps1 `
  -Realtalk dataset/Chat_1_Emi_Elise.json

.EXAMPLE
./scripts/run_coarse_user_profile_generation.ps1 `
  -Realtalk dataset/Chat_1_Emi_Elise.json `
  -Output dataset/output/user/emi_elise_coarse_profile.json
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$Realtalk,

    [string]$Output = '',
    [string]$Config = 'config.ini',
    [ValidateRange(1, 1000000)]
    [int]$MaxUtterances = 180,
    [ValidateRange(1, 100000000)]
    [int]$MaxChars = 24000
)

$ErrorActionPreference = 'Stop'

$workspace = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $workspace '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $venvPython) {
    $python = $venvPython
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $pythonCommand) {
        throw 'Python runtime not found. Create .venv or add Python to PATH.'
    }
    $python = $pythonCommand.Source
}

function Resolve-WorkspacePath {
    param([string]$PathValue)

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $workspace $PathValue))
}

$realtalkPath = Resolve-WorkspacePath $Realtalk
if (-not (Test-Path -LiteralPath $realtalkPath -PathType Leaf)) {
    throw "Input dialogue file not found: $realtalkPath"
}

$configPath = Resolve-WorkspacePath $Config
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "Model config file not found: $configPath"
}

$arguments = @(
    '-m',
    'src.experiments.coarse_user_profile_generation',
    '--realtalk', $realtalkPath,
    '--config', $configPath,
    '--max-utterances', $MaxUtterances,
    '--max-chars', $MaxChars
)

if ($Output) {
    $outputPath = Resolve-WorkspacePath $Output
    $arguments += @('--output', $outputPath)
}

Push-Location $workspace
try {
    & $python @arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
