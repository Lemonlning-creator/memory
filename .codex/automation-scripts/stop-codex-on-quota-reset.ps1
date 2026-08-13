param(
    [ValidateSet('Check', 'Stop')]
    [string]$Mode = 'Check',

    [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'

function Get-CodexQuotaSnapshot {
    $codexExe = (Get-Command codex.exe -ErrorAction Stop | Select-Object -First 1).Source

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $codexExe
    $startInfo.Arguments = 'app-server --stdio'
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardInput = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo

    try {
        if (-not $process.Start()) {
            throw 'Unable to start the Codex app server.'
        }

        $initialize = @{
            id = 1
            method = 'initialize'
            params = @{
                clientInfo = @{
                    name = 'quota-reset-monitor'
                    version = '1.0'
                }
                capabilities = @{
                    experimentalApi = $true
                }
            }
        } | ConvertTo-Json -Compress -Depth 6

        $readLimits = @{
            id = 2
            method = 'account/rateLimits/read'
            params = @{}
        } | ConvertTo-Json -Compress -Depth 4

        $process.StandardInput.WriteLine($initialize)
        $process.StandardInput.WriteLine($readLimits)
        $process.StandardInput.Flush()

        $deadline = [DateTime]::UtcNow.AddSeconds(20)
        while ([DateTime]::UtcNow -lt $deadline) {
            $readTask = $process.StandardOutput.ReadLineAsync()
            if (-not $readTask.Wait(5000)) {
                continue
            }

            $line = $readTask.Result
            if ([string]::IsNullOrWhiteSpace($line)) {
                if ($process.HasExited) {
                    throw 'Codex app server exited before returning quota data.'
                }
                continue
            }

            try {
                $message = $line | ConvertFrom-Json
            }
            catch {
                continue
            }

            if ($message.id -eq 2) {
                if ($message.error) {
                    throw ('Codex quota request failed: ' + ($message.error | ConvertTo-Json -Compress))
                }

                $snapshot = $message.result.rateLimits
                $window = $snapshot.primary
                if ($null -eq $window) {
                    throw 'Codex did not return a primary quota window.'
                }

                $usedPercent = [int]$window.usedPercent
                return [pscustomobject]@{
                    limitId = $snapshot.limitId
                    usedPercent = $usedPercent
                    remainingPercent = [Math]::Max(0, 100 - $usedPercent)
                    resetsAt = $window.resetsAt
                    resetsAtLocal = if ($null -ne $window.resetsAt) {
                        [DateTimeOffset]::FromUnixTimeSeconds([long]$window.resetsAt).ToLocalTime().ToString('o')
                    }
                    else {
                        $null
                    }
                }
            }
        }

        throw 'Timed out while reading the Codex quota.'
    }
    finally {
        try {
            $process.StandardInput.Close()
        }
        catch {
        }

        try {
            if (-not $process.HasExited -and -not $process.WaitForExit(2000)) {
                $process.Kill()
            }
        }
        catch {
        }

        $process.Dispose()
    }
}

$quota = Get-CodexQuotaSnapshot

if ($Mode -eq 'Check' -or $quota.remainingPercent -ne 100) {
    [pscustomobject]@{
        action = if ($quota.remainingPercent -eq 100) { 'ready' } else { 'pending' }
        quota = $quota
        stoppedProcessIds = @()
    } | ConvertTo-Json -Compress -Depth 5
    exit 0
}

$targets = @(
    Get-CimInstance Win32_Process -Filter "Name = 'codex.exe'" |
        Where-Object { $_.CommandLine -match '(?i)(^|\s)app-server(\s|$)' }
)

$targetIds = @($targets | ForEach-Object { [int]$_.ProcessId })

if (-not $WhatIf) {
    foreach ($targetId in $targetIds) {
        Stop-Process -Id $targetId -Force -ErrorAction SilentlyContinue
    }
}

[pscustomobject]@{
    action = if ($WhatIf) { 'would-stop' } else { 'stopped' }
    quota = $quota
    stoppedProcessIds = $targetIds
} | ConvertTo-Json -Compress -Depth 5
