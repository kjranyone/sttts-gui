#Requires -Version 5.1
<#
.SYNOPSIS
    Development launcher for sttts-gui

.DESCRIPTION
    Runs cargo build, then launches the GUI. No Python or uv needed (the backend is Rust, in the same process as the GUI).
    If -Mode is omitted, the launch mode is asked interactively.
    cargo build runs every time (only changed crates are recompiled, so a stale build can't slip through).
    Models (Irodori-TTS / ASR) are downloaded automatically on the first launch.

.EXAMPLE
    .\dev.ps1                       # choose the mode interactively, then launch
    .\dev.ps1 -Mode real            # real engine, no prompt
    .\dev.ps1 -DebugBuild -Mode mock
#>
[CmdletBinding()]
param(
    # mock: no model download / real: runs Irodori + ASR (downloads models on first run).
    # If omitted, you are asked to choose (default: real).
    [ValidateSet('mock', 'real')]
    [string]$Mode,

    # Use the debug profile (target/debug) instead of release
    [switch]$DebugBuild,

    # Extra arguments passed through to the GUI executable
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArgs
)

$ErrorActionPreference = 'Stop'
$Root = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }

function Fail([string]$Message) {
    Write-Host "[dev.ps1] ERROR: $Message" -ForegroundColor Red
    exit 1
}
function Step([string]$Message) {
    Write-Host "[dev.ps1] $Message" -ForegroundColor Cyan
}
function Select-Mode {
    # Ask for the launch mode (empty = real)
    Write-Host ""
    Write-Host "Choose a launch mode:" -ForegroundColor Cyan
    Write-Host "  [1] real : real engine (Irodori + ASR; models download on first run; default)"
    Write-Host "  [2] mock : no model download (for checking the UI and wiring)"
    while ($true) {
        $choice = Read-Host "Choice [1/2] (empty = 1)"
        switch ($choice.Trim().ToLower()) {
            ''     { return 'real' }
            '1'    { return 'real' }
            'real' { return 'real' }
            '2'    { return 'mock' }
            'mock' { return 'mock' }
            default {
                Write-Host "  Enter 'real' or 'mock' (1/2)" -ForegroundColor Yellow
            }
        }
    }
}

# --- 0) Prerequisites
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Fail 'cargo not found. Please install Rust (rustup).'
}

# --- 0.5) Mode (ask if not given)
if (-not $PSBoundParameters.ContainsKey('Mode')) {
    $Mode = Select-Mode
}

# --- 1) Build (every time; cargo's incremental build finishes in seconds when nothing changed)
$Config = if ($DebugBuild) { 'debug' } else { 'release' }
$Exe = Join-Path $Root "target\$Config\sttts-gui.exe"
Step "Running cargo build ($Config) (a few minutes the first time, incremental afterwards)"
Push-Location $Root
try {
    if ($DebugBuild) { cargo build } else { cargo build --release }
    if ($LASTEXITCODE -ne 0) { Fail 'cargo build failed' }
}
finally { Pop-Location }

# --- 2) Launch
Step "Launching sttts-gui (mode=$Mode, build=$Config)"
& $Exe "--$Mode" @AppArgs
exit $LASTEXITCODE
