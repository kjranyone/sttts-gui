#Requires -Version 5.1
<#
.SYNOPSIS
    Development launcher for sttts-gui

.DESCRIPTION
    gui: runs cargo build for the GUI, then launches it. No Python or uv needed (the backend is Rust, in the same process as the GUI).
    say: builds the CLI (sttts-say) on its own, as the release does (without the microphone / ASR code).
    If no command is given, you are asked what to do. With -Mode alone, the command is gui.
    cargo build runs every time (only changed crates are recompiled, so a stale build can't slip through).
    Models (Irodori-TTS / ASR) are downloaded automatically on the first launch.
    dev.sh is the same launcher for macOS / Linux (sttts-say only; the GUI is Windows-only).

.EXAMPLE
    .\dev.ps1                       # choose what to do interactively
    .\dev.ps1 -Mode real            # GUI with the real engine, no prompt
    .\dev.ps1 gui -DebugBuild -Mode mock
    .\dev.ps1 say                   # build target\release\sttts-say.exe
#>
[CmdletBinding()]
param(
    # gui: build and launch the GUI / say: build the CLI (sttts-say).
    # If omitted, gui when -Mode is given, otherwise you are asked to choose.
    [Parameter(Position = 0)]
    [ValidateSet('gui', 'say')]
    [string]$Command,

    # GUI only. mock: no model download / real: runs Irodori + ASR (downloads models on first run).
    # If omitted: real with "gui", otherwise you are asked to choose.
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
function Select-Task {
    # Ask what to do (empty = GUI with the real engine)
    Write-Host ""
    Write-Host "Choose what to do:" -ForegroundColor Cyan
    Write-Host "  [1] real : launch the GUI with the real engine (Irodori + ASR; models download on first run; default)"
    Write-Host "  [2] mock : launch the GUI without models (for checking the UI and wiring)"
    Write-Host "  [3] say  : build the CLI (sttts-say)"
    while ($true) {
        $choice = Read-Host "Choice [1/2/3] (empty = 1)"
        switch ($choice.Trim().ToLower()) {
            ''     { return 'real' }
            '1'    { return 'real' }
            'real' { return 'real' }
            '2'    { return 'mock' }
            'mock' { return 'mock' }
            '3'    { return 'say' }
            'say'  { return 'say' }
            default {
                Write-Host "  Enter 1, 2 or 3 (real / mock / say)" -ForegroundColor Yellow
            }
        }
    }
}
function Build([string]$Package) {
    # One package per cargo call: building the GUI and sttts-say together would unify their features
    # and put the microphone / ASR code into sttts-say
    Step "Running cargo build -p $Package ($Config) (a few minutes the first time, incremental afterwards)"
    Push-Location $Root
    try {
        if ($DebugBuild) { cargo build -p $Package } else { cargo build --release -p $Package }
        if ($LASTEXITCODE -ne 0) { Fail 'cargo build failed' }
    }
    finally { Pop-Location }
}

# --- 0) Prerequisites
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Fail 'cargo not found. Please install Rust (rustup).'
}

# --- 0.5) What to do (ask if not given)
if (-not $Command) {
    $Task = if ($Mode) { $Mode } else { Select-Task }
    if ($Task -eq 'say') { $Command = 'say' } else { $Command = 'gui'; $Mode = $Task }
}
if ($Command -eq 'say' -and ($Mode -or $AppArgs)) {
    Fail '-Mode and extra arguments are for the GUI. "say" only builds sttts-say.'
}
$Config = if ($DebugBuild) { 'debug' } else { 'release' }

# --- 1) CLI: build only
if ($Command -eq 'say') {
    Build 'sttts-say'
    $Exe = Join-Path $Root "target\$Config\sttts-say.exe"
    # A background sttts-say process of the previous build is replaced by itself on the next call
    Step "Built $Exe"
    exit 0
}

# --- 2) GUI: build every time (cargo's incremental build finishes in seconds when nothing changed), then launch
if (-not $Mode) { $Mode = 'real' }
Build 'sttts-gui'
$Exe = Join-Path $Root "target\$Config\sttts-gui.exe"
Step "Launching sttts-gui (mode=$Mode, build=$Config)"
& $Exe "--$Mode" @AppArgs
exit $LASTEXITCODE
