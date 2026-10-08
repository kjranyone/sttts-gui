#Requires -Version 5.1
<#
.SYNOPSIS
    sttts-gui 開発用起動スクリプト

.DESCRIPTION
    前提確認 → (必要なら) Python 環境同期 / cargo build → GUI を起動する。
    -Mode を省略した場合は起動モードを対話式で尋ねる。

.EXAMPLE
    .\dev.ps1                       # モードを対話式で選択して起動
    .\dev.ps1 -Mode real            # 実エンジンモードを直接指定(対話なし)
    .\dev.ps1 -Mode real -Sync      # uv sync --extra xpu を実行してから起動
    .\dev.ps1 -Build                # 強制再ビルドしてから起動
    .\dev.ps1 -DebugBuild -Mode mock
#>
[CmdletBinding()]
param(
    # mock: モデルDLなし / real: Irodori+ASR を実行(初回はモデルDLあり)。
    # 省略した場合は対話式で選択を求める。
    [ValidateSet('mock', 'real')]
    [string]$Mode,

    # backend の Python 環境を同期する (uv sync --extra xpu)
    [switch]$Sync,

    # cargo build を強制する(通常はバイナリが無い場合のみ自動ビルド)
    [switch]$Build,

    # デバッグプロファイル (target/debug) を使う(既定は release)
    [switch]$DebugBuild,

    # GUI 実行ファイルへの追加引数(そのまま透過)
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
    # 起動モードを対話式に選択させる(空欄で mock)
    Write-Host ""
    Write-Host "起動モードを選択してください:" -ForegroundColor Cyan
    Write-Host "  [1] mock : モデルDLなしで起動(UI/配線の確認用・既定)"
    Write-Host "  [2] real : 実エンジンで起動(Irodori + ASR / 初回はモデル自動DL)"
    while ($true) {
        $choice = Read-Host "選択 [1/2] (空欄=1)"
        switch ($choice.Trim().ToLower()) {
            ''     { return 'mock' }
            '1'    { return 'mock' }
            'mock' { return 'mock' }
            '2'    { return 'real' }
            'real' { return 'real' }
            default {
                Write-Host "  'mock' または 'real'(1/2)を入力してください" -ForegroundColor Yellow
            }
        }
    }
}

# --- 0) 前提コマンド
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Fail 'cargo が見つかりません。Rust(rustup) をインストールしてください。'
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Fail 'uv が見つかりません。https://docs.astral.sh/uv/ からインストールしてください。'
}

# --- 0.5) モード選択(未指定なら対話式)
if (-not $PSBoundParameters.ContainsKey('Mode')) {
    $Mode = Select-Mode
}

# --- 1) Python 環境(backend/.venv)
$VenvPython = Join-Path $Root 'backend\.venv\Scripts\python.exe'
$NeedSync = $Sync -or (($Mode -eq 'real') -and -not (Test-Path $VenvPython))
if ($NeedSync) {
    Step 'backend の Python 環境を同期します (uv sync --extra xpu / 初回は数GBのダウンロード)'
    Push-Location (Join-Path $Root 'backend')
    try {
        uv sync --extra xpu
        if ($LASTEXITCODE -ne 0) { Fail 'uv sync --extra xpu に失敗しました' }
    }
    finally { Pop-Location }
}
elseif (-not (Test-Path $VenvPython)) {
    if ($Mode -eq 'mock') {
        Step 'backend/.venv がありません → mock モードは PATH 上の python(標準ライブラリのみ)で動作します'
        if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
            Fail 'python が見つかりません。mock モードには Python 3.10+ が必要です。'
        }
    }
    else {
        Fail 'backend/.venv がありません。-Sync を付けて実行してください(例: .\dev.ps1 -Mode real -Sync)'
    }
}

# --- 2) ビルド
$Config = if ($DebugBuild) { 'debug' } else { 'release' }
$Exe = Join-Path $Root "target\$Config\sttts-gui.exe"
if ($Build -or -not (Test-Path $Exe)) {
    Step "cargo build ($Config) を実行します(初回は数分かかります)"
    Push-Location $Root
    try {
        if ($DebugBuild) { cargo build } else { cargo build --release }
        if ($LASTEXITCODE -ne 0) { Fail 'cargo build に失敗しました' }
    }
    finally { Pop-Location }
}

# --- 3) 起動
Step "sttts-gui を起動します (mode=$Mode, build=$Config)"
& $Exe "--$Mode" @AppArgs
exit $LASTEXITCODE
