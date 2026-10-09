#Requires -Version 5.1
<#
.SYNOPSIS
    sttts-gui 開発用起動スクリプト

.DESCRIPTION
    前提確認 → Python 環境同期 / cargo build → GUI を起動する。
    -Mode を省略した場合は起動モードを対話式で尋ねる。
    real モードでは uv sync を毎回実行する(依存が最新なら数秒。pyproject.toml の
    依存追加が自動で反映され、手動の同期が要らない)。cargo build も毎回実行する
    (変更クレートのみ再コンパイルされるため、再ビルド漏れが起きない)。
    PyTorch バックエンドは初回に -Backend で指定すれば backend/.venv に記録され、
    以降は省略してよい。

.EXAMPLE
    .\dev.ps1                       # モードを対話式で選択して起動
    .\dev.ps1 -Mode real            # 実エンジンモードを直接指定(対話なし)
    .\dev.ps1 -Mode real -Backend cu128   # NVIDIA GPU (CUDA 12.8) 用に同期して起動(以降は記録される)
    .\dev.ps1 -DebugBuild -Mode mock
#>
[CmdletBinding()]
param(
    # mock: モデルDLなし / real: Irodori+ASR を実行(初回はモデルDLあり)。
    # 省略した場合は対話式で選択を求める(既定は real)。
    [ValidateSet('mock', 'real')]
    [string]$Mode,

    # PyTorch バックエンド: xpu(Intel Arc・既定) / cu128(NVIDIA CUDA 12.8) / cpu
    # 省略時は前回同期したバックエンド(backend/.venv/.sttts-backend)、なければ xpu。
    [ValidateSet('xpu', 'cu128', 'cpu')]
    [string]$Backend,

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
    # 起動モードを対話式に選択させる(空欄で real)
    Write-Host ""
    Write-Host "起動モードを選択してください:" -ForegroundColor Cyan
    Write-Host "  [1] real : 実エンジンで起動(Irodori + ASR / 初回はモデル自動DL・既定)"
    Write-Host "  [2] mock : モデルDLなしで起動(UI/配線の確認用)"
    while ($true) {
        $choice = Read-Host "選択 [1/2] (空欄=1)"
        switch ($choice.Trim().ToLower()) {
            ''     { return 'real' }
            '1'    { return 'real' }
            'real' { return 'real' }
            '2'    { return 'mock' }
            'mock' { return 'mock' }
            default {
                Write-Host "  'real' または 'mock'(1/2)を入力してください" -ForegroundColor Yellow
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

# --- 1) Python 環境(backend/.venv)。real は毎回 uv sync(最新なら数秒)
$VenvPython = Join-Path $Root 'backend\.venv\Scripts\python.exe'
$BackendMarker = Join-Path $Root 'backend\.venv\.sttts-backend'
if ($Mode -eq 'real') {
    if (-not $PSBoundParameters.ContainsKey('Backend')) {
        $Backend = 'xpu'
        if (Test-Path $BackendMarker) {
            $saved = (Get-Content $BackendMarker -Raw).Trim()
            if ($saved -in @('xpu', 'cu128', 'cpu')) { $Backend = $saved }
        }
    }
    Step "backend の Python 環境を同期します (uv sync --extra $Backend / 初回は数GBのダウンロード)"
    Push-Location (Join-Path $Root 'backend')
    try {
        # --inexact: 手動で足した追加 extra (reazonspeech 等) を消さない
        uv sync --inexact --extra $Backend
        if ($LASTEXITCODE -ne 0) { Fail "uv sync --extra $Backend に失敗しました" }
    }
    finally { Pop-Location }
    Set-Content -Path $BackendMarker -Value $Backend -NoNewline
}
elseif (-not (Test-Path $VenvPython)) {
    Step 'backend/.venv がありません → mock モードは PATH 上の python(標準ライブラリのみ)で動作します'
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Fail 'python が見つかりません。mock モードには Python 3.10+ が必要です。'
    }
}

# --- 2) ビルド(毎回実行。cargo の差分ビルドにより、変更がなければ数秒で終わる)
$Config = if ($DebugBuild) { 'debug' } else { 'release' }
$Exe = Join-Path $Root "target\$Config\sttts-gui.exe"
Step "cargo build ($Config) を実行します(初回のみ数分・以降は差分ビルド)"
Push-Location $Root
try {
    if ($DebugBuild) { cargo build } else { cargo build --release }
    if ($LASTEXITCODE -ne 0) { Fail 'cargo build に失敗しました' }
}
finally { Pop-Location }

# --- 3) 起動
Step "sttts-gui を起動します (mode=$Mode, build=$Config)"
& $Exe "--$Mode" @AppArgs
exit $LASTEXITCODE
