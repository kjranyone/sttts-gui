#Requires -Version 5.1
<#
.SYNOPSIS
    sttts-gui 開発用起動スクリプト

.DESCRIPTION
    cargo build → GUI を起動する。Python も uv も要らない(バックエンドは GUI と同じプロセスの Rust)。
    -Mode を省略した場合は起動モードを対話式で尋ねる。
    cargo build は毎回実行する(変更クレートのみ再コンパイルされるため、再ビルド漏れが起きない)。
    モデル(Irodori-TTS / ASR)は初回の起動時に自動でダウンロードされる。

.EXAMPLE
    .\dev.ps1                       # モードを対話式で選択して起動
    .\dev.ps1 -Mode real            # 実エンジンモードを直接指定(対話なし)
    .\dev.ps1 -DebugBuild -Mode mock
#>
[CmdletBinding()]
param(
    # mock: モデルDLなし / real: Irodori+ASR を実行(初回はモデルDLあり)。
    # 省略した場合は対話式で選択を求める(既定は real)。
    [ValidateSet('mock', 'real')]
    [string]$Mode,

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

# --- 0.5) モード選択(未指定なら対話式)
if (-not $PSBoundParameters.ContainsKey('Mode')) {
    $Mode = Select-Mode
}

# --- 1) ビルド(毎回実行。cargo の差分ビルドにより、変更がなければ数秒で終わる)
$Config = if ($DebugBuild) { 'debug' } else { 'release' }
$Exe = Join-Path $Root "target\$Config\sttts-gui.exe"
Step "cargo build ($Config) を実行します(初回のみ数分・以降は差分ビルド)"
Push-Location $Root
try {
    if ($DebugBuild) { cargo build } else { cargo build --release }
    if ($LASTEXITCODE -ne 0) { Fail 'cargo build に失敗しました' }
}
finally { Pop-Location }

# --- 2) 起動
Step "sttts-gui を起動します (mode=$Mode, build=$Config)"
& $Exe "--$Mode" @AppArgs
exit $LASTEXITCODE
