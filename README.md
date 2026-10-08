# sttts-gui

**speech-to-text-to-speech GUI** — マイク音声をリアルタイムに文字起こしし(ストリーミングASR)、
確定文を文チャンクへ分割して [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) で逐次合成・再生する
Windows デスクトップアプリです。UI は Rust 製 **GPUI**(gpui-kit / Zed 系)、モデル実行は
**uv で管理された Python バックエンド**が担い、両者は stdio NDJSON で接続されます。

```
┌─────────────────────────────┐          ┌──────────────────────────────────┐
│ GUI (Rust / gpui-kit)       │  stdin   │ backend (Python / uv)            │
│  ・文字起こし表示(partial/final)│ ◀────── │  mic(48k)→soxr→16k→silero VAD    │
│  ・テキスト発話・モデル選択・履歴│  NDJSON  │  → faster-whisper(kotoba, CPU)   │
│  ・rodio でチャンクWAVを逐次再生│  ──────▶ │  → 文チャンク分割 → TTSキュー     │
└─────────────────────────────┘          │  → irodori-tts (Intel XPU)       │
                                         └──────────────────────────────────┘
```

Irodori-TTS は文単位の非ストリーミング合成のため、**確定文→句点等でチャンク分割→チャンクごとに
逐次合成→完成したチャンクから即座に送出・再生**という疑似ストリーミング構成を採用しています
(Irodori-TTS-Server の SSE モードと同じ発想)。

## 構成

| パス | 内容 |
|---|---|
| `crates/protocol/` | GUI⇄backend の NDJSON メッセージ型(Rust/serde) |
| `crates/gui/` | GPUI クライアント(gpui-kit 0.7 + rodio) |
| `backend/` | Python バックエンド(uv プロジェクト) |
| `backend/src/sttts_server/` | stdio サーバ本体・チャンク分割・エンジン実装 |
| `backend/src/sttts_server/engines/` | `tts_irodori` / `asr_whisper` / `vad_silero` / `mic` / `mock` |
| `output/` | 生成 WAV(チャンクごとに自動保存) |
| `data/config.json` | GUI 設定(モデル・キャプション・自動発話等) |

## セットアップ(Windows 11 + Intel Arc B570)

要件:

- Windows 11 / Intel Arc ドライバ **32.0.101.7028 以上**(XPU 用。古い場合は更新)
- Rust stable(cargo 1.99 で検証)/ MSVC ビルドツール
- [uv](https://docs.astral.sh/uv/) と git

```bat
:: 1) Python 環境(torch 2.10 XPU 含む。初回は数GBダウンロード)
cd backend
uv sync --extra xpu

:: 2) XPU 疎通確認
uv run --no-sync python -c "import torch; print(torch.xpu.is_available())"  :: True になること

:: 3) GUI ビルド(リポジトリルートへ戻る)
cd ..
cargo build --release
```

注意: 本リポジトリの Python 環境はグローバルに入れず、必ず上記 venv(uv)で隔離してください。

## 実行

開発用起動スクリプト `dev.ps1`(Windows PowerShell)。**`-Mode` を省略すると起動モードを対話式で尋ねます**:

```powershell
.\dev.ps1                        # モードを対話式で選択(1: mock / 2: real、空欄で mock)
.\dev.ps1 -Mode real             # 実エンジンモードを直接指定(対話なし・自動化向け)
.\dev.ps1 -Mode real -Sync       # uv sync --extra xpu してから起動
.\dev.ps1 -Build                 # 強制再ビルドして起動
.\dev.ps1 -DebugBuild            # debug プロファイルで起動
```

スクリプトは 前提確認(cargo / uv)→ 必要時のみ `uv sync --extra xpu` と
`cargo build`(バイナリ未生成時は自動)→ GUI 起動、を行います。

直接起動する場合:

```bat
:: モックモード(モデルDLなし。UI/配線の確認用)
cargo run --release -p sttts-gui -- --mock

:: 実エンジンモード(初回発話時に Irodori モデル ≈3GB と ASR モデル ≈1GB を自動DL)
cargo run --release -p sttts-gui -- --real
```

使い方:

1. **入力/出力デバイス**を右パネル上部のセレクタで選択(選択は設定に保存され、
   入力デバイス変更時はマイクセッションが自動で張り直されます)
2. **入力レベルメーター**がマイク稼働中は dB 表示とバーで入力レベルを可視化します
   (ASR に音が入っているかの確認用。緑→黄→赤)
3. 「発話」パネルにテキストを入力し**発話**ボタン → チャンク合成されて順次再生されます
4. **マイク開始** → 話すと部分文字起こし(灰色)→ 確定文(白)が表示され、
   「ASR確定文を自動発話」が ON なら確定文がそのまま TTS されます
5. モデルは右パネルで切替(切替後の最初の発話でロード/再ロード)
6. 「キュー取消」は未合成チャンクを破棄します(合成中の1チャンクは中断できません=ライブラリ仕様)

### バックエンド単体のセルフチェック

```bat
cd backend
uv run --no-sync python -m sttts_server --self-check-tts "こんにちは" --model v4.1-small-mf
uv run --no-sync python -m sttts_server --self-check-asr
```

## モデルカタログ

| エイリアス | 実体 | 備考 |
|---|---|---|
| `v4.1-small-mf` | Aratako/Irodori-TTS-v4.1-Small-MF | MeanFlow 4steps。**既定**(B570 で RTF≈0.9 を実測) |
| `v4.1-small` | Aratako/Irodori-TTS-v4.1-Small | RF 40steps |
| `v4-large` | Aratako/Irodori-TTS-v4-Large | 高品質。bf16 で VRAM ≈6.6GB+、生成は遅め |
| `v4.1-small-int8` / `v4-large-int8` | 量子化版(subfolder int8-weight-only) | OOM 時のフォールバック |

ASR: `kotoba-tech/kotoba-whisper-v2.0-faster`(CTranslate2 / **CPU で動作**)+
silero-vad(ONNX)。VAD 発話終了時にバッファ全体を再デコードして高品質な確定文を作り、
発話中は partial_interval_ms(既定800ms)ごとに部分表示を更新します。

## テスト

```bat
cargo test            :: プロトコル単体 + モックバックエンドとの往復統合テスト
cd backend && uv run --no-sync pytest   :: チャンク分割・設定マージ等
```

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| `torch.xpu.is_available()` が False | Intel ドライバを 32.0.101.7028+ へ更新 |
| TTS でメモリ不足(OOM) | モデルを INT8 版へ / `decode_mode` sequential 維持 / codec の CPU 追い出し(将来設定化) |
| 最初の発話まで数分かかる | 正常です。irodori の依存(torch/transformers 等)の import とモデル構築に冷起動で数分かかります。2回目以降の発話はチャンク1個あたり数秒です(v4.1-Small-MF 実測) |
| マイクを開けない / レートエラー | バックエンドは既定レート(通常48kHz)で開き soxr で16kへ変換します。他アプリの排他占有を解除 |
| ASR の部分表示が遅い | `partial_interval_ms` を調整、または ASR モデルを小型に |
| 発話が止まって進まない | バックエンドログ(GUI 下段)を確認。モデル初回 DL 中は待つ必要があります |
| タスクマネージャに python.exe が残る | GUI を強制終了した場合は子のバックエンドが孤児化します。正常終了は「終了」ボタンから。孤児は手動で終了してください |

## ライセンス・クレジット

- [Irodori-TTS](https://github.com/Aratako/Irodori-TTS)(コード: MIT)
  - モデルのライセンスはチェックポイントごとに異なります。配布・利用の際は各モデルカードを確認してください
    - [v4.1-Small-MF](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF) / [v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) / [v4.1-Small-Quantized](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-Quantized): MIT
    - [v4-Large](https://huggingface.co/Aratako/Irodori-TTS-v4-Large) / [v4-Large-Quantized](https://huggingface.co/Aratako/Irodori-TTS-v4-Large-Quantized): **Gemma Terms of Use**
    - コーデック [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim): MIT
  - 上記 Irodori-TTS の各モデルカードには、ライセンスに加えて倫理的な利用制限があります(本人の同意なく声優・著名人など実在人物の声を複製・なりすましに使わない、誤情報やディープフェイク目的に使わない 等)
- [kotoba-whisper-v2.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0)(Apache-2.0)/ 本アプリが使う CTranslate2 変換版 [kotoba-whisper-v2.0-faster](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-faster)(MIT)
- [silero-vad](https://github.com/snakers4/silero-vad)(MIT)
- gpui-kit(Apache-2.0)/ Zed GPUI(Apache-2.0)
- 本リポジトリのコード: ライセンスは未確定(リポジトリオーナーが確定してください)
