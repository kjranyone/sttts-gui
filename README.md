# sttts-gui

**speech-to-text-to-speech GUI** — マイク音声をリアルタイムに文字起こしし(ストリーミングASR)、
確定文を文チャンクへ分割して [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) で逐次合成・再生する
Windows デスクトップアプリです。UI は Rust 製 **GPUI**(gpui-kit / Zed 系)、マイク・VAD・ASR・TTS も
すべて **同じ Rust プロセス**で動きます。Python も PyTorch も不要です。

```
┌──────────────────────────────────────────────────────────────────────────┐
│ sttts-gui (1 プロセス)                                                      │
│  GUI (gpui-kit)  ◀── メッセージ(チャネル) ──▶  crates/engine                │
│   ・文字起こし表示(partial/final)             mic(cpal)→16k→Silero VAD       │
│   ・テキスト発話・声・履歴                      → ASR ワーカー                  │
│   ・rodio でチャンク WAV を逐次再生               Nemotron(ONNX) / kotoba     │
│   ・発話終了→初音 の計測表示                       (burn・GPU) / Gemini(クラウド)│
│                                                 → 文チャンク分割 → TTS キュー │
│                                                 → Irodori-TTS(burn・GPU)    │
└──────────────────────────────────────────────────────────────────────────┘
```

Irodori-TTS は文単位の非ストリーミング合成のため、**確定文→句点等でチャンク分割→チャンクごとに
逐次合成→完成したチャンクから即座に送出・再生**という疑似ストリーミング構成を採用しています
(Irodori-TTS-Server の SSE モードと同じ発想)。

## 構成

| パス | 内容 |
|---|---|
| `crates/gui/` | GPUI クライアント(gpui-kit 0.7 + rodio)。エンジンをプロセス内で起動する(`backend.rs`) |
| `crates/engine/` | バックエンド本体。設定・文チャンク分割・TTS ワーカー・ライブセッション(VAD→ASR)・投機的 TTS・発話表現。外界は `Platform` トレイトで注入するので、実モデル・実デバイス無しでテストできる |
| `crates/protocol/` | GUI⇄エンジンのメッセージ型(Rust/serde) |
| `crates/irodori/` | Irodori-TTS の純 Rust 推論(burn / wgpu。設計・精度・速度は `docs/irodori-rs.md`) |
| `crates/whisper/` | kotoba-whisper-v2.0 の純 Rust 推論(burn / wgpu) |
| `crates/nemotron/` | Nemotron 3.5 ASR streaming(ONNX、onnxruntime / CPU)の Rust 実装 |
| `crates/gemini/` | Gemini Live API(クラウド ASR)のクライアント |
| `crates/audio/` | マイク入力(cpal)・リサンプル・Silero VAD・WAV ソース |
| `crates/hub/` | HuggingFace Hub のキャッシュ探索とモデル自動ダウンロード |
| `tools/reference/` | 開発用: PyTorch の参照出力を書き出す Python スクリプト(数値一致テスト用。アプリの実行には使わない) |
| `output/` | 生成 WAV(チャンクごとに自動保存) |
| `data/config.json` | GUI 設定(モデル・キャプション・自動発話等) |
| `data/backend.json` | 任意。エンジンの上級設定(下記) |

## セットアップ(Windows 11 + Intel Arc B570)

要件:

- Windows 11 / GPU は Vulkan が使えるもの(Intel Arc、NVIDIA、AMD。Intel Arc B570 で検証)
- Rust stable(1.95 以上)/ MSVC ビルドツール

```bat
cargo build --release
```

これだけです。モデル(Irodori-TTS ≈3GB、コーデック ≈0.4GB、ASR は選んだエンジンの分)は
**初回起動時に自動でダウンロード**されます(HuggingFace のキャッシュ `~/.cache/huggingface/hub` を共有)。

## 実行

開発用起動スクリプト `dev.ps1`(Windows PowerShell)。**`-Mode` を省略すると起動モードを対話式で尋ねます(空欄で real)**:

```powershell
.\dev.ps1                        # モードを対話式で選択(1: real / 2: mock、空欄で real)
.\dev.ps1 -Mode real             # 実エンジンモードを直接指定(対話なし・自動化向け)
.\dev.ps1 -DebugBuild            # debug プロファイルで起動
```

スクリプトは `cargo build`(毎回実行。変更がなければ差分ビルドで数秒)→ GUI 起動、を行います。
コード編集後に古いバイナリが起動することはありません。

直接起動する場合:

```bat
:: モックモード(モデルDLなし。UI/配線の確認用)
cargo run --release -p sttts-gui -- --mock

:: 実エンジンモード(初回発話時にモデルを自動DL)
cargo run --release -p sttts-gui -- --real
```

使い方(画面は「何を話したか / どう伝えるか / どの声で届けるか」を分けて見せます):

- **ストリーム(中央)**: 1枚のカード = 話した内容(上段・桜色)と、それを届けた声(下段・藤色)。
  認識の途中経過は薄い文字で、確定すると白になります。届けた後は ▶(もう一度聞く)/
  ↻(今の声でもう一度話す)/ 訂正(文を入力欄へ移して直す)が使えます
- **入力欄(下)**: 文字で話すときに入力して **話す**(Ctrl+Enter)。発話中は **止める** で
  再生中・未再生の音声と未合成チャンクを破棄します(合成中の1チャンクは Irodori の仕様上
  中断できませんが、完了後に結果を捨てるので鳴りません)
- **右レール**
  - **ライブ**: ライブ開始/停止とマイクの入力レベル
  - **音声キュー**: 「自動再生」ON で確定文をそのまま発話。OFF ではカードで止まり、
    「この内容で話す」「訂正する」「話さない」を選ぶ。「テンポと間を再現」は
    元音声の速さを Irodori に渡します
  - **声**: 声バンクの選択と、話し方の指示(Irodori の caption)
  - **認識**: クラウド(Gemini)/ ローカルの切替。Gemini の API キーはここで入力します
- **タイトルバー**: 全体の状態(準備完了 / 読み込み中 / エラー)、ライブ中の表示、
  **応答**(話し終わってから最初の音声が再生キューに入るまで。直近値と直近20回の中央値)、
  ⚙ 詳細設定、ログの開閉
- **詳細設定(⚙)**: 入出力デバイス、音声合成モデル、seed など、環境で一度決めれば
  普段は触らない設定
- **ステータスバー**: 認識時間 / 初音まで / 合成速度(RTF)/ VRAM・RAM。エラーが出るとログのボタンに
  件数が出ます。ログは `data/gui.log` にも保存されます(起動ごとに作り直し)

入力欄の「演技」パレット、転写文と合成用絵文字の扱い、対応する全絵文字は
[Irodori への表現指示](docs/irodori-annotations.md)を参照してください。
目的、データ経路、現時点の制約は[発話の表現を再構築する設計と実装状況](docs/acting-reconstruction-design.md)に記録しています。

> **ヘッドホン推奨**: マイクは TTS 再生中も開いたままです。スピーカー再生だと合成音声を
> マイクが拾い、それが文字起こし → 自動発話されてループします(エコーキャンセル未実装)。
> デモ・収録はヘッドホンで行ってください。

## 声のバンク(voice cloning)

参照音声(wav / flac、10秒程度・話者の声)をアプリのウィンドウへドラッグ&ドロップするか、
右レール「声」の「＋」から選ぶと `data/voices/` に取り込まれ、そのまま選択されます。
画像(png / jpg / webp)を落とすと選択中の声のアイコンになり、音声と一緒に落とせば
新しい声に付きます。「ゴミ箱」で削除できます。Explorer での管理は不要です。選択すると Irodori-TTS は
その音声を話者参照として合成し、話し方を模倣します。未選択(既定の声)は
キャプション/自動音質での合成になります。選択は `data/config.json` に保存されます。
参照音声の符号化結果はプロセス内にキャッシュされ、同じ声を使い続ける限り再計算しません。

注意: 参照音声は本人の同意のある声のみ使ってください(モデルカードの利用制限参照)。

## 上級設定(`data/backend.json`)

GUI に UI の無い設定は `data/backend.json`(任意。環境変数 `STTTS_CONFIG` で別パス)に書くと、
起動時に既定値へマージされます(GUI から送られる設定はその上に適用)。例:

```json
{
  "asr": { "engine": "nemotron", "vad_min_silence_ms": 280 },
  "pipeline": { "first_chunk_mora_max": 12, "speculative_tts": false },
  "tts": { "warmup": true, "sampling": { "duration_scale": 1.1 } }
}
```

| キー | 既定 | 説明 |
|---|---|---|
| `asr.engine` | `kotoba` | `kotoba`(kotoba-whisper、GPU)/ `nemotron`(ONNX、CPU)/ `gemini`(クラウド、要 APIキー)。GUI の「認識」ドロップダウンでも切替可 |
| `asr.model` | `kotoba-tech/kotoba-whisper-v2.0` | kotoba 用の HF リポジトリ |
| `asr.final_beam_size` | `2` | kotoba の確定デコードのビーム幅(1 にすると少し速い) |
| `asr.partial_interval_ms` | `800` | 途中経過デコードの間隔。`0` で無効 |
| `asr.preload` | `true` | 起動時に ASR をロードして「マイク開始」を即座に使えるようにする |
| `asr.vad_min_silence_ms` | `280` | 無音がこの長さ続いたら発話終了。短いほど速いが文中の間で切れやすい |
| `asr.vad_threshold` | `0.5` | Silero VAD のしきい値 |
| `asr.nemotron_model_dir` | `null` | Nemotron の ONNX ディレクトリ(null で HF から自動DL。自前exportグラフはここへ) |
| `asr.nemotron_chunk_ms` | `320` | ストリーミングチャンク。HF パッケージは 320 のみ |
| `asr.nemotron_precision` | `fp16` | `int8` は精度劣化するため非推奨 |
| `asr.nemotron_threads` | `4` | onnxruntime のスレッド数 |
| `asr.gemini_api_key` | `null` | AI Studio の API キー。**通常は GUI で入力する**(「認識」で Gemini を選ぶと「キー」欄が出る。`data/config.json` に Windows DPAPI で暗号化保存される)。GUI 未入力なら backend.json のこの値 → 環境変数 `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
| `asr.gemini_mode` | `VERBATIM` | 話し方を残す逐語転写。`SMART`=フィラー除去・句読点整形 |
| `asr.gemini_timeout_s` | `20` | 1発話の確定待ちタイムアウト |
| `pipeline.first_chunk_mora_min` / `max` | `8` / `12` | 先頭チャンクを読点または約 8〜12 モーラの文節境界で切る(`max=0` で無効) |
| `pipeline.chunk_min_chars` | `16` | 2チャンク目以降の最小文字数 |
| `pipeline.chunk_max_chars` | `80` | これを超える塊は読点 / 文節境界で分割(句読点の無い ASR 出力対策) |
| `pipeline.speculative_tts` | `false` | 投機的 TTS(下記) |
| `pipeline.speculative_stable_partials` | `2` | 同じ先頭チャンクが何回連続したら先行合成するか |
| `pipeline.performance_enabled` | `true` | 元音声の速さと間を Irodori の発話単位指示へ写す。GUI の「テンポと間を再現」で切替 |
| `pipeline.performance_wait_ms` | `150` | ASR 確定後に表現分析を待つ上限。超過時は表現を付けず発話 |
| `tts.warmup` | `true` | モデル決定時にロード + 短文合成を先行して初回の待ちを無くす |
| `tts.num_steps` | `null` | MeanFlow のステップ数(null で checkpoint 既定の 4) |
| `tts.sampling` | `{}` | Irodori の `SamplingRequest` 項目を上書き(GUI 未対応でも使える)。使える項目: `num_steps` `duration_scale` `seconds` `min_seconds` `max_seconds` `max_ref_seconds` `ref_normalize_db` `ref_ensure_max` `trim_tail` `tail_window_size` `tail_std_threshold` `tail_mean_threshold` `watermark`。`text` / `caption` / `ref_*` / `no_ref` / `seed` は発話ごとにアプリが決めるため指定不可、知らない項目もエラーで知らせます(黙って捨てません)。設定変更は次の発話から反映 |

### 何が速くなったか

- **ASR を VAD スレッドから分離**: デコードは専用ワーカー。確定(final)を最優先し、
  途中経過(partial)は最新1件に合体。確定が来た発話の partial は待機中・処理中とも破棄。
  音声キューは無制限で、ASR が遅れても音声は捨てません。
- **マイクを先に開く(mic-first)**: ASR のロードと並行してマイクを開き、ロード中の音声は
  保持してロード完了後に流します。レベルメーターはロード中も止まりません。
- **先頭チャンクを短く**: Irodori は1チャンク全体を一括生成するので、初音までの時間は
  先頭チャンクの長さにほぼ比例します。先頭だけ読点か約 8〜12 モーラで切り、以降は大きめに。
  全角 `！？` も文末として扱います。
- **seed はリクエスト単位**: ランダム seed のときも1回の発話内の全チャンクで同じ seed を
  使うため、参照音声なしでもチャンク間で声質が変わりません。
- **投機的 TTS(既定 OFF)**: 同じ先頭チャンクが `speculative_stable_partials` 回連続した
  partial から先頭チャンクを先行合成します。結果はエンジン内に保持し、確定文の先頭チャンク・
  声・設定が**完全一致したときだけ**再生に回します。不一致なら破棄するので、誤った
  音声が鳴ることはありません(外れた場合は GPU 時間を1チャンク分無駄にします)。
- **Nemotron の partial は続きから再開**: 伸びていく発話は直前のデコード状態から再開するので、
  partial のコストが発話の長さに依存しません。

## レイテンシ KPI

主 KPI は **発話終了→初音** = 話し終わり → 最初の音声チャンクが再生キューに入るまで。
エンジンが計測フィールドをメッセージに載せ(`asr_final.vad_wait_ms` / `asr_ms`、
`tts_audio.first_chunk_ms` / `e2e_ms` / `rtf` / `stages` 等)、GUI は受信→再生キュー投入の時間を足して
ヘッダに表示します。内訳はおおよそ:

```
発話終了→初音 ≈ VAD 待ち(≈ vad_min_silence_ms) + ASR 確定デコード + TTS 先頭チャンク合成
```

## モデルカタログ

TTS: Irodori-TTS **v4.1 Small MeanFlow**(`v4.1-small-mf`、Aratako/Irodori-TTS-v4.1-Small-MF)のみ。
Intel Arc B570 で RTF ≈ 0.3(定常)。v4.1-small(RF)・v4-large・INT8 版は Rust 版では扱いません
(MeanFlow のチェックポイントのみ対応)。

ASR(`asr.engine`):

| エンジン | 実体 | 備考 |
|---|---|---|
| `kotoba`(既定) | `kotoba-tech/kotoba-whisper-v2.0`(Apache-2.0)を burn で実行 | GPU(wgpu)。句読点は出ない。10 秒の発話で約 4 秒(encoder が 30 秒窓固定で律速)。速度改善は今後の課題 |
| `nemotron` | `nemotron-3.5-asr-streaming-0.6b` の ONNX export(cache-aware FastConformer-RNNT、onnxruntime / CPU、コード Apache-2.0 / 重み OpenMDW-1.1) | **句読点をネイティブ出力**・whisper large-v3 級の精度。一括デコード RTF ≈ 0.35(20 論理コア・4 スレッド)。モデル ≈2.5GB(fp16)でロードに数秒・メモリ約 5GB。**既知の弱点: 母音のみの連続(「あいうえお」等)を正しく認識しない** |
| `gemini` | Google AI Studio「Gemini 3.5 Transcribe Live」(`gemini-3.5-transcribe-live` / Live API WebSocket) | **クラウド**。発話中に約100ms単位の PCM を送り、途中結果を表示し、ローカル VAD の終了時に確定。既定は `VERBATIM`。要 API キー(GUI の「キー」欄で入力・「キーを取得」で AI Studio を開く) |

VAD は Silero VAD(ONNX、モデルはバイナリに埋め込み)。ローカル ASR は VAD 発話終了時にバッファ全体を
デコードし、発話中は `partial_interval_ms`(既定800ms)ごとに部分表示を更新します。
kotoba は句読点を出さないため、チャンク分割は文節境界の近似と長さで切ります
(nemotron のみ句読点をネイティブ出力するため、読点で綺麗に切れます)。

### Nemotron 1120ms export(発話確定をさらに速く)

HF 配布の fp16 パッケージは 320ms チャンクのみ。chunk=1120ms は発話単位デコードがより速いので、
速さを優先する場合は自前で export します(CPU だけで可、ベースモデル ~2.5GB を DL):

```bash
git clone --depth 1 https://github.com/codavidgarcia/nemotron-3.5-asr-streaming-onnx
cd nemotron-3.5-asr-streaming-onnx
uv run --no-project --with "torch>=2.6" --with "transformers>=5.13.0" \
  --with "librosa>=0.10" --with "onnx>=1.17" --with "onnxruntime>=1.20" \
  --with "onnxmltools>=1.12" --with scipy --with soundfile --with "sentencepiece>=0.2" \
  python export/export_onnx.py --output-dir ../onnx-out --chunk-ms 1120 --validate
uv run --no-project --with "torch>=2.6" --with onnx --with onnxruntime \
  --with "onnxmltools>=1.12" python export/quantize.py --model-dir ../onnx-out --fp16
```

`encoder_1120ms_fp16.onnx(.data)` / `encoder_1120ms_first_fp16.onnx(.data)` と `decoder.onnx` /
`joiner.onnx` / `tokens.txt` / `nemotron_onnx_config.json` を `data/nemotron-onnx/`(gitignore 済み)へ
平置きし、`data/backend.json` に:

```json
{"asr": {"engine": "nemotron", "nemotron_model_dir": "data/nemotron-onnx", "nemotron_chunk_ms": 1120}}
```

(この export 作業だけは PyTorch が要ります。アプリの実行には不要です。)
参照実装のライセンスは Apache-2.0(`crates/nemotron/LICENSE`)、重みは OpenMDW-1.1(NVIDIA 所有・商用可)。

## テスト

```bat
cargo test -p sttts-engine      :: チャンク分割・ASR ワーカー・投機的 TTS・キャンセル・計測フィールド・設定・セッション(mic-first / クールダウン)・往復
cargo test --workspace --release  :: 全クレート。PyTorch 等の参照データが無いパリティテストは skip します
```

数値一致(パリティ)テストは PyTorch の参照出力を使います。参照データは開発用の Python 環境で作ります
(アプリの実行には不要):

```bat
cd tools\reference
uv run python dump_irodori_ref.py    :: → target\irodori-ref
uv run python dump_whisper_ref.py    :: → target\whisper-ref
```

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| 起動時に GPU を初期化できない | Vulkan 対応の GPU とドライバを確認。Intel Arc は最新ドライバへ。エンジンは GPU 専用で、CPU への自動フォールバックはありません |
| 合成の途中で「音声合成デバイスが停止しました」 | GPU のデバイス喪失です。アプリを再起動してください(同一プロセスでは復帰できません) |
| 文中の短い間で発話が切れる | `asr.vad_min_silence_ms` を 350〜400 に戻す |
| スピーカーで自分の合成音声を拾ってループする | ヘッドホンを使う(マイクは再生中も開いています) |
| 最初の発話まで時間がかかる | 初回はモデルのダウンロードと GPU カーネルの準備があります。`tts.warmup`(既定 ON)により起動直後からバックグラウンドでロードが始まります |
| マイクを開けない | 他アプリの排他占有を解除。入力デバイスは詳細設定で選べます |
| ASR の部分表示・確定が遅い | クラウド(Gemini)を選ぶのが最速。ローカルなら `asr.engine: "nemotron"`、または `asr.partial_interval_ms: 0` で partial を止めて確定を優先 |
| 発話が止まって進まない | ログ(GUI 下段)を確認。モデル初回 DL 中は待つ必要があります |

## ライセンス・クレジット

- [Irodori-TTS](https://github.com/Aratako/Irodori-TTS)(コード: MIT)
  - [v4.1-Small-MF](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF): MIT
  - コーデック [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim): MIT
  - 透かし [SilentCipher](https://huggingface.co/sony/silentcipher)
  - 上記 Irodori-TTS の各モデルカードには、ライセンスに加えて倫理的な利用制限があります(本人の同意なく声優・著名人など実在人物の声を複製・なりすましに使わない、誤情報やディープフェイク目的に使わない 等)
- [kotoba-whisper-v2.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0)(Apache-2.0)
- Nemotron 3.5 ASR streaming(コード Apache-2.0 / 重み OpenMDW-1.1)
- [silero-vad](https://github.com/snakers4/silero-vad)(MIT。モデルを埋め込み、`crates/audio/assets/LICENSE`)
- [burn](https://burn.dev)(Apache-2.0 / MIT)/ onnxruntime(MIT)
- gpui-kit(Apache-2.0)/ Zed GPUI(Apache-2.0)
- 本リポジトリのコード: ライセンスは未確定(リポジトリオーナーが確定してください)
