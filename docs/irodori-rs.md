# Irodori-TTS 純 Rust 実装(crates/irodori)

PyTorch / Python に依存しない Irodori-TTS(v4.1 Small MF / RF、v4 Large、それぞれの int8 weight-only 版)の推論。**目標: 純 Rust で、PyTorch 実装と数値が一致し(高精度)、Intel Arc を含む GPU で速い(高速)。**

## 方針

- **バックエンド**: burn 0.22。`Tensor<D>`(次元数のみ const generic)と `Device`(実行時に選ぶ)を使う。
  - CPU: `Device::flex()`(純 Rust)。**参照・テスト専用**(運用での CPU 推論はしない: AGENTS.md)。
  - GPU: `gpu` feature → wgpu(Vulkan)。Intel Arc で PyTorch XPU / Level Zero を通らない。
- **モデルは自前の構造体**で持つ(burn の `Module` derive は使わない)。重みは `Weights::tensor::<D>(name, &device)` で名前から取り出す。
- **精度**: 各段階を PyTorch の CPU fp32 と突き合わせる。目標は最大絶対誤差 ≤ 1e-4 × 参照の最大絶対値(`testing::assert_close(.., 1e-4)`)。ModernBERT や DiT のように深い段は 1e-3 まで許容してよいが、理由を残す。
- **速度は後段**: まず一致、次に GPU、最後に最適化。最初から最適化しない(実際、最適化の中身は下の「GPU で分かったこと」のとおり、演算の速さよりメモリと形状の扱いだった)。
- **PyTorch 実装が正**。原典は `tools/reference/.venv/Lib/site-packages/irodori_tts/`(`uv sync` で作る参照用環境)(`model.py` / `inference_runtime.py` / `meanflow.py` / `rf.py` / `codec.py` / `duration.py` / `text_normalization.py` / `attention.py`)。挙動に迷ったら原典を読み、参照出力で確かめる。
- 対象は MeanFlow(`flow_parameterization == "meanflow"`、4 ステップ)と RF(`rf_velocity`、40 ステップ + CFG)。両者の重みの違いは MeanFlow の区間長埋め込み(`delta_cond_module`)の有無だけで、DiT・条件エンコーダ・長さ予測は共通。
  RF のサンプラ(`sampler::sample_euler_rf_cfg_padded`)は原典 `sample_euler_rf_cfg` の全オプション(CFG の 3 方式、CFG をかける時刻範囲、truncation、score rescale、話者 K/V の強調、話者なし側の mask / noise、linear / sway)を持つ。
- テキストのバックボーンは、v4.1 Small が ModernBERT-ja(`modernbert`)、v4 Large が T5Gemma 2 のテキストエンコーダ(`t5gemma`)。
  `text_encoder_config_json` の `model_type` で選ぶ。v4 Large は話者の潜在を 4 フレームずつまとめ(`speaker_patch_size`)、参照音声は最長 120 秒。
- torchao で量子化したチェックポイント(`*-Quantized/int8-weight-only`)は、int8 の重み(`_weight_qdata`)と行ごとの scale を
  `weights` が読み、全結合(`nn::Linear`)が burn の量子化テンソル(Q8S、u32 詰め)として GPU に int8 のまま置く。掛け算の直前に
  f32 へ戻す(`qdata * scale`。torchao と同じ値)ので、計算は f32 のまま。原典の量子化版は bf16 で計算するが、ここでは合わせない。
  量子化されるのは DiT・話者エンコーダ・テキストのバックボーンの注意と MLP。正規化・射影・長さ予測は bf16 の重みを f32 にして使う。
  int8-dynamic(活性も量子化)・float8・int4 は未対応(読み込み時にエラー)。

## 状態

**実装済み・検証済み**: テキスト → 音声の全工程(正規化、トークナイザ、ModernBERT、条件エンコーダ、長さ予測、DiT と話者エンコーダ、MeanFlow / RF + CFG のサンプラ、DACVAE のデコード/エンコード、SilentCipher 透かし)。
各段階とエンドツーエンドが、PyTorch(CPU, fp32)の参照と **CPU(flex)でも GPU(wgpu / Vulkan)でも一致**する(`cargo test -p irodori --release`、GPU は `--features gpu` と `IRODORI_DEVICE=gpu`)。

| 段階 | 最大絶対誤差(参照の最大値に対する比) |
|---|---|
| トークン ID・マスク | 完全一致 |
| ModernBERT / 条件エンコーダ | 2e-5(5e-7) |
| 長さ予測 | 5e-7 |
| DiT 1 ステップ | 3e-5(6e-6) |
| 4 ステップ後の潜在(CPU / GPU) | 2e-3 / 4e-3(5e-4 / 1e-3)。f32 の足し込み順の違いが 4 ステップで積み上がる |
| DACVAE デコード / エンコード | 1e-5 |
| 透かし(差分の相関 / SDR) | 0.999999 / 0.00 dB |
| 最終音声(ケース A〜D) | 8e-5 〜 4e-3(相対 1e-4 〜 4e-3。最大は GPU のケース A) |

RF(v4.1 Small、CPU)は `tests/rf.rs` で、ケース RA〜RE(text CFG のみ / caption / 参照音声 / alternating・sway・話者 K/V 強調・rescale・truncation / joint)を確かめた。
DiT 1 回(CFG のバッチ 1〜3)は相対 2e-6、ステップ数 6〜8 の最終潜在は相対 4e-6 〜 6e-4、最終音声は相対 1e-5 〜 6e-4。RF の GPU での一致と速度は未確認。

int8(v4.1 Small、`tests/int8.rs`)は、int8 の重みを fp32 に戻して PyTorch(CPU, fp32)で動かした参照と比べる(ケース QA / QC)。
テキスト条件は相対 1e-6、最終潜在は相対 3e-5 〜 3e-4、最終音声は相対 1e-4 〜 7e-4。v4 Large は `tests/large.rs`(LA / LC と、int8 の QA / QC)。

乱数だけは PyTorch と同じ列にならない(`rand` の標準正規。seed を渡せば再現はする)。参照との比較では初期ノイズを注入している。

### 速度(Intel Arc B570 / Vulkan、f32、同一プロセスで 36 発話を連続合成)

| 発話の長さ | 合成時間 | RTF |
|---|---|---|
| 約 1 秒 | 0.5 秒 | 0.5 |
| 3〜5 秒 | 0.9〜1.4 秒 | 0.3 |
| 9 秒 | 2.3 秒 | 0.25 |
| 14 秒 | 3.7 秒(メモリの競合で 6〜8 秒になることがある) | 0.26(〜0.55) |

比較: 同じマシンの PyTorch XPU(`uv run` のバックエンド)は 3 秒の発話に約 4 秒(RTF 1.4)かかり、長さの違う発話を数回続けると `UR_RESULT_ERROR_OUT_OF_RESOURCES` → `DEVICE_LOST` で落ちた(PyTorch 2.10 / 2.11 / 2.14.1 のどれでも再現)。この実装は同じ条件で落ちない。
起動後の最初の数発話は、GPU のカーネルをコンパイルするので 1〜数秒余計にかかる。`Tts::warmup()`(約 10〜30 秒)を先に呼べば、最初の発話から定常の速度になる。

## 使い方

```
# 参照出力(PyTorch, CPU のみ。約 1 分)
cd tools/reference && uv run python dump_irodori_ref.py
cd tools/reference && uv run python dump_irodori_ref.py --rf     # RF(v4.1 Small)→ target/irodori-ref-rf
cd tools/reference && uv run python dump_irodori_ref.py --int8   # v4.1 Small の int8 → target/irodori-ref-int8
cd tools/reference && uv run python dump_irodori_ref.py --large  # v4 Large → target/irodori-ref-large(--large-int8 も)

# テスト(CPU)/ GPU
cargo test -p irodori --release
IRODORI_DEVICE=gpu cargo test -p irodori --release --features gpu -- --test-threads=1

# 合成(CLI)。段階ごとの時間は IRODORI_TRACE=1 で出る
cargo run -p irodori --release --features gpu --example tts -- --warmup     --text "こんにちは、よろしくお願いします。" [--caption "落ち着いた声で"] [--ref ref.wav] --out out.wav
```

```rust
let paths = irodori::pipeline::TtsPaths::from_hf_cache()?;      // HF キャッシュのモデル
let tts = irodori::pipeline::Tts::load(&paths, &irodori::gpu_device())?;
tts.warmup()?;                                                    // 任意: カーネルを先にコンパイル
let out = tts.synthesize(&SamplingRequest { text: "…".into(), no_ref: true, ..Default::default() })?;
// out.audio: Vec<f32>(モノ)、out.sample_rate: 48000
```

## GPU で分かったこと(Arc B570、ドライバ 32.0.101.8860、burn 0.22 / wgpu)

実装の大半は「速い演算を書く」ことではなく、burn-wgpu の落とし穴を避けることだった。

1. **`burn/vulkan`(SPIR-V コンパイラ)は使わない**。reduce 系(`sum` / `mean` / `max` / `softmax` / `attention`)が誤った値を返す(例: ランダムな `[4, 9, 33]` のテンソルの `sum_dim(1)` が CPU(flex)と合わない)。WGSL 経路(`burn/wgpu` だけ)は全演算が CPU と一致する。
2. **メモリ管理は `ExclusivePages`**(`device::gpu_device`)。既定の適応型は、長い発話を 1 回処理したあと以降のすべての演算が約 10 倍遅くなる状態に入った(メモリ使用量は変わらない)。
3. **`burn/fusion` と `burn/autotune` は使わない**。有効にすると初回のカーネル探索・コンパイルが極端に長く(ModernBERT だけで 40 秒以上)、定常速度は変わらない。
4. **大きな単一の確保を避ける**。トークン埋め込み表(300MB)は CPU に置いて必要な行だけ送る(GPU に置くとメモリプールが肥大して全体が遅くなった)。畳み込みの im2col と透かしの活性は固定長の列塊(`CHUNK`)に分ける。GPU メモリは他のアプリと取り合いになり、一時テンソルが数百 MB になると演算が数倍〜10 倍遅くなる。
5. **畳み込みは im2col + 行列積**。burn の `conv1d` / `conv2d` は GPU でカーネル探索が極端に遅いことがある。行列積は速い(2〜3 TFLOPS)。3x3 の `conv2d` は、縁取り付きの平らな配列上の一次元のずらし(`Geometry`)にして、4D のまま切り出して詰め直すコピーを避ける。
6. **形状を揃える**。GPU のカーネルは形状の整列クラスごとに作られ、初回は 1 本数百 ms。テキストのトークン数・潜在の長さは数段階(`bucket`)に零詰めして(無効位置は注意のキーから除外するので結果は同じ)、コーデックは固定長(窓 25 フレーム + 文脈 8 フレーム)の窓で処理する。
7. **無効トークンを計算しない**。テキストは 256、キャプションは 512 に固定でパディングされるが、無効位置は注意から完全に除かれるので、有効な先頭部分だけで計算しても同じ結果になる(空キャプションは省略)。
8. **f16 は使えない**(試して外した)。行列積を f16 にすると潜在が最大 11% ずれる(重みだけを f16 にしても 30% ずれた)うえ、速度も変わらなかった。
9. **PyTorch のバージョンは関係ない**。同じ失敗が 2.10 / 2.11 / 2.14.1 のすべてで出た。

## 参照出力(PyTorch, CPU, fp32)

```
cd tools/reference && uv run python dump_irodori_ref.py     # → target/irodori-ref/{refs.safetensors,meta.json,ref.wav}
```

約 1 分。`target/` は git 管理外。`irodori::testing::refs()` が読む(無ければ `None`。テストは無い時にスキップせず、`eprintln!` して return する)。モデルは `testing::checkpoint_dir()`(HF キャッシュ)。

キーは `<case>.<stage>.<n>[.<name>]`。ケース: A=無参照・無キャプション、B=キャプションあり、C=参照音声あり(`ref.wav`)、D=長文。`n` は同じ段が呼ばれた順(0 始まり)。

| stage | 内容 |
|---|---|
| `tok_text.out0/out1` | テキストのトークン ID(I64 [1,256])とマスク(BOOL)。入力文は meta.json の `tokenizer_calls` |
| `tok_caption.out0/out1` | キャプション(空文字でも 512 長で出る) |
| `backbone.in.ids/mask.n`, `backbone.out.n` | ModernBERT。n=0:text, 1:caption, 2:text, 3:caption(duration 用と sampler 用で 2 回ずつ) |
| `encode_conditions.out0..5.n` | text_state, text_mask, speaker_state, speaker_mask, caption_state, caption_mask |
| `duration.in.*`, `duration.out.0` | 長さ予測の入力と出力(log フレーム数) |
| `dit.in.{x_t,t,delta_t,...}.n`, `dit.out.n` | MeanFlow 4 ステップ。x_t.0 が初期ノイズ(torch の RNG は再現できないので注入する) |
| `codec_encode.in/out`, `codec_decode.in/out` | DACVAE。ケース C はエンコード(参照音声)も |
| `watermark.in0/out0`, `final_audio` | SilentCipher 前後と最終音声 |

## モジュールと担当(パイプライン順)

`crates/irodori/src/` — 各ファイルは担当が実装する。API は下の署名を基準にし、必要なら追加してよい(既存の署名を変えるときは doc に書く)。

| module | 内容 | 検証する参照 |
|---|---|---|
| `weights`, `config`, `testing` | 実装済み(safetensors mmap、`config_json`) | — |
| `text` | `normalize_text`(`text_normalization.py`)、`build_duration_features`(`duration.py`) | `tokenizer_calls` の入力 → `tok_text`、`duration.in.duration_features` |
| `tokenizer` | `tokenizer/tokenizer.json`(Unigram)で `batch_encode(texts, max_length, add_bos)` → (ids, mask)。`tokenizers` クレート(`unstable_wasm` = 純 Rust 正規表現) | `tok_text`, `tok_caption` |
| `modernbert` | ModernBERT-ja-310m(25 層、全注意と窓付き注意、RoPE、GeGLU)。重みは `pretrained_text_backbone.backbone.*` | `backbone.out.*` |
| `condition` | `PretrainedConditionProjector`(residual_mlp)と `encode_text/caption` | `encode_conditions.out0,4` |
| `dit` | 話者エンコーダ(`ReferenceLatentEncoder`)、`encode_conditions` の話者側、`forward_with_encoded_conditions`、`build_context_kv_cache`、JointAttention / LowRankAdaLN / SwiGLU / RoPE | `encode_conditions.out2`, `dit.out.*`(入力は `dit.in.*` を注入) |
| `duration` | `DurationPredictor`(`token_sum_dual_adarn_zero_no_aux`) | `duration.out.0` |
| `sampler` | `sample_euler_meanflow`(4 ステップ、`linspace(1,0)`)、`sample_euler_rf_cfg_padded`(RF + CFG)、`unpatchify`、`find_flattening_point`(末尾トリム) | `dit.*` 全 4 ステップと最終潜在 |
| `pth` | PyTorch `.pth` / `.ckpt`(zip + pickle)の読み取り(依存なしの最小実装。透かしの ckpt も読む) | DACVAE `weights.pth` |
| `codec` | DACVAE のデコーダ(と参照音声用エンコーダ、ラウドネス正規化)。透かし枝は `forward_no_conv` のみ | `codec_decode`, `codec_encode` |
| `watermark` | SilentCipher 44.1k(`sony/silentcipher` の ckpt)。`encode_batch` | `watermark.out0` |
| `pipeline` | `Tts::load(dir) → synthesize(SamplingRequest) → audio`。上を結ぶ。乱数は自前の RNG(seed 指定可、PyTorch とは別系列) | `final_audio`(注入ノイズで一致) |

## 規約

- 1 モジュール 1 ファイル + `crates/irodori/tests/<module>.rs`。依存の追加は最小限(既に Cargo.toml にあるものを使う)。
- unsafe を増やさない。`anyhow::Result` を返す。重い処理をテスト内で重複させない(重みのロードは 1 テストに 1 回)。
- ハードウェア方針(AGENTS.md): **マイク・実 GPU に触れる検証を書かない**(GPU の検証は私=メインが行う)。テストは flex(CPU)のみ。
- コミットは自分のブランチ/ワークツリーに。メッセージ末尾に `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`。

## 設計上の判断とフォローアップ(レビュー指摘の整理)

- `flex`(CPU)バックエンドは PyTorch 参照とのパリティテスト用。本番の推論は GPU(`gpu` feature)で行い、アプリ側に CPU へのフォールバックは作らない。
- 参照データ・チェックポイントが無い環境ではパリティテストは `eprintln!` して戻る(CI を落とさない)。数値検証が必要なときは `IRODORI_REF_DIR` を設定して実行する。
- 参照音声は wav / flac のみ。mp3 などは未対応(変換して渡す)。
- フォローアップ: `RmsNorm` / 読み戻しヘルパの共通化(`Linear` は `nn` にまとめた。長さ予測と条件の射影は別実装のまま)、`Conv` の重み二重保持の解消(VRAM 削減)、`gpu_device()` の `Result` 化。
