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
DiT 1 回(CFG のバッチ 1〜3)は相対 2e-6、ステップ数 6〜8 の最終潜在は相対 4e-6 〜 6e-4、最終音声は相対 1e-5 〜 6e-4。GPU(Arc B580 / wgpu)でも同じテストが通る(最終潜在は相対 2e-3 以内)。

int8(v4.1 Small、`tests/int8.rs`)は、int8 の重みを fp32 に戻して PyTorch(CPU, fp32)で動かした参照と比べる(ケース QA / QC)。
テキスト条件は相対 1e-6、最終潜在は相対 3e-5 〜 3e-4、最終音声は相対 1e-4 〜 7e-4。v4 Large は `tests/large.rs`(LA / LC と、int8 の QA / QC)。T5Gemma 2 単体は 256 / 512 トークンで相対 4e-7 / 8e-7、
最終潜在は相対 1e-5 〜 6e-4(int8 は 7e-5 〜 6e-4)、最終音声は相対 4e-5 〜 1.6e-3。
GPU(Arc B580 / wgpu、12GB)では int8 の重みを量子化テンソルのまま置いて v4.1 Small int8 と v4 Large int8 の一致を確かめた
(最終潜在は相対 1.3e-3 以内)。短い文(約 2 秒)の 6 ステップのサンプリングは v4.1 Small int8 で 1.2〜2.0 秒、v4 Large int8 で 2.7〜3.6 秒。
v4 Large(fp32、13GB)は 12GB の GPU に載らないので GPU では未確認。

乱数だけは PyTorch と同じ列にならない(`rand` の標準正規。seed を渡せば再現はする)。参照との比較では初期ノイズを注入している。

### 速度(Intel Arc B580 / Vulkan、ドライバ 32.0.101.8626、f32、同一プロセスで連続合成、透かしあり)

v4.1 Small MF(既定、4 ステップ):

| 発話の長さ | 合成時間 | RTF |
|---|---|---|
| 約 2 秒 | 0.23 秒 | 0.13 |
| 4〜5 秒 | 0.35〜0.45 秒 | 0.09〜0.10 |
| 9 秒 | 0.69 秒 | 0.08 |
| 19 秒 | 1.34 秒 | 0.07 |

生成長を 8 秒に固定し、透かしなし・seed 42 で測った値(自前カーネル導入の前後):

| | 導入前 | 現在 |
|---|---|---|
| MF 4 ステップ | 1.55 秒 | 0.35 秒 |
| RF 40 ステップ + text CFG | 6.2 秒 | 1.83 秒 |

[irodori-engine-comparison](https://github.com/idunafu/irodori-engine-comparison) と同じ条件(caption と参照音声あり、CFG 3 / 3 / 5、
seed 42、透かしなし、参照の潜在はキャッシュ、2 回目以降の中央値。参照音声は比較リポジトリのものでなく同梱の narrator)で測った値:

| 発話の長さ | MF 導入前 | MF 現在 | RF 導入前 | RF 現在 |
|---|---|---|---|---|
| 7.4 秒 | 1.54 秒 | 0.35 秒 | 8.8 秒 | 2.1 秒 |
| 13.2 秒 | 2.49 秒 | 0.58 秒 | 12.5 秒 | 4.0 秒 |
| 22.7 秒 | 5.64 秒 | 0.91 秒 | 60.4 秒 | 8.6 秒 |
| 30 秒 | 6.55 秒 | 1.09 秒 | 61.7 秒 | 9.0 秒 |

参考(導入前、Arc B570): MF で約 1 秒の発話が 0.5 秒、3〜5 秒が 0.9〜1.4 秒、14 秒が 3.7 秒だった。

比較: 同じマシンの PyTorch XPU(`uv run` のバックエンド)は 3 秒の発話に約 4 秒(RTF 1.4)かかり、長さの違う発話を数回続けると `UR_RESULT_ERROR_OUT_OF_RESOURCES` → `DEVICE_LOST` で落ちた(PyTorch 2.10 / 2.11 / 2.14.1 のどれでも再現)。この実装は同じ条件で落ちない。
起動後の最初の数発話は、GPU のカーネルをコンパイルし、行列積のタイルを形状ごとに選ぶので 1〜数秒余計にかかる。`Tts::warmup()`(B580 で MF 約 4 秒、RF 約 8 秒)を先に呼べば、最初の発話から定常の速度になる。

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

実装の大半は「速い演算を書く」ことではなく、burn-wgpu の落とし穴を避けることだった。ただし行列積だけは burn の既定が遅すぎたので、
自前の WGSL カーネルにした(下の 11)。

1. **`burn/vulkan`(SPIR-V コンパイラ)は使わない**。reduce 系(`sum` / `mean` / `max` / `softmax` / `attention`)が誤った値を返す(例: ランダムな `[4, 9, 33]` のテンソルの `sum_dim(1)` が CPU(flex)と合わない)。WGSL 経路(`burn/wgpu` だけ)は全演算が CPU と一致する。
2. **メモリ管理は `ExclusivePages`**(`device::gpu_device`)。既定の適応型は、長い発話を 1 回処理したあと以降のすべての演算が約 10 倍遅くなる状態に入った(メモリ使用量は変わらない)。
3. **`burn/fusion` と `burn/autotune` は使わない**。有効にすると初回のカーネル探索・コンパイルが極端に長く(ModernBERT だけで 40 秒以上)、定常速度もあまり変わらない
   (B580 で autotune だけを有効にすると MF 8 秒の合成が 1.55 → 1.15 秒、初回は 20〜40 秒。行列積は autotune でも 3〜3.7 TFLOPS 止まり)。
4. **大きな単一の確保を避ける**。トークン埋め込み表(300MB)は CPU に置いて必要な行だけ送る(GPU に置くとメモリプールが肥大して全体が遅くなった)。畳み込みの im2col は固定長の列塊(`CHUNK`)に分ける。GPU メモリは他のアプリと取り合いになり、一時テンソルが数百 MB になると演算が数倍〜10 倍遅くなる。
   VRAM を使い切ると溢れた分がシステムメモリへ追い出され、さらに遅くなる(B580 の 12 GB で、30 秒の音声の透かしが以前は 1 秒弱 → 10〜12 秒。下の 5)。
5. **畳み込みは行列積として計算する**。burn の `conv1d` / `conv2d` は GPU でカーネル探索が極端に遅いことがある。コーデックの stride 1 の畳み込みは、
   im2col を作らずに入力から直接列を読む行列積のカーネル(`gemm::conv1d`)で、前段の Snake・バイアス・残差の足し込みも同じカーネルで行う。
   転置畳み込み(kernel = 2 * stride)は行列積 + 重なり加算(`gemm::conv_transpose1d`)。GPU でない・形が合わないときは従来の im2col + 行列積。
   透かしの 3x3 の `conv2d` は、縁取り付きの平らな配列上の一次元のずらし(`Geometry`)にして、4D のまま切り出して詰め直すコピーを避ける。
   GPU ではこれも同じ行列積のカーネルで、近傍を「行の幅 × 段 + 列」のずらしとして入力から直接読み、ゲート(`conv * sigmoid(gate)`)と
   無効な位置の 0 埋めを書き出しで行う(`gemm::gated_conv_bn`。BatchNorm の統計と適用も自前のカーネル)。時間方向に切らずに全体を 1 回で計算するので、
   同時に持つのは層の入力・出力程度になる(30 秒の音声で一時メモリが 5.5 GB → 2.1 GB、透かしだけの時間が 0.96 → 0.47 秒)。
   CPU は列塊ごとに近傍を並べた行列を作って行列積にする(同じ式。パリティテストの参照)。
6. **形状を揃える**。GPU のカーネルは形状の整列クラスごとに作られ、初回は 1 本数百 ms。テキストのトークン数・潜在の長さは数段階(`bucket`)に零詰めして(無効位置は注意のキーから除外するので結果は同じ)、コーデックは固定長(窓 25 フレーム + 文脈 8 フレーム)の窓で処理する。
7. **無効トークンを計算しない**。テキストは 256、キャプションは 512 に固定でパディングされるが、無効位置は注意から完全に除かれるので、有効な先頭部分だけで計算しても同じ結果になる(空キャプションは省略)。
8. **CFG の条件あり・なしは 1 バッチにまとめる**(independent と joint。原典と同じ)。以前は burn の行列積でバッチ 2〜3 の forward が
   バッチ 1 の 5〜7 倍かかったのでバッチ 1 を本数ぶん回していたが、自前の行列積(11)では行が増えるほど GPU が埋まって効率が上がる
   (B580 で潜在 256 フレームのバッチ 1 が約 31ms、バッチ 2 が約 51ms)。各サンプルは独立に計算されるので結果は変わらない。
   alternating はステップごとに外す条件が替わるので、2 回に分けて回す。時刻だけで決まる adaLN の変調は全ステップ分をまとめて作り、CFG の各パスで共有する。
   int8 の重みは掛け算の直前に f32 へ戻すので、速度は f32 版とほぼ同じ(減るのは GPU メモリ)。`IRODORI_TRACE=1` でステップごとの時間が出る。
9. **f16 は使えない**(試して外した)。行列積を f16 にすると潜在が最大 11% ずれる(重みだけを f16 にしても 30% ずれた)うえ、速度も変わらなかった。
10. **PyTorch のバージョンは関係ない**。同じ失敗が 2.10 / 2.11 / 2.14.1 のすべてで出た。
11. **行列積は自前の WGSL カーネル**(`gemm`、`gemm.wgsl`)。burn の既定の行列積(autotune なし)は Arc で FP32 性能の 1 割も出ない形状が多く
    (B580 で 2048 角が 0.8 TFLOPS、コーデックの初段 `[768x5376]x[5376x492]` が 0.5 TFLOPS)、DiT もコーデックもほぼ全時間が行列積だった。
    共有メモリでタイル化し、累積をレジスタに置く(行・列のループを Rust 側で展開してから埋め込む。展開しないと Intel のコンパイラが
    累積を配列のままメモリに置いて、burn より遅くなった)と、2048 角が 9 TFLOPS 前後、DiT の形状で 4〜7 TFLOPS。
    1 スレッドが受け持つ大きさ(タイル)は形状ごとに初回だけ実測して選ぶ。どのタイルでも各出力は K 方向に同じ順で足すので結果は変わらない。
    出力のタイルが少なく GPU が埋まらない形状(DiT の `N = 1280` など)は K を分けて部分和を足す(分割数は形状だけで決まるので、同じ形状なら結果は毎回同じ)。
    burn の行列積との差は CPU 比で同程度(相対 2〜5e-6)。DiT の q / k の正規化 + RoPE と adaLN の正規化 + 変調も自前のカーネルにまとめた。
    注意の QK^T と PV もヘッドごとのバッチ付き行列積(`gemm::bmm`)で計算する(burn の注意は内部で burn の行列積を使い、
    潜在 768 フレームの RF 40 ステップで 15 秒 → 9 秒)。キー長が 4 の倍数になるよう、文脈の K / V とマスクを詰め物で伸ばす。
    カーネルは burn の `extension`(`#[backend_extension]`)と burn-cubecl の `template`(WGSL をそのまま起動)で呼ぶ。CPU(flex)は burn のまま。

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
| `gemm` | GPU の自前 WGSL カーネル: 行列積(形状ごとのタイル選択と K 分割)、コーデックの畳み込み・転置畳み込み、DiT の q / k の正規化 + RoPE、adaLN の正規化 + 変調。CPU は burn のまま | 各段の参照(GPU で `IRODORI_DEVICE=gpu`) |
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
