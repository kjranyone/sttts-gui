# Irodori-TTS 純 Rust 実装(crates/irodori)

PyTorch / Python に依存しない Irodori-TTS(v4.1 Small MF)の推論。**目標: 純 Rust で、PyTorch 実装と数値が一致し(高精度)、Intel Arc を含む GPU で速い(高速)。**

## 方針

- **バックエンド**: burn 0.22。`Tensor<D>`(次元数のみ const generic)と `Device`(実行時に選ぶ)を使う。
  - CPU: `Device::flex()`(純 Rust)。**参照・テスト専用**(運用での CPU 推論はしない: AGENTS.md)。
  - GPU: `gpu` feature → wgpu(Vulkan)。Intel Arc で PyTorch XPU / Level Zero を通らない。
- **モデルは自前の構造体**で持つ(burn の `Module` derive は使わない)。重みは `Weights::tensor::<D>(name, &device)` で名前から取り出す。
- **精度**: 各段階を PyTorch の CPU fp32 と突き合わせる。目標は最大絶対誤差 ≤ 1e-4 × 参照の最大絶対値(`testing::assert_close(.., 1e-4)`)。ModernBERT や DiT のように深い段は 1e-3 まで許容してよいが、理由を残す。
- **速度は後段**: まず一致、次に GPU、最後に最適化(固定長パディングのマスク済みトークンの除去、f16 など)。最初から最適化しない。
- **PyTorch 実装が正**。原典は `backend/.venv/Lib/site-packages/irodori_tts/`(`model.py` / `inference_runtime.py` / `meanflow.py` / `codec.py` / `duration.py` / `text_normalization.py` / `attention.py`)。挙動に迷ったら原典を読み、参照出力で確かめる。
- MeanFlow(`flow_parameterization == "meanflow"`)のみ対象。RF(CFG あり)は対象外。

## 参照出力(PyTorch, CPU, fp32)

```
cd backend && uv run --no-sync python scripts/dump_irodori_ref.py     # → target/irodori-ref/{refs.safetensors,meta.json,ref.wav}
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
| `sampler` | `sample_euler_meanflow`(4 ステップ、`linspace(1,0)`)、`unpatchify`、`find_flattening_point`(末尾トリム) | `dit.*` 全 4 ステップと最終潜在 |
| `pth` | PyTorch `.pth`(zip + pickle)の読み取り(依存なしの最小実装) | DACVAE `weights.pth` |
| `codec` | DACVAE のデコーダ(と参照音声用エンコーダ、ラウドネス正規化)。透かし枝は `forward_no_conv` のみ | `codec_decode`, `codec_encode` |
| `watermark` | SilentCipher 44.1k(`sony/silentcipher` の ckpt)。`encode_batch` | `watermark.out0` |
| `pipeline` | `Tts::load(dir) → synthesize(SamplingRequest) → audio`。上を結ぶ。乱数は自前の RNG(seed 指定可、PyTorch とは別系列) | `final_audio`(注入ノイズで一致) |

## 規約

- 1 モジュール 1 ファイル + `crates/irodori/tests/<module>.rs`。依存の追加は最小限(既に Cargo.toml にあるものを使う)。
- unsafe を増やさない。`anyhow::Result` を返す。重い処理をテスト内で重複させない(重みのロードは 1 テストに 1 回)。
- ハードウェア方針(AGENTS.md): **マイク・実 GPU に触れる検証を書かない**(GPU の検証は私=メインが行う)。テストは flex(CPU)のみ。
- コミットは自分のブランチ/ワークツリーに。メッセージ末尾に `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`。
