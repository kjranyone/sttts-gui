# Irodori TTS GUI — 実装計画(Windows 11 + Intel Arc B570 向け)

- 作成日: 2026-10-08
- ステータス: **計画確定・未実装**(本リポジトリには本計画書のみ存在する)
- ゴール: [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) を Intel Arc B570(XPU)でローカル実行し、テキスト入力→音声生成→再生→WAV保存ができる Windows デスクトップGUIアプリ(PySide6)を作る

---

## 1. 調査で確定した重要事実(2026-10-08 時点)

### 1-1. モデルの所在 — 「v4.1-Large」は存在しない
- ユーザーが指定した「Irodori TTS v4.1 Large」というモデルは **Hugging Face に存在しない**。v4.1 系は Small のみ。
- Aratako 名義の実在チェックポイント:

| モデル | 規模 | 備考 |
|---|---|---|
| `Aratako/Irodori-TTS-v4-Large` | 3.29B | **Large の最新。本計画の既定モデル**。safetensors 単一ファイル 約13.2GiB(FP32保存)、非 gated |
| `Aratako/Irodori-TTS-v4-Large-Quantized` | 同上の量子化 | torchao 5種(int8-weight-only / int8-dynamic / int4-weight-only / fp8×2)。サブフォルダ形式。**XPU は公式未検証** |
| `Aratako/Irodori-TTS-v4.1-Small` | 約766M | duration predictor 改良の最新Small |
| `Aratako/Irodori-TTS-v4.1-Small-Quantized` | 同上 | 同上 |

- したがってGUIは**モデル切替式**とし、既定を v4-Large、フォールバック/高速用に v4.1-Small と INT8 量子化版を選べるようにする。
- アーキテクチャ: RF-DiT(Rectified Flow Diffusion Transformer)24層/2048次元、テキストエンコーダは T5Gemma-2-1b 微調整品、48kHz 出力、日本語専用。ゼロショット音声クローン(参照音声最大120秒)/ Voice Design(キャプション指定)/ 絵文字による感情制御 に対応。
- ライセンス: **Gemma Terms of Use**(配布時に利用規約の案内が必要)。

### 1-2. Intel Arc(GPU)対応は公式で完結している
- 本家リポジトリが **PR #20 で XPU バックエンドを正式サポート済み**。`irodori_tts/inference_runtime.py` に xpu 分岐が組み込み済みで、**ソース改変は一切不要**。
- `pyproject.toml` の `xpu` extra(Linux/Windows マーカー明記):

```toml
xpu = [
    "torch>=2.10.0,<2.11.0; sys_platform == 'linux' or sys_platform == 'win32'",
    "torchao>=0.16.0,<0.17.0; sys_platform == 'linux' or sys_platform == 'win32'",
    "torchaudio>=2.10.0,<2.11.0; sys_platform == 'linux' or sys_platform == 'win32'",
    "torchcodec>=0.10.0,<0.11.0; sys_platform == 'linux' or sys_platform == 'win32'",
    "triton-xpu==3.6.0; sys_platform == 'linux' or sys_platform == 'win32'",
]
```

- 導入は **`uv sync --extra xpu` のみ**。IPEX(既にEOL, 2026-03終了)も oneAPI Base Toolkit も追加不要(pip wheel にランタイム同梱)。**torch-directml は不採用**(2024-09 更新停止、torch 2.4 固定のため新しめの op が壊れる)。
- PyTorch 側: Windows XPU 公式 wheel は PyTorch 2.7 から。B570(Battlemage)は 2.7 以降が公式サポート対象。ドライバ要件は **32.0.101.7028 以上**。

### 1-3. B570 の 10GB VRAM と精度戦略
- FP32: 重みだけで 13.2GB → **不可**。
- **bf16: 重み約6.6GB → 既定**。T5エンコーダ+DACVAEコーデック+活性化込みで 8〜9GB まで張り得るため、低VRAM設定(`decode_mode="sequential"`, `num_candidates=1`, 巨大な `seconds` 指定を避ける)をデフォルト値にする。
- **INT8 weight-only: 重み約3.7GB → 余裕**。torchao 0.16+ は XPU の int8/int4 に対応(PyTorch 2.10+TorchAO ブログは Windows 11 + Arc で検証済み)が、公式チェックポイント自体は「XPU未検証」。OOM時に自動提案するフォールバックとする。
- codec(DACVAE, 約410MB)は `codec_device` / `codec_precision` で分離可能 → メモリが厳しければ CPU へ追い出し。

### 1-4. プログラムからの呼び出し方(ライブラリ直呼びが正攻法)
サーバー版([Aratako/Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server))は extras が cpu/cu128/rocm のみで **XPU 非対応**のため不採用。GUI からは公式ライブラリを直接 import する。実装の雛形(`infer.py` の main 相当):

```python
from irodori_tts.inference_runtime import (
    InferenceRuntime, RuntimeKey, SamplingRequest,
    default_runtime_device, download_hf_checkpoint, save_wav,
)

checkpoint_path = download_hf_checkpoint("Aratako/Irodori-TTS-v4-Large")
# 量子化版は repo/subfolder 形式: download_hf_checkpoint("Aratako/Irodori-TTS-v4-Large-Quantized/int8-weight-only")

runtime = InferenceRuntime.from_key(RuntimeKey(
    checkpoint=checkpoint_path,
    model_device="xpu",            # "cpu" / "cuda" / "mps" / "xpu"
    codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
    model_precision="bf16",        # "fp32" | "bf16"(bf16 は CUDA/XPU のみ)
    codec_device="xpu",
    codec_precision="fp32",
    codec_deterministic_encode=True,
    codec_deterministic_decode=True,
    compile_model=False,
    compile_dynamic=False,
))

result = runtime.synthesize(SamplingRequest(
    text="こんにちは、私はAIです。",
    caption="落ち着いた、近い距離感の女性話者",  # Voice Design 用(None 可)
    ref_wav="reference.wav",        # 音声クローン用(なしの場合 no_ref=True または caption のみ)
    no_ref=False,
    num_candidates=1,
    decode_mode="sequential",       # 低VRAM 既定
    seconds=None,                   # None なら内蔵 duration predictor が自動推定
    duration_scale=1.0,
    num_steps=None,                 # None で RF:40 / MeanFlow:4
    cfg_scale_text=3.0, cfg_scale_caption=3.0, cfg_scale_speaker=5.0,
    cfg_guidance_mode="independent",
    seed=1234,                      # None なら毎回乱数。result.used_seed で実シード取得可
    t_schedule_mode="linear", sway_coeff=-1.0,
    lora_adapter=None,
))

out_path = save_wav("output.wav", result.audio, result.sample_rate)  # 48kHz WAV
```

- 便利 API: `list_available_runtime_devices()`(xpu 検出)/ `list_available_runtime_precisions(device)`(cuda/xpu → ["fp32","bf16"])/ `default_runtime_device()`(Intel GPU 専用機なら自動で "xpu")。
- デバイス選択に cuda ハードコードは一切ない。GUI はデバイス自動検出 → xpu 優先、cpu フォールバック。

### 1-5. 実装マシンの状態(2026-10-08 診断済み)
- Windows 11(build 26200)/ CPU: Core Ultra 7 265KF / RAM 47.7GB / C: 空き約830GB
- GPU: **Intel Arc B570 / ドライバ 32.0.101.8860**(要件 32.0.101.7028 以上を満たす。最新WHQLは 32.0.101.9034 — XPU が不安定なら更新)
- oneAPI Toolkit 2026 インストール済み(今回の構成では不要だが害はない)
- Python 3.12.10 + uv 0.9.27(本家は `.python-version` 3.10 だが uv が管理 Python を自動取得。GUI プロジェクト側は 3.12 で統一予定、irodori-tts は `requires-python >=3.10`)
- **注意**: グローバル pip に torch 2.11.0+cpu が入っている → 専用 venv(uv)で隔離し、絶対にグローバルへ入れない
- git identity 設定済み(git 依存パッケージ dacvae / silentcipher の clone に git が必要)

### 1-6. 既知のリスクと対策
| リスク | 対策 |
|---|---|
| Windows の torchcodec DLL 問題(Issue #40、学習前処理で発覚) | 推論パスでの import 要否を環境構築時に確認。`import irodori_tts` が壊れる場合は依存整理で対処 |
| 出力 WAV が Windows 標準メディアプレーヤーで再生できない場合がある(Issue #33) | GUI 内蔵プレーヤ(sounddevice)で再生。保存は `save_wav`(torchaudio→soundfile フォールバック)を使用 |
| XPU の OOM(共有メモリを合算した容量を報告するが実際は割れない / -997 エラーの実体が OOM の場合あり) | OOM 検出 → 「INT8 切替」「codec を CPU へ」「sequential デコード」をGUIから提案 |
| 量子化チェックポイントの XPU 非検証 | 既定は非量子 bf16。INT8 は実機検証フェーズで動作確認してから既定化の可否を判断 |
| B570 での生成速度 | 目安: v4-Large bf16 で**数十秒/文**(40 steps)。Arc A750 + 500M モデルが約8秒の実績あり。`t_schedule_mode="sway"` で高速化可。速度実測は検証フェーズで行う |

---

## 2. アプリ設計

### 2-1. プロジェクト構成(本リポジトリに作る)
```
irodori-tts-gui/
  pyproject.toml        # 依存: irodori-tts[xpu] @ git+https://github.com/Aratako/Irodori-TTS
                        #       + PySide6 + sounddevice + soundfile + numpy
                        # [tool.uv] の pytorch-xpu index 設定は本家 pyproject をミラー
  run.bat               # uv run python -m app.main
  README.md             # セットアップ/使い方/トラブルシュ/ライセンス案内
  PLAN.md               # 本書
  app/
    main.py             # エントリポイント
    core/
      settings.py       # 設定の保存・読込(JSON → data/config.json)
      models.py         # モデルカタログ(下記4種+今後追加)
      runtime_manager.py# InferenceRuntime ロード/合成/save_wav ラッパ、HF DL 進捗コールバック、OOM フォールバック提案
      history.py        # 生成履歴(JSONL → data/history.jsonl)
    gui/
      main_window.py    # メインウィンドウ
      worker.py         # QThread: モデルロード/合成を UI スレッド外で実行、進捗シグナル
      theme.py          # Fusion スタイル + ダーク QSS
  output/               # 生成 WAV(タイムスタンプ+seed のファイル名で自動保存)
  data/                 # 設定・履歴
```

### 2-2. モデルカタログ(models.py の初期内容)
| 表示名 | checkpoint 指定 | precision 既定 |
|---|---|---|
| v4-Large (bf16) | `Aratako/Irodori-TTS-v4-Large` | bf16 / xpu |
| v4-Large INT8 | `Aratako/Irodori-TTS-v4-Large-Quantized/int8-weight-only` | **bf16 必須**(モデルカード指定)/ xpu |
| v4.1-Small (bf16) | `Aratako/Irodori-TTS-v4.1-Small` | bf16 / xpu |
| v4.1-Small INT8 | `Aratako/Irodori-TTS-v4.1-Small-Quantized/int8-weight-only` | bf16 / xpu |

### 2-3. UI 構成(単一ウィンドウ / PySide6)
- **テキスト入力**: 大きな QPlainTextEdit + 文字数カウンタ + 絵文字感情制御のヒント
- **声の指定タブ**:
  - タブ1「Voice Design」: キャプション入力(参照音声なし)
  - タブ2「音声クローン」: 参照 WAV を複数選択(合計最大120秒)、長さチェック、プレビュー再生
- **パラメータパネル**: num_steps(既定40)/ cfg_scale_text・caption・speaker / seed(+ランダムボタン・used_seed 表示)/ duration_scale / seconds 手動指定 / num_candidates(既定1)/ decode_mode(既定 sequential)/ t_schedule_mode(linear・sway)/ max_ref_seconds
- **モデル設定**: モデル選択ドロップダウン / デバイス(自動検出結果を表示: xpu → cpu)/ model_precision / codec_device・codec_precision / compile_model
- **生成**: QThread でロード→合成。ログペインに進捗(初回はモデルDL 約12.3GB+コーデック 約410MB)。生成中は生成ボタン無効化(推論の途中キャンセルは非対応のため注記を UI に表示)
- **再生・保存・履歴**: sounddevice で即時再生/停止、WAV を output/ へ自動保存、履歴リスト(ダブルクリックで再生、削除、パラメータ再呼び出し)
- ステータスバー: モデルロード状態 / 使用デバイス / 直近の生成時間

### 2-4. スレッド・ライフサイクル設計
- モデルは **初回生成時にバックグラウンドロードして以降常駐**(InferenceRuntime はキャッシュ・スレッドセーフ設計)。
- 合成リクエストは単発キュー(生成中は新規受け付けない)。UI フリーズ防止のためすべての torch 呼び出しをワーカースレッドに閉じる。
- HF ダウンロードは `snapshot_download` の進捗フックをシグナルに変換してプログレス表示。

---

## 3. 実装ステップ(次セッション以降の作業順)

1. **環境構築・疎通**
   - pyproject.toml 作成(uv の `[[tool.uv.index]]` pytorch-xpu 設定は本家 pyproject.toml をミラー。git 依存に git が必要)
   - `uv sync` → `uv run python -c "import torch; print(torch.xpu.is_available())"` が True になることを確認
   - `import irodori_tts` の動作確認(torchcodec の Windows DLL 問題が出たら対処)
2. **コア層**: settings / models / runtime_manager / history を実装(OOM 検出時は INT8 切替・codec CPU 化・sequential を提案)
3. **GUI**: 上記 UI + ワーカースレッド実装
4. **検証**
   - まず v4.1-Small bf16 で短文の XPU 合成(疎通・速度実測)
   - v4-Large bf16 で実生成(VRAM ピーク確認。OOM なら INT8 既定化 or codec を CPU へ)
   - INT8 量子化版の XPU 動作検証(公式未検証のため)
   - GUI 全体(生成→再生→保存→履歴)の動作確認
5. **仕上げ**: README.md(run.bat、ドライバ要件、初回 DL 容量と目安時間、速度実測値、ライセンス=Gemma Terms of Use の案内、トラブルシュ: XPU False → ドライバ更新、OOM → INT8+sequential、WAV 再生問題 → Issue #33 言及)

---

## 4. 参考URL

- 本体: https://github.com/Aratako/Irodori-TTS (PR #20 が XPU 対応)
- 中核コード: https://github.com/Aratako/Irodori-TTS/blob/main/irodori_tts/inference_runtime.py / `infer.py` / `docs/parameters.md`
- モデル: https://huggingface.co/Aratako/Irodori-TTS-v4-Large / https://huggingface.co/Aratako/Irodori-TTS-v4-Large-Quantized / https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small / コーデック: https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim
- サーバー版(不採用の根拠): https://github.com/Aratako/Irodori-TTS-Server
- PyTorch XPU: https://download.pytorch.org/whl/xpu/ / https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html / https://pytorch.org/blog/pytorch-2-7-intel-gpus/ / Intel前提条件: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-9.html / torchao+XPU: https://pytorch.org/blog/pytorch-2-10torchao/
- ドライバ: https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html
- Intel Arc 実績記事: https://toaru-hitorigoto.com/?p=5187 (Ubuntu/A750 改変編 — 本家公式対応済みのため改変は不要になった) / https://touch-sp.hateblo.jp/entry/2026/08/04/081532 (B580 + `uv sync --extra xpu` 実績)
- 既知 Issue: https://github.com/Aratako/Irodori-TTS/issues/33 (WAV再生) / https://github.com/Aratako/Irodori-TTS/issues/40 (torchcodec DLL)
