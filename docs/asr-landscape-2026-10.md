# 日本語 ASR ランドスケープ(2026-10 調査)

sttts-gui の要件(マイク対話・発話単位・確定 0.3〜0.6 秒・Windows CPU/onnxruntime・句読点歓迎)で
2026-10 時点の公開情報を整理した選定メモ。

## ベンチマークの現在地

[neosophie の 9 モデル比較](https://neosophie.com/en/blog/20260226-japanese-asr-benchmark)(2026-02、
RTX5090、実メディア音声 20 クリップ ≈580 秒)。**実メディア(ノイズ混じり)での計測**であり、
マイク対話(クリーン寄り)では順位が動く点に注意。

| モデル | CER | RTF | 備考 |
|---|---|---|---|
| Qwen3-ASR-1.7B | **0.140** | 0.036 | 52 言語、LLM デコーダ |
| whisper-large-v3-turbo | 0.184 | 0.013 | |
| voxtral-mini-4b-realtime | 0.212 | 0.209 | realtime 版あり、重い |
| granite-speech-4.1-2b | 0.262 | 0.051 | クリーン音声では CER ~10% だがノイズ/多話者で 70%+ に両極化 |
| cohere-transcribe-03-2026 | 0.297 | 0.063 | ja/en/zh/ko、OSS |
| parakeet-tdt-0.6b-v3 | 0.321 | **0.003** | **日本語非対応**(25 欧州言語)+ sherpa-onnx では非ストリーミング → 本案対象外 |
| reazonspeech-nemo-v2 | 0.329 | 0.020 | |
| reazonspeech-k2-v2 | 0.445 | 0.027 | 実メディアでは苦戦。クリーン音声(JSUT 系)では 8% 台の系列 |
| kotoba-whisper-v2.0 | 0.495 | 0.008 | 実メディアで最下位。対話(クリーン)では体感良好の実績あり |

## sttts-gui の視点での整理

### 最有力の新候補: Qwen3-ASR(1.7B / 0.6B)

- 日本語を含む 52 言語、上記ベンチで精度トップ
- **chunked streaming で CPU リアルタイム以下**([arXiv 2604.14493](https://arxiv.org/html/2604.14493v1)、
  最低レイテンシ 0.56 秒)
- **sherpa-onnx が Qwen3-ASR-0.6B-int8 を公式サポート**(`--extra reazonspeech` のランタイム資産が
  ほぼそのまま流用できる。`asr_reazon.py` と同じ OfflineRecognizer 形式)
- 1.7B は有志の sherpa-onnx int8 変換
  ([shigedonsan/qwen3-asr-1.7b-sherpa-onnx-int8-4096](https://huggingface.co/shigedonsan/qwen3-asr-1.7b-sherpa-onnx-int8-4096)、
  Windows x64 CPU 向け)があるが、量子化に敏感([sherpa-onnx#3535](https://github.com/k2-fsa/sherpa-onnx/issues/3535))
- [antirez/qwen-asr](https://github.com/antirez/qwen-asr)(C 実装)の評価: CPU 推論は 0.6B が最適、速度差は小
- LLM デコーダなので**母音列(あいうえお問題)や句読点出力**にも強い可能性(要検証)

### その他の動き

- **kodama-ja-streaming-small**: moonshine-streaming-small を日本語化した個人開発モデル。
  ONNX + CPU で動く軽量ストリーミング([Qiita 検証](https://qiita.com/youtoy/items/69dbeca7e9e1ecae88d9))。
  精度は要検証
- **Cohere Transcribe**(2026-03、OSS): ja/en/zh/ko
- 商用 API: Microsoft MAI-Transcribe-2-Streaming(60 言語・低遅延)、OpenAI GPT-Realtime-2。
  ローカル運用の本案には不採用

### 句読点復元(後段付けの軸)

句読点を出さないエンジ(reazonspeech / kotoba)に後段で足す定石:

- 日本語 BERT(cl-tohoku 系)の句読点付与ファインチューニング
  ([bobfromjapan/bert_japanese_punctuation](https://huggingface.co/bobfromjapan/bert_japanese_punctuation) 等)
- hayamimi(oboroge0/hayamimi)の 4 クラス句読点モデル: int8 で 37MB・約 4.6ms/行(実績値)
- 研究: Cadence(LLM ベース多言語句読点、2025)、llm-jp-modernBERT(2025)

## 現行構成の位置づけと推奨

| エンジン | 速度(発話確定) | 句読点 | 弱点 |
|---|---|---|---|
| nemotron(現行デフォ) | 0.31s 平均 | ○ | 母音列を認識しない |
| reazonspeech | 0.24s | ✗ | 固有名詞・実メディア精度 |
| kotoba | 3.7s(CPU) | ✗ | 遅い(母音列は OK) |

**推奨: Qwen3-ASR-0.6B(sherpa-onnx 公式 int8)を試験導入する。**
ランタイム共有で導入コストが最小、精度はベンチトップ級、CPU リアルタイム。
検証項目: (1) 発話単位確定レイテンシ(i5-12600KF 実測) (2) 句読点出力の有無
(3) 母音列「あいうえお」の認識 (4) 会話音声の体感精度。上手くいけば nemotron と
入れ替え or 併存選択制にする。
