"""プロトコル定数・カタログ。Rust側 crates/protocol/src/lib.rs と同期して変更すること。"""

from __future__ import annotations

PROTOCOL_VERSION = 1

# エンジン phase
IDLE = "idle"
LOADING = "loading"
READY = "ready"
ERROR = "error"

# TTS モデルカタログ(hello メッセージでGUIへ通知)。
# id はエイリアス。実 checkpoint への写像は engines/tts_irodori.py の CATALOG を参照。
MODEL_CATALOG: list[dict] = [
    {
        "id": "v4.1-small-mf",
        "label": "Irodori v4.1 Small MeanFlow(高速・会話向け)",
        "size": "約766M / 4steps",
        "note": "ストリーミング会話の既定候補",
    },
    {
        "id": "v4.1-small",
        "label": "Irodori v4.1 Small(RF 40steps)",
        "size": "約766M",
        "note": "小規模・標準品質",
    },
    {
        "id": "v4-large",
        "label": "Irodori v4 Large(高品質)",
        "size": "約3.29B / bf16 約6.6GB VRAM",
        "note": "高品質モード。B570 では生成が遅い場合あり",
    },
    {
        "id": "v4-large-int8",
        "label": "Irodori v4 Large INT8(低VRAM)",
        "size": "約3.29B / 重み約3.7GB",
        "note": "OOM時のフォールバック。XPU公式未検証",
    },
    {
        "id": "v4.1-small-int8",
        "label": "Irodori v4.1 Small INT8",
        "size": "約766M",
        "note": "",
    },
]
