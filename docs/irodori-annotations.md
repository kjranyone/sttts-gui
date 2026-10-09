# Irodori への表現指示

Irodori-TTS v4.1 は読み上げる `text` 内の絵文字、`caption`、参照音声、`duration_scale` を別々に受け取る。絵文字の意味は[公式対応表](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small/blob/main/EMOJI_ANNOTATIONS.md)、条件の役割は[公式パラメータガイド](https://github.com/Aratako/Irodori-TTS/blob/main/docs/parameters.md)を参照する。絵文字の効果は確定的ではなく、声・前後の文・モデルの版で変わる。

## このアプリの受け渡し

1. Gemini Transcribe Live の確定文を原文として表示・保持する。表現用の絵文字は原文へ書き戻さない。
2. 元音声から無音と発話速度を測る。発話の途中で Gemini へ PCM を送り、VAD 終了時に確定を要求する。ローカル音響分析と任意の SER は転写と並行する。
3. 表現反映が有効で、観測値が話者内の最近の発話と大きく異なるときだけ、`⏩` / `🐢` と控えめな `duration_scale` を発話単位で選ぶ。最初の数発話では基準がないため速度を変えない。発話内の長い間が多い場合は caption に「間を取りながら」を加える。語の位置までは指定しない。
4. 任意の emotion2vec+ が明確な分類を返した場合、次の表の感情絵文字と短いスタイル文を選ぶ。配布モデルの「开心/happy」のような二言語ラベルも解釈する。曖昧・未知・期限超過は付与しない。参照音声は選択された声のままにする。
5. TTS のチャンク分割後、各チャンクの先頭へ発話単位の絵文字を付ける。`caption` には声の既定文と今回のスタイル文を結合する。`duration_scale` は今回のジョブだけに渡す。GUI の発話カードには転写文と適用した表現を分けて表示する。

| SER の分類 | 自動付与する絵文字 | caption に付ける短文 |
| --- | --- | --- |
| happy | 😊 | 楽しげに |
| sad | 😭 | 悲しげに |
| angry | 😠 | 不満げに |
| fearful | 😰 | 緊張した話し方で |
| surprised | 😲 | 驚いて |

`neutral`、`unknown`、`other`、`disgusted` は自動で演技を付けない。スコアは校正済みの感情確率として表示しない。録音の音量だけで感情を決めない。投機的 TTS を明示的に有効にした場合は、先頭チャンクの一致を優先し、自動表現付与はその発話で使わない。

## 利用者が入力できる絵文字

文字入力では絵文字をそのまま Irodori の `text` へ送る。入力欄の「演技」パレットは、依存ピン `89f9d8f` の `EMOJI_PALETTE_ITEMS` と同じ 45 種を同じ順で出す。v4 / v4.1 の[公式対応表](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small/blob/main/EMOJI_ANNOTATIONS.md)と同じ集合である。

長さ予測の `ALLOWED_ANNOTATION_EMOJIS` は 56 種で、この 45 種に ⏱️ 🍭 🎛️ 🎭 🐱 👏 💦 📄 📣 🤢 🥹 を加えたもの。追加の 11 種は公開の演技表に意味が載っていないのでパレットには含めない。入力欄へ直接打つと合成テキストへ渡る。

用途で分けた一覧は次のとおり。

| 用途 | 絵文字 |
| --- | --- |
| 話し方・感情 | 👂 😏 🥺 🫶 😭 😱 😪 ⏩ 🐢 😰 😆 💥 😠 😲 😖 😟 🫣 🙄 😊 😎 🙏 🥴 😌 🤔 💪 📖 |
| 間・息・非言語音 | 😮‍💨 ⏸️ 🤭 🥵 🌬️ 😮 👅 💋 😴 🥤 🤧 😒 🥱 👌 🎵 🤐 👃 |
| 音の効果 | 📢 📞 |

`⏸️` はテキスト内の置いた位置に間を促す指示で、ミリ秒単位の長さを保証しない。入力欄のパレットは、どの絵文字もカーソル位置へ入れる。録音内の無音を自動で特定の語に対応付けるには語単位アラインメントが必要で、Live 転写の発話単位時刻だけでは足りないため、現在は自動挿入しない。

`🤐` は公式表にあるが、[挙動が期待と異なる報告](https://github.com/Aratako/Irodori-TTS/issues/35)がある。自動分類からは挿入しない。笑い・咳・ため息なども、感情分類だけから発生したとみなさない。文中へ手動で入力した場合はそのまま送る。

## ローカル感情モデルを使う場合

標準設定は `pipeline.emotion_engine = "none"`。音響分析だけが動く。emotion2vec+ を使うには、依存を `uv sync --extra <torch extra> --extra emotion` で入れ、[emotion2vec+ base](https://huggingface.co/emotion2vec/emotion2vec_plus_base)をマイク開始前にローカルフォルダへ取得して、`data/backend.json` で次を指定する。

```json
{
  "pipeline": {
    "emotion_engine": "emotion2vec",
    "emotion_model_dir": "C:/models/emotion2vec_plus_base",
    "performance_wait_ms": 150
  }
}
```

モデルは CPU で実行し、Gemini の転写文は置き換えない。モデルの日本語での判定品質と本体の CPU 遅延は実測で検証する。起動中にモデルを自動ダウンロードしない。モデルが無い、結果が曖昧、解析が期限を超えた場合は文字列を明瞭に読み上げる。Gemini の転写が失敗した場合は自動発話しない。
