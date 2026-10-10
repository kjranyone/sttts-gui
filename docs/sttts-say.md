# sttts-say(ヘッドレス合成 CLI)

GUI を開かずに、Irodori-TTS でセリフを WAV にする CLI。エージェント(Agent Skills)や動画制作のワークフローから
「このキャラクターに、このセリフを、こう喋らせる」部品として呼ぶためのもの。マイクは開かない。

- 本体: `crates/engine/src/say.rs`(テストは偽の TTS で完結。`cargo test -p sttts-engine say::`)
- バイナリ: `crates/say`(`sttts-say.exe`。配布は GitHub Releases、ソースからは `cargo build --release -p sttts-say`。`live` feature なしの engine でビルドし、マイク・ASR のコードを含まない)
- Skill: `skills/sttts-say/SKILL.md`(`~/.claude/skills/` か、使うプロジェクトの `.claude/skills/` へコピーする)

## GPU のプロセス間ロック

GUI と CLI が同時に GPU を初期化しないよう、GPU を使う前に `%TEMP%\sttts-gpu.lock` を排他ロックし、プロセスの終了まで持つ
(`sttts_engine::tts::gpu_device`)。後から来た側は待たずにエラーで止まる。GUI は起動直後のウォームアップで取るので、
GUI を開いている間は `sttts-say` は合成できない。ロックはプロセスが終われば OS が外すので、強制終了しても残らない。

## キャラクターを固定する仕組み

Irodori の声質は「参照音声 > caption + seed」の順に強く効く。参照音声なしの声は caption と seed だけで決まり、
seed が同じでも文が変わると声質が揺れる。そこで CLI は次の 3 つを単位にする。

| 単位 | 実体 | 役割 |
|---|---|---|
| 声 | `data/voices/<名前>.{wav,flac}` と `<名前>.json`(`caption` / `seed` / `sampling`) | キャラクター。片方だけでもよい。wav は GUI の声バンクと共通 |
| テイク | 出力 `X.wav` と `X.json`(行・解決済みの指定・実際の seed・区間) | 再現(`--like`)と声への昇格(`voice save`)の元 |
| 台本 | JSONL(1 行 = 1 テイク) | 増分レンダリング。指定が前回と同じ行は合成しない |

- **オーディション → 昇格**: `audition` で seed だけを変えた候補を作り、人が聴いて選んだテイクを `voice save` で声にする。
  テイクの音声がそのまま参照音声になり、声質の caption(行の話し方は除く)と seed が `<名前>.json` に残る。既存の声は上書きしない。
- **話し方は行ごと**: 声の caption に行の `style` を重ねる(`<声質>。話し方は<style>。`)。自動発話の表現計画と同じ書き方(`performance::compose_caption`)。
- **seed は 1 テイクで共通**: 長いセリフはチャンクに分けて合成するが、seed はテイク単位で 1 つ(チャンク間で声質が変わらない)。
  ファイル出力では初音を急がないので、先頭の短縮はせず `pipeline.chunk_max_chars` 程度まで文を束ねる。
- **増分レンダリングの比較キー**は解決済みの指定(本文・caption・参照音声の中身のハッシュ・sampling・モデル・決めてある seed)。
  seed を決めていない行は、前回たまたま出た seed のテイクが保たれる(気に入ったものが勝手に変わらない)。撮り直しは wav を消すか seed を変える。
- **誤りは合成前に全部出す**: 台本は全行を解決(声の存在・sampling の予約キー/未知のキー・id の重複)してから合成を始める。
  何も変わっていなければ TTS をロードしない(GPU に触れない)。

## 設定

GUI と同じ `data/backend.json` を読む(`tts.model` / `tts.num_steps` / `tts.sampling` / `voice` / `pipeline.chunk_max_chars`)。
sampling は `tts.sampling` → 声 → 行の順に重なる。声を指定しない行は `backend.json` の `voice` を使う。
表示言語(標準エラーの文言)は GUI と同じく `data/config.json` の `language` → OS の表示言語。

使い方の詳細(コマンド・台本の書式・caption の書き方)は `skills/sttts-say/SKILL.md` を参照。
