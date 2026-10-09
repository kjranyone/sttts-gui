# AGENTS.md

sttts-gui で作業するエージェントへの指示。人間のコントリビューターにも参考になります。

## プロジェクト概要

音声対話 GUI。Rust(GPUI / gpui-kit)の 1 プロセスで、マイク → Silero VAD → ASR(Nemotron / kotoba-whisper / Gemini)→ Irodori-TTS を動かす。
Python は使わない。ターゲット環境は Windows + Intel Arc GPU(Vulkan / wgpu)。詳細は `README.md` と `docs/irodori-rs.md` を参照。

- `crates/gui` — GPUI クライアント(`backend.rs` がエンジンをプロセス内で起動する)
- `crates/engine` — バックエンド本体(設定・チャンク分割・TTS ワーカー・セッション・投機的 TTS)。外界は `Platform` トレイトで注入(テストは偽物)
- `crates/irodori` — Irodori-TTS の純 Rust 推論(burn)
- `crates/whisper` / `crates/nemotron` / `crates/gemini` — ASR エンジン
- `crates/audio` — マイク(cpal)・リサンプル・Silero VAD・WAV ソース
- `crates/protocol` — GUI⇄エンジンのメッセージ型

## 開発コマンド

- 起動: `.\dev.ps1`(毎回差分 `cargo build` → GUI 起動。`-Mode mock|real`、`-DebugBuild`)
- テスト: `cargo test -p sttts-engine`(実モデル・実デバイス不要。偽物で全体を検証する設計)
- 各クレートのパリティテスト(`crates/irodori` `crates/whisper` `crates/nemotron` `crates/audio`)は、PyTorch 等の参照データ(`tools/reference/` のスクリプトで作る)が無い環境では `eprintln!` して戻る
- Rust: `cargo check` / `cargo build --release`(MSRV 1.95。`rust-version` 宣言済み)

## ハードウェア検証ポリシー(最重要)

**2026-10-08 の事故**: ヘッドレス E2E プローブスクリプトが実マイク + whisper(CPU) + Irodori(XPU 初期化)を同時に起動し、Windows が BugCheck 0xD1 でクラッシュした(イベントログ 22:30、`C:\WINDOWS\Minidump\100826-9171-01.dmp` 参照)。プローブのキャンセル時に子プロセスが残りマイクを掴んだままになったことが競合の疑い。

このため:

1. **実マイク・実GPU に触れる検証スクリプトを新規に書かない。**
   開発段階の検証は `crates/engine` のテスト(偽の `Platform` / 偽の `AudioSource` / 偽の `Vad` / `MockAsr` / `MockTts` を注入)で完結させる。
2. **GUI の実動作確認は Computer Use で行う。** exe を起動し、アクセシビリティ要素の操作とスクリーンショットで観察する。レベルメーターやログの確認にはスクリプトより画面観察が効く。実デバイスに触る検証はユーザーの指示時のみ、他にマイクを使用中のプロセスがないことを確認してから実施する。
3. **やむを得ずプロセスを起動する場合**: 子プロセスは必ず終了させ、一時スクリプトは検証後すぐ削除する。無人ループは回さない。
4. **長時間の自動観測(数十秒以上マイクを開き続ける等)はユーザーの同意を得てから。** 放置中に音声ハウリング(スピーカー音をマイクが拾い auto_speak でループする)が発生しても検知できない。

## 設計の前提(Irodori)

- **Irodori-TTS のフル機能を、このアプリが塞がない。** Irodori 本来できる設定(`SamplingRequest` の全項目等)が、GUI やエンジンの都合でできなくなっている状態を作らない。GUI が未対応の項目も `data/backend.json` の `tts.sampling`(項目名は Irodori と同じ)で必ず指定できること。Irodori 側の項目が増えたら、ラッパで握りつぶさず通す(未知の項目は黙って捨てず、エラーで知らせる)。
- **発話ごとにアプリが決める項目**(`text` / `caption` / `ref_*` / `no_ref` / `seed`)だけは `tts.sampling` で上書きさせない(`RESERVED_SAMPLING_KEYS`)。黙って捨てずエラーにする。
- **Irodori の推論は GPU で行い、CPU 推論は実装しない。** 速度のために GPU を使うのが前提で、CPU への既定切替・フォールバックは作らない(codec も同じデバイス)。デバイスの問題は GPU 側で直す。burn の CPU バックエンド(flex)はパリティテスト用の参照実装であり、本番経路ではない。
- **TTS モデルは環境固有で一度決めたら滅多に変えない。** 主画面に出さず、GUI では「詳細設定」(既定は閉)の奥に置く。同様に、環境で一度決まる設定(デバイス・精度・codec)も主画面に増やさない。

## コーディング規約

- **リソースのライフサイクルを最後まで見る。** 「応答を返した」≠「リソースを解放した」。デバイス・セッション・スレッドは、状態遷移の完了(マイクのクローズ、join)を保証してから次の状態へ進めること。具体的には:
  - 停止系 API(`LiveSession::stop` 等)は、戻り時点でデバイス解放を保証する(join タイムアウト時も強制クローズ)。
  - 遷移中(開始中/停止中)の再入・連打を GUI とエンジンの両面で拒否する。デバイスの短時間反復 open/close は BugCheck 0xD1 の実績あり(2026-10 に2度)。
  - 停止→再開にはクールダウン(`SESSION_RESTART_COOLDOWN_S`)を挟む。
  - ライフサイクルを変える変更は `mic_first_*` / `session_restart_cooldown_*` / `live_session_*` 等の該当テストを必ず通す。
- **余計なフラグを立てない。** 必要な処理(モデル取得・ディレクトリ作成など)は既定で自動実行し、ユーザーに手動の前準備やオプション指定を要求しない。設定は一度決まれば記録して再指定を不要にする。フラグを足したくなったら、まず「何も指定しなくても正しく動く」設計にできないかを考える。
- **後方互換シムを書かない。** 依存の破壊的変更はフロア引き上げで対処する。
- エンジンのスレッド構成(マイク → VAD スレッド → ASR ワーカー)は `crates/engine/src/session.rs` の docs を参照。**ASR ロードはマイクオープンと並行**(mic-first)であり、レベルメーターはいかなるブロッキング中も止まらないこと。この挙動のテスト(`mic_first_*`)があるので変更時は通すこと。
- GUI⇄エンジン間のメッセージは `crates/protocol/src/lib.rs` が唯一の定義。
- 自動発話(auto_speak)はスピーカー音の再取り込みでループしうる。挙動を変える際はヘッドセット運用を壊さないこと。
