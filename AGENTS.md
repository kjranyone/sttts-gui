# AGENTS.md

sttts-gui で作業するエージェントへの指示。人間のコントリビューターにも参考になります。

## プロジェクト概要

音声対話 GUI。Rust(GPUI / gpui-kit)クライアント ⇄ Python バックエンド(faster-whisper ASR → Irodori-TTS)を stdio NDJSON で接続する。ターゲット環境は Windows + Intel Arc GPU(XPU)/CPU ASR。詳細は `README.md` と `PLAN.md` を参照。

## 開発コマンド

- 起動: `.\dev.ps1`(毎回差分 `cargo build` → GUI 起動。`-Mode mock|real`、`-Sync`、`-Backend xpu|cu128|cpu`)
- バックエンドテスト: `cd backend && uv run --no-sync pytest`(実モデル不要。fake で全体を検証する設計)
- Rust: `cargo check` / `cargo build --release`(MSRV 1.95。`rust-version` 宣言済み)

## ハードウェア検証ポリシー(最重要)

**2026-10-08 の事故**: ヘッドレス E2E プローブスクリプトが実マイク + whisper(CPU) + Irodori(XPU 初期化)を同時に起動し、Windows が BugCheck 0xD1 でクラッシュした(イベントログ 22:30、`C:\WINDOWS\Minidump\100826-9171-01.dmp` 参照)。プローブのキャンセル時に子プロセスが残りマイクを掴んだままになったことが競合の疑い。

このため:

1. **実マイク・実GPU(XPU/CUDA)に触れる検証スクリプトを新規に書かない。**
   開発段階の検証は `backend/tests/`(fake engine / fake VAD / fake source 注入)で完結させる。実装は `source_factory` / `vad_factory` / mock エンジン(`mock_load_delay_ms` 等)で差し替え可能になっている。
2. **GUI の実動作確認は Computer Use で行う。** exe を起動し、アクセシビリティ要素の操作とスクリーンショットで観察する。レベルメーターやログの確認にはスクリプトより画面観察が効く。実デバイスに触る検証はユーザーの指示時のみ、他にマイクを使用中のプロセスがないことを確認してから実施する。
3. **やむを得ずプロセスを起動する場合**: 子プロセスは必ず `try/finally` で kill し、一時スクリプトは検証後すぐ削除する。バックエンドを単体で動かす必要がある場合は `python -m sttts_server --self-check-*`(録音5秒の明示的モード)を使い、無人ループは回さない。
4. **長時間の自動観測(数十秒以上マイクを開き続ける等)はユーザーの同意を得てから。** 放置中に音声ハウリング(スピーカー音をマイクが拾い auto_speak でループする)が発生しても検知できない。

## 設計の前提(Irodori)

- **Irodori-TTS のフル機能を、このアプリが塞がない。** Irodori 本来できる設定(`SamplingRequest` の全項目、`RuntimeKey` の codec 設定、LoRA 等)が、GUI や backend の都合でできなくなっている状態を作らない。GUI が未対応の項目も `data/backend.json` の `tts.sampling`(項目名は Irodori と同じ)/ `tts.codec_*` で必ず指定できること。Irodori 側の項目が増えたら、ラッパで握りつぶさず通す。
- **発話ごとにアプリが決める項目**(`text` / `caption` / `ref_*` / `no_ref` / `seed`)だけは `tts.sampling` で上書きさせない(`RESERVED_SAMPLING_KEYS`)。黙って捨てずエラーにする。
- **TTS モデルは環境固有で一度決めたら滅多に変えない。** 主画面に出さず、GUI では「詳細設定」(既定は閉)の奥に置く。同様に、環境で一度決まる設定(デバイス・精度・codec)も主画面に増やさない。

## コーディング規約

- **リソースのライフサイクルを最後まで見る。** 「応答を返した」≠「リソースを解放した」。デバイス・セッション・子プロセス・スレッドは、状態遷移の完了(マイクのクローズ、join、kill)を保証してから次の状態へ進めること。具体的には:
  - 停止系 API(`LiveSession.stop` 等)は、戻り時点でデバイス解放を保証する(join タイムアウト時も強制クローズ)。
  - 遷移中(開始中/停止中)の再入・連打を GUI と backend の両面で拒否する。デバイスの短時間反復 open/close は BugCheck 0xD1 の実績あり(2026-10 に2度)。
  - 停止→再開にはクールダウン(`SESSION_RESTART_COOLDOWN_S`)を挟む。
  - ライフサイクルを変える変更は `test_mic_first_*` / `test_session_restart_cooldown_*` 等の該当テストを必ず通す。
- **後方互換シムを書かない。** 依存の破壊的変更はフロア引き上げで対処する(例: silero-vad 6 の `reset_states()` → `pyproject.toml` を `>=6` に)。
- バックエンドのスレッド構成(マイク → VAD スレッド → ASR ワーカー)は `backend/src/sttts_server/session.py` の docstring を参照。**ASR ロードはマイクオープンと並行**(mic-first)であり、レベルメーターはいかなるブロッキング中も止まらないこと。この挙動のテスト(`test_mic_first_*`)があるので変更時は通すこと。
- GUI⇄backend 間プロトコルは `crates/protocol/src/lib.rs` と `backend/src/sttts_server/protocol.py` の**両方を必ず同期**して変更する。
- 自動発話(auto_speak)はスピーカー音の再取り込みでループしうる。挙動を変える際はヘッドセット運用を壊さないこと。
