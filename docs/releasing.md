# リリース手順

sttts-gui のバージョンの付け方と、GitHub Releases へ配布物を出す手順。ビルドと公開は GitHub Actions(`.github/workflows/release.yml`)が行い、
手元でやるのは「バージョンと CHANGELOG を書いてタグを push する」ことだけ。

## 決まりごと

| 項目 | 決まり |
|---|---|
| バージョン | [SemVer](https://semver.org/lang/ja/)。ワークスペース共通で `Cargo.toml` の `[workspace.package] version` の 1 か所 |
| タグ | `vX.Y.Z`。プレリリースは `vX.Y.Z-rc.1` など(`-` を含むタグは Releases でプレリリース扱いになる) |
| 変更履歴 | `CHANGELOG.md`([Keep a Changelog](https://keepachangelog.com/ja/1.1.0/))。普段の変更は `[Unreleased]` に書きためる |
| 配布物 | Windows x64(GUI+CLI と CLI のみ)、macOS・Linux 向けの CLI(実験的)、`SHA256SUMS.txt`(下記) |
| モデル | 同梱しない。初回実行時に HuggingFace から取得される |

### バージョンの上げ方(0.x の間)

1.0 になるまでは、SemVer の慣習どおり **マイナーを破壊的変更に使う**。

- `0.Y.0` に上げる: 利用者の手直しが要る変更。例: `sttts-say` の引数・出力 JSON・台本(JSONL)の書式の非互換、`data/backend.json` / `data/voices/*.json` の項目名の変更、必要なドライバ・OS の変更。
- `0.Y.Z` に上げる: 互換を保った機能追加と修正。

`sttts-say` の出力 JSON と台本の書式は、エージェントやスクリプトから使われる外向けの仕様として扱う。

### CHANGELOG の書き方

- 利用者から見た変化を書く(内部のリファクタリングは書かない)。見出しは `Added` / `Changed` / `Deprecated` / `Removed` / `Fixed` / `Security`。
- 破壊的変更は `Changed` か `Removed` の先頭に **BREAKING:** と書き、移行方法を 1 行添える。
- 英語で書く(GitHub のリリースノートにそのまま使われる)。

## 手順

### 1. 準備

- main が CI(`.github/workflows/ci.yml`)を通っていること。
- 手元でテストを通す:

  ```powershell
  cargo test -p sttts-engine -p sttts-say -p sttts-i18n -p sttts-protocol
  ```

- GUI の実動作を確認する(AGENTS.md の方針どおり、exe を起動して Computer Use で操作する)。実マイク・実 GPU を使う確認は、マイクを使う他のプロセスがないことを確かめてから行う。

### 2. バージョンと CHANGELOG を更新する

1. `Cargo.toml` の `[workspace.package] version` を新しいバージョンにする。
2. **`Cargo.lock` を更新する**(`cargo check -p sttts-gui -p sttts-say` を 1 回実行すればよい)。
   リリース用ワークフローは `--locked` でビルドするので、`Cargo.lock` を更新し忘れるとビルドが失敗する。
3. `CHANGELOG.md` の `## [Unreleased]` の下に `## [X.Y.Z] - YYYY-MM-DD` を作り、Unreleased の中身をそこへ移す(`[Unreleased]` の見出しは空で残す)。
4. 末尾のリンク定義を更新する:

   ```markdown
   [Unreleased]: https://github.com/kjranyone/sttts-gui/compare/vX.Y.Z...HEAD
   [X.Y.Z]: https://github.com/kjranyone/sttts-gui/compare/vW.W.W...vX.Y.Z
   ```

   最初のリリースは `[0.1.0]: https://github.com/kjranyone/sttts-gui/releases/tag/v0.1.0` にする。

### 3. コミットしてタグを push する

```powershell
git switch -c release/vX.Y.Z
git commit -am "Release vX.Y.Z"
git push -u origin release/vX.Y.Z      # PR を作り、CI が通ったら main へマージ
git switch main
git pull
git tag -a vX.Y.Z -m "vX.Y.Z"           # マージ後の main のコミットに付ける
git push origin vX.Y.Z
```

タグは **CI を通った main のコミット** に付ける。タグの push がリリースの合図になる。

### 4. ワークフローを見守る

```powershell
gh run watch     # 実行中のワークフローを選んで経過を見る
```

ワークフローは次のジョブからなる。どれかが失敗すると Releases には何も出ない。

1. **meta**: タグと `Cargo.toml` の version が一致するか、`CHANGELOG.md` に `## [X.Y.Z]` の節があるかを確かめ、リリースノートを作る
2. **windows**: テスト → `sttts-gui` と `sttts-say` を **別々の cargo 呼び出しで** リリースビルド → zip 2 種
   (一緒にビルドすると features が統合され、`sttts-say` にマイク・ASR が入るため)
3. **unix**(macOS Apple Silicon / Linux x64): テスト → `sttts-say` だけをリリースビルド → tar.gz。Linux は glibc を古めにするため ubuntu-22.04 でビルドする
4. **publish**: 全部の配布物の `SHA256SUMS.txt` を書き、GitHub Release を作る

### 5. 配布物を確かめる

[Releases](https://github.com/kjranyone/sttts-gui/releases) に次の 5 つがあること。

| ファイル | 中身 |
|---|---|
| `sttts-gui-vX.Y.Z-x86_64-pc-windows-msvc.zip` | `sttts-gui.exe`、`sttts-say.exe`、`skills/sttts-say/`、README 3 言語、LICENSE、PRIVACY、CHANGELOG |
| `sttts-say-vX.Y.Z-x86_64-pc-windows-msvc.zip` | `sttts-say.exe`(マイク・ASR なし)、`skills/sttts-say/`、README 3 言語、LICENSE、PRIVACY、CHANGELOG |
| `sttts-say-vX.Y.Z-aarch64-apple-darwin.tar.gz` | macOS 版 `sttts-say`(実験的)と同じ文書一式 |
| `sttts-say-vX.Y.Z-x86_64-unknown-linux-gnu.tar.gz` | Linux 版 `sttts-say`(実験的)と同じ文書一式 |
| `SHA256SUMS.txt` | 上の 4 つの SHA-256 |

macOS / Linux 版は実機で未検証の実験版で、リリースノートにもそう書かれる。実機で音声が正しく出ることを確かめたら、
README とリリースノートの「experimental」を外す(ワークフローの meta ジョブの文言)。

ダウンロードして確かめる:

```powershell
gh release download vX.Y.Z -D $env:TEMP\sttts-vX.Y.Z
cd $env:TEMP\sttts-vX.Y.Z
Get-FileHash *.zip -Algorithm SHA256      # SHA256SUMS.txt と一致すること
Expand-Archive sttts-say-vX.Y.Z-x86_64-pc-windows-msvc.zip say
.\say\sttts-say.exe help                  # GPU に触れない
.\say\sttts-say.exe voice list            # GPU に触れない
```

合成(GPU を使う)まで確かめる場合は、GUI を閉じたうえで短い 1 行だけにする(`speak --text "テスト"`)。

## 失敗したとき

- **公開前に失敗した**(Release が作られていない): 原因を直して main に入れ、タグを付け直す。

  ```powershell
  git push origin :refs/tags/vX.Y.Z   # リモートのタグを消す
  git tag -d vX.Y.Z
  # 修正をマージしてから、手順 3 のタグ付けをやり直す
  ```

- **公開後に問題が見つかった**: 公開したタグは動かさない(ダウンロード済みの利用者と食い違うため)。修正して次のパッチ(`vX.Y.Z+1`)を出す。
  深刻な問題なら、GitHub 上でその Release の説明に注意書きを足すか、プレリリースに戻す。

## 補足

- 署名は SignPath Foundation に申請中(README の「Code signing policy」)。承認されるまで exe は署名なしで、初回起動時に SmartScreen の警告が出る。承認後は release.yml の zip 作成の前に SignPath の署名ステップを足し、README の「Status」の注記を消す。
- exe のバージョン情報(ProductName `sttts-gui` と ProductVersion)は `Cargo.toml` から生成している(`crates/gui/build.rs` / `crates/say/build.rs`)。SignPath はこの一致を確かめる。
- 配布先に必要なもの: [Visual C++ 再頒布可能パッケージ](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist)(x64)、DirectX 12 / Vulkan に対応した GPU ドライバ。
- ビルドには libclang が要る(cpal の ASIO 対応)。GitHub の Windows ランナーには LLVM が入っており、ワークフローは `LIBCLANG_PATH` でそれを指している。
