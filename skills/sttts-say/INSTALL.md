# Installing sttts-say (instructions for an AI agent)

The user asked you to install `sttts-say` and its Agent Skill. `sttts-say` is a local CLI that turns Japanese lines into WAV files with Irodori-TTS on the GPU, in a chosen character voice.
Follow these steps in order. Report to the user what you installed and where. Do not synthesize anything during the install.

## 1. Check the machine

| OS | Asset | Status |
|---|---|---|
| Windows x64 | `sttts-say-vX.Y.Z-x86_64-pc-windows-msvc.zip` | supported (verified on Intel Arc) |
| macOS Apple Silicon | `sttts-say-vX.Y.Z-aarch64-apple-darwin.tar.gz` | experimental (CI only, not verified on real hardware) |
| Linux x64 | `sttts-say-vX.Y.Z-x86_64-unknown-linux-gnu.tar.gz` | experimental (CI only); needs a Vulkan driver, OpenSSL 3, glibc 2.35+ |

- **No match:** for any other OS or CPU (Intel Mac, ARM Linux, ...), stop and tell the user there is no build for it.
- **Experimental builds:** on macOS / Linux, go on and mention in your report that the build is experimental.
- **GPU required:** a GPU is required (DirectX 12 / Vulkan on Windows, Metal on macOS). There is no CPU mode.
- **Windows runtime:** Windows also needs the Visual C++ Redistributable (x64). If `sttts-say --help` fails to start with a missing `VCRUNTIME140.dll`, tell the user to install it from https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist.

## 2. Download the latest release and verify it

1. **Find the release.** Get the latest release tag from `https://api.github.com/repos/kjranyone/sttts-gui/releases/latest` (`tag_name`, e.g. `v0.2.0`; or use `gh release view --repo kjranyone/sttts-gui`).
2. **Download the files.** Download the asset for this machine and `SHA256SUMS.txt` from `https://github.com/kjranyone/sttts-gui/releases/download/<tag>/<file>`. Save them into a new, empty temporary folder.
3. **Check the hash.** Verify the archive's SHA-256 against its line in `SHA256SUMS.txt`:
   - Windows: `Get-FileHash -Algorithm SHA256`
   - macOS: `shasum -a 256`
   - Linux: `sha256sum`

   If it does not match, delete the download and stop.

## 3. Install the executable

The archive contains `sttts-say(.exe)` and `skills/sttts-say/` (the Skill matching this binary), plus licenses and READMEs. On macOS / Linux, the archive's files are inside a top-level folder.

- **Windows:** extract the whole zip into `%LOCALAPPDATA%\Programs\sttts-say\`. Then add that folder to the user's `PATH`, with this PowerShell code:
  ```powershell
  $dir = "$env:LOCALAPPDATA\Programs\sttts-say"
  $p = [Environment]::GetEnvironmentVariable('Path', 'User')
  if (($p -split ';') -notcontains $dir) { [Environment]::SetEnvironmentVariable('Path', "$p;$dir", 'User') }
  ```
  New terminals see the change. In the current session, call the full path.
- **macOS / Linux:** copy `sttts-say` to `~/.local/bin/sttts-say` and run `chmod +x` on it. If `~/.local/bin` is not on `PATH`, tell the user, and use the full path meanwhile.
  - On macOS, if the binary is blocked as unsigned, run `xattr -d com.apple.quarantine ~/.local/bin/sttts-say`.

Voices, settings and outputs are stored in the app root:
- Windows: next to the executable (`%LOCALAPPDATA%\Programs\sttts-say\data`, `output`).
- macOS: `~/Library/Application Support/sttts-gui`.
- Linux: `~/.local/share/sttts-gui`.

## 4. Install the Skill

Copy the archive's `skills/sttts-say/` folder (not the one on GitHub `main`, which can be ahead of the release) into your skills directory:

- **Claude Code:** `~/.claude/skills/sttts-say/` (available in every project). For one project only, use `<project>/.claude/skills/sttts-say/`.
- **Other agents:**
  - If your agent loads Agent Skills (`SKILL.md` with front matter), use its personal skills directory.
  - Otherwise, keep the folder next to the executable and tell the user to point the agent at `SKILL.md` when they want voice lines.

If the folder already exists, replace it, because it describes the old binary.

## 5. Check the install (no GPU, no download)

```sh
sttts-say --help
sttts-say voice list
```

Both must succeed. The model is not downloaded yet: synthesis downloads it on first use and compiles GPU kernels. This takes a few minutes and needs about 3.5 GB for the default model.

## 6. Tell the user

Tell the user:
- **What you installed:** the version, the executable path, the Skill path, and whether `PATH` changed.
- **How to use it:** ask in plain words, e.g. 「このセリフを落ち着いた女性の声で読んで」 or "make narration for this script". The Skill explains the rest to the agent.
- **First run:** the first synthesis downloads about 3.5 GB and takes minutes. Later calls take about a second.
- **Rules:**
  - Only use reference audio of people who consented. Impersonating real people is prohibited by the Irodori-TTS model cards.
  - The larger v4 models fall under the Gemma Terms of Use.

Do not make a test line: the first real request downloads the model.

## Updating and uninstalling

- **Update:** repeat steps 2–4 with the new release.
- **Uninstall:**
  1. Run `sttts-say stop` (newer versions keep a background process for a few minutes).
  2. Delete the executable, the Skill folder and the app root.
  3. On Windows, remove the folder from `PATH`.
  4. The downloaded models are in the HuggingFace cache (`~/.cache/huggingface/hub`, models whose names start with `Aratako`). Delete them only if the user asks.
