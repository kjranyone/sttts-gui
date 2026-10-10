---
name: sttts-say
description: Synthesize Japanese voice lines with Irodori-TTS through the local sttts-say CLI — a fixed character voice, a per-line delivery ("say X like Y"), incremental re-rendering of a script, and per-chunk timings for subtitles. Use when a workflow needs spoken audio files (narration, character dialogue, video voice-over) generated on this machine, or when the user wants to create, audition, or pin down a character voice.
---

# sttts-say: character voice lines with Irodori-TTS

`sttts-say` turns lines of text into WAV files with Irodori-TTS on the local GPU. It never opens the microphone.
Results are JSON on stdout; progress and errors go to stderr; exit code 1 on error.

## Find the executable

1. `sttts-say` on `PATH`.
2. Otherwise download the archive for this machine from https://github.com/kjranyone/sttts-gui/releases (ask the user before downloading) and verify it against `SHA256SUMS.txt`:
   - Windows x64: `sttts-say-vX.Y.Z-x86_64-pc-windows-msvc.zip` — needs the Visual C++ Redistributable and a DirectX 12 / Vulkan GPU driver.
   - macOS Apple Silicon: `sttts-say-vX.Y.Z-aarch64-apple-darwin.tar.gz` — **experimental**. The binary is unsigned; if macOS blocks it, run `xattr -d com.apple.quarantine sttts-say`.
   - Linux x64: `sttts-say-vX.Y.Z-x86_64-unknown-linux-gnu.tar.gz` — **experimental**. Needs a Vulkan driver and OpenSSL 3 (glibc 2.35 or newer).
   The macOS / Linux builds pass CI but have not been verified on real hardware yet; if the audio sounds wrong, tell the user rather than retrying.
3. From source: `cargo build --release -p sttts-say` in the repo, then `target/release/sttts-say(.exe)`.

Data lives in the app root shared with the GUI: `data/voices/` (voices), `data/backend.json` (synthesis settings), `output/say/` (default output).
On Windows the app root is next to the executable when that folder is writable (else `%LOCALAPPDATA%\sttts-gui`); on macOS it is `~/Library/Application Support/sttts-gui`, on Linux `~/.local/share/sttts-gui` (or `$XDG_DATA_HOME/sttts-gui`). `STTTS_ROOT` overrides it.

## Rules that keep the machine safe

- **Only one GPU user at a time.** If the sttts-gui window is open, `sttts-say` stops with "Another sttts process … is using the GPU". Ask the user to close the GUI; do not retry in a loop, and do not run several `sttts-say` processes in parallel. Put many lines in one script and run `render` once instead.
- The first run downloads the model and compiles GPU kernels (can take minutes). Later runs load in seconds; each process loads the model once, so batch work into one `render`.
- You cannot hear the audio. When quality matters (choosing a voice, judging a delivery), give the user the file paths and let them choose.

## Concepts

- **Voice** (= character): `data/voices/NAME.wav|flac` (reference audio, strongest identity) and/or `data/voices/NAME.json`:
  ```json
  { "caption": "落ち着いた低めの女性の声", "seed": 123456, "sampling": { "duration_scale": 1.05 } }
  ```
  `caption` describes *who* is speaking (timbre, age, character). Unknown keys are errors.
- **Line**: `text` + optional `voice`, `style` (how this line is delivered), `caption` (replaces the voice caption), `seed`, `sampling`.
  The model receives `caption` as `"<voice caption>。話し方は<style>。"`.
- **Take**: every output `X.wav` gets `X.json` with the full settings and the seed actually used. Any take can be reproduced (`--like`) or promoted to a voice (`voice save`).

## Writing "say X like Y"

- Put the character in the voice (`caption` / reference audio), and the acting in `style`. Write both in Japanese, short and concrete:
  - style: `囁くように`, `嬉しそうに弾んだ声で`, `怒りを抑えて低く`, `泣きそうになりながら`, `早口でまくしたてるように`
  - caption: `元気な十代の少年の声`, `穏やかな年配の男性ナレーター`, `透明感のある若い女性の声`
- Emoji inside `text` also steer Irodori (e.g. `😭`, `😊`, `⏸️` for a pause, `🤭` for a giggle). See the repo's `docs/irodori-annotations.md`.
- Speed: `"sampling": {"duration_scale": 1.2}` is slower, `0.85` is faster. Other Irodori options go in `sampling` with Irodori's own names; the reserved keys (`text`, `caption`, `ref_*`, `no_ref`, `seed`) are rejected.

## Commands

```sh
# One line
sttts-say speak --text "おはよう、今日もいい天気だね" --voice mio --style "眠そうに" --out out/l01.wav

# Retake: same settings, new seed / different style
sttts-say speak --like out/l01.wav --seed random --out out/l01b.wav
sttts-say speak --like out/l01.wav --style "照れながら" --out out/l01c.wav

# A whole script (JSONL, one line per take) -> DIR/<id>.wav + DIR/<id>.json + DIR/manifest.json
sttts-say render ep1.jsonl --out-dir ep1_audio

# Voices
sttts-say voice list
sttts-say voice show mio

# Models
sttts-say model list
sttts-say model use v4.1-small
```

Script (`ep1.jsonl`):
```jsonl
{"id":"s01","voice":"mio","text":"ねえ、聞いて!","style":"興奮気味に"}
{"id":"s02","voice":"narrator","text":"その日、町は静まり返っていた。"}
{"id":"s03","voice":"mio","text":"……ごめんね。","style":"消え入りそうに","sampling":{"duration_scale":1.15}}
```

`render` is incremental: a line whose settings match its previous take is left as is (`"status":"unchanged"`), and when nothing changed the model is not even loaded. To change one line, edit it and re-run. To force a retake of a line, delete its `.wav` or give it a different `seed`.
All lines are validated (voices exist, sampling keys valid, ids unique) before anything is synthesized.

## Choosing the model

`sttts-say model list` shows the models, which one is current, and whether its weights are already downloaded.

- `v4.1-small-mf` (default): MeanFlow, 4 steps. Fast; good for drafts and auditions.
- `v4.1-small`: RF, 40 steps with guidance (CFG). Reads kanji and clones the reference voice more accurately, at about 20 times the computation. Prefer it for final renders when time allows.
- `v4.1-small-int8`: `v4.1-small` with int8 weights, for GPUs with little memory (0.9 GB download).
- `v4-large`: 3.3B parameters; best at following captions and long reference audio. Needs about 16 GB of GPU memory (13 GB download).
- `v4-large-int8`: `v4-large` with int8 weights, for mid-range GPUs (3.8 GB download).

The v4 Large models are subject to the Gemma Terms of Use (`model use` prints the link); tell the user before switching to them.

`sttts-say model use NAME` records the choice in `data/backend.json` (`tts.model`); it is not a per-command flag. Ask the user before switching (the first use downloads the weights, see `download_mb`). After switching, `render` re-synthesizes every line, because the model is part of each take's settings. RF-only options (`cfg_scale_text`, `cfg_scale_speaker`, `cfg_guidance_mode`, ...) go in `sampling`; MeanFlow ignores them.

## Pinning a character (do this before producing many lines)

A voice without reference audio is defined only by caption + seed, and its timbre drifts when the text changes. Fix it with a reference take:

1. Audition seeds: `sttts-say audition --text "はじめまして、ミオです。よろしくね。" --caption "明るい十代の女の子の声" --count 6`
   (a neutral-ish line of 5–10 seconds works best as a future reference).
2. Ask the user to listen and pick one of the returned `takes[].out`.
3. Save it: `sttts-say voice save mio --from <chosen take .wav>`
   This copies the take as `data/voices/mio.wav` (reference audio) and stores its caption (without the per-line style) and seed in `mio.json`. Existing voices are never overwritten.
4. From then on use `--voice mio` / `"voice":"mio"`; only `style` changes per line.

Voices saved this way also appear in the GUI's voice bank (the GUI uses the reference audio; the caption and seed in `NAME.json` are used by `sttts-say`). Only use reference audio of people who consented; impersonating real people is prohibited by the Irodori-TTS model cards.

## Using the output in a video

- Each take result has `duration_ms` and `segments` (`text`, `start_ms`, `end_ms` per synthesized chunk) — use them for subtitle timing (SRT) and for placing clips on a timeline. `manifest.json` in the render directory keeps the same data for later.
- Audio is mono 16-bit PCM WAV. Concatenate / mix with ffmpeg, e.g. `ffmpeg -f concat -safe 0 -i list.txt -c copy dialogue.wav`.
