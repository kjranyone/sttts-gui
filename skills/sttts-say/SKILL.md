---
name: sttts-say
description: Synthesize Japanese voice lines with Irodori-TTS through the local sttts-say CLI — a fixed character voice, a per-line delivery ("say X like Y"), incremental re-rendering of a script, and per-chunk timings for subtitles. Use when a workflow needs spoken audio files (narration, character dialogue, video voice-over) generated on this machine, or when the user wants to create, audition, or pin down a character voice.
---

# sttts-say: character voice lines with Irodori-TTS

`sttts-say` turns lines of text into WAV files with Irodori-TTS on the local GPU. It never opens the microphone.
Results are JSON on stdout; progress and errors go to stderr; exit code 1 on error.

## Find the executable

1. `sttts-say` on `PATH`.
2. The install location used by [INSTALL.md](INSTALL.md): `%LOCALAPPDATA%\Programs\sttts-say\sttts-say.exe` (Windows) or `~/.local/bin/sttts-say` (macOS / Linux).
3. Otherwise install it by following [INSTALL.md](INSTALL.md) (ask the user first). It downloads the release for this machine, verifies it against `SHA256SUMS.txt`, and installs the matching Skill:
   - Windows x64 is supported.
   - macOS Apple Silicon and Linux x64 are **experimental**: they pass CI but have not been verified on real hardware. If the audio sounds wrong, tell the user rather than retrying.
4. From source: `.\dev.ps1 say` (Windows) or `./dev.sh say` (macOS / Linux) in the repo. The executable is `target/release/sttts-say(.exe)`.

Data lives in the app root shared with the GUI: `data/voices/` (voices), `data/backend.json` (synthesis settings), `output/say/` (default output).
On Windows the app root is next to the executable when that folder is writable (else `%LOCALAPPDATA%\sttts-gui`); on macOS it is `~/Library/Application Support/sttts-gui`, on Linux `~/.local/share/sttts-gui` (or `$XDG_DATA_HOME/sttts-gui`). `STTTS_ROOT` overrides it.

## Rules that keep the machine safe

- **Only one GPU user at a time.** If sttts-gui is using the GPU, `sttts-say` stops with "Another sttts process … is using the GPU". Ask the user to close the GUI; do not retry in a loop, and do not run several `sttts-say` processes in parallel (they queue behind each other anyway).
- The very first run downloads the model and compiles GPU kernels (can take minutes). After that, synthesis goes through a background process that keeps the model loaded (see [The background process](#the-background-process)). **Run `sttts-say stop` when your task is finished**, so the GPU memory is freed.
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

# Release the GPU now (ends the background process that keeps the model loaded)
sttts-say stop
```

Script (`ep1.jsonl`):
```jsonl
{"id":"s01","voice":"mio","text":"ねえ、聞いて!","style":"興奮気味に"}
{"id":"s02","voice":"narrator","text":"その日、町は静まり返っていた。"}
{"id":"s03","voice":"mio","text":"……ごめんね。","style":"消え入りそうに","sampling":{"duration_scale":1.15}}
```

`render` is incremental: a line whose settings match its previous take is left as is (`"status":"unchanged"`), and when nothing changed the model is not even loaded. To change one line, edit it and re-run. To force a retake of a line, delete its `.wav` or give it a different `seed`.
All lines are validated (voices exist, sampling keys valid, ids unique) before anything is synthesized.

## The background process

`speak`, `render` and `audition` do not synthesize in the process you start. They hand the work to a background `sttts-say serve` process that keeps the model loaded on the GPU, so only the first call pays for loading the model.

| Call (default model, one short line) | Time |
|---|---|
| First call (starts the background process and loads the model) | about 9 s |
| Later calls | about 1–2 s |

What this means for you:

- **Nothing to set up.** The first synthesis call starts the background process. Never run `sttts-say serve` yourself.
- **Calls behave the same as before.** Your call waits until the result is ready. stdout JSON, stderr progress, exit codes and relative paths work the same as before. The background process runs each request in your current directory. If it cannot be used, your call synthesizes by itself and gives the same output.
- **Settings are read on every call.** Your call reads `data/backend.json`, `STTTS_ROOT` and the display language and sends them with each request. Voices and scripts are read fresh for each request. Edits take effect on the next call, with no restart. After `model use` (or a change to `tts.num_steps`), the next call frees the old model and loads the new one, so that call is slow again.
- **One request at a time.** Concurrent calls queue up and run one after another, so running them in parallel gives no speedup. Several lines are still best as one `render` script (incremental, one manifest).
- **It outlives your call, on purpose.** A `sttts-say` process that stays after your command returns is the background process, not a leak. Do not kill it, and do not delete its files in the temp directory (`sttts-say-server.json` / `.lock`).
- **It holds the GPU while it lives.** It keeps the GPU lock and the model's GPU memory (up to about 16 GB with `v4-large`) even while idle.
- **How it ends:**
  - It exits by itself after 10 minutes without requests.
  - `sttts-say stop` ends it now. If it is synthesizing, it finishes first. Output: `{"stopped": true}`, or `{"stopped": false}` when none was running.
  - Run `stop` when your task is done, and before the user runs other GPU-heavy work.
- **With the GUI:**
  - If sttts-gui starts while the background process is idle, the background process exits and gives the GPU to the GUI.
  - If the GUI is already using the GPU, synthesis fails with "Another sttts process … is using the GPU". Ask the user to close the GUI.
- **After an update.** If the `sttts-say` executable is replaced by a new build, the next call stops the old background process and starts a new one.
- **If it dies mid-request.** If the background process dies during a request, the call fails with "The sttts-say background process stopped unexpectedly". The next call starts a fresh one. Retry once, not in a loop.
- **Security.** It listens only on 127.0.0.1 and accepts only requests that carry a random token from its registration file.

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
