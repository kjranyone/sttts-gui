# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Model choice: Irodori-TTS v4.1 Small (RF, 40 steps with classifier-free guidance) besides the default v4.1 Small MeanFlow. It reads kanji and clones voices more accurately at about 20 times the computation, which suits `sttts-say`. Choose it with `sttts-say model use v4.1-small` (recorded in `data/backend.json` as `tts.model`) or in the GUI's Advanced settings.
- `sttts-say model list` / `model use NAME`.
- The RF sampler options of Irodori in `tts.sampling` (and in the GUI's synthesis parameters): `cfg_scale_text`, `cfg_scale_caption`, `cfg_scale_speaker`, `cfg_scale`, `cfg_guidance_mode`, `cfg_min_t`, `cfg_max_t`, `truncation_factor`, `rescale_k`, `rescale_sigma`, `speaker_kv_scale`, `speaker_kv_min_t`, `speaker_kv_max_layers`, `speaker_uncond_mode`, `t_schedule_mode`, `sway_coeff`.
- Each card shows the seed its voice was synthesized with. Click it to make it the fixed seed, so a voice you liked while on random can be used again.

### Changed

- `tts.num_steps` and `tts.sampling.num_steps` default to the model's own step count (4 for MeanFlow, 40 for RF) when unset.

## [0.1.0] - 2026-10-10

### Added

- `sttts-say`: a command-line tool that synthesizes lines with Irodori-TTS without the GUI, for agents (Agent Skills) and video workflows.
  - Voices (`data/voices/NAME.wav` + `NAME.json` with caption / seed / sampling), takes with a reproducible record (`--like`), seed auditions, and promotion of a take to a reference voice (`voice save`).
  - Incremental rendering of JSONL scripts, with per-chunk timings for subtitles.
  - Agent Skill in `skills/sttts-say/SKILL.md`.
- Only one process uses the GPU at a time (sttts-gui and sttts-say exclude each other with a lock file).
- Release builds on GitHub Releases: `sttts-gui` (GUI + CLI) and a CLI-only `sttts-say` archive for Windows x64, with SHA-256 checksums.
- UI in English, Japanese and Simplified Chinese. It follows the Windows display language and can be switched in ⚙ Settings; READMEs in all three languages.
- `HF_HUB_OFFLINE=1` disables model downloads (only models already in the cache are used).
- Windows version information (product name `sttts-gui`, version from `Cargo.toml`) in both executables.
- Privacy policy (`PRIVACY.md`) and code signing policy (README).
- Experimental `sttts-say` builds for macOS (Apple Silicon) and Linux x64. They are built and unit-tested in CI but not yet verified on real hardware. Data goes to `~/Library/Application Support/sttts-gui` (macOS) or `~/.local/share/sttts-gui` / `$XDG_DATA_HOME/sttts-gui` (Linux).
- CI on Windows, macOS and Linux.

### Changed

- `sttts-engine`: the mic, VAD and ASR are behind the `live` feature (on by default); `sttts-say` is built without them.
- After you finish an utterance and start the next one, the previous card shows "Transcribing…" instead of a second "Listening…".

### Fixed

- An utterance too short to be recognized no longer leaves a card stuck in "Listening…"; it is removed.
- Gemini: when the server sends no final transcript or completion signal after an utterance, the text received so far is used 2.5 s later instead of waiting 20 s and failing with "transcription timed out". The following utterances are no longer held up behind it.
- When reading aloud starts while you are still talking (long utterances), the card's text keeps growing with the recognition instead of freezing at the point reading started.
- Reading aloud while you are still talking no longer stops for the rest of a long utterance when the recognizer revises text it has already read (e.g. Gemini adding spaces); it carries on from the same position.
- Expression: "pausing as it goes" (間を取りながら) is no longer added to almost every utterance. Only gaps of 320 ms or more between speech count as pauses (ordinary breaks between phrases and the silence before/after the utterance no longer do), and only longer utterances (1.5 s or more of speech) get it.

[Unreleased]: https://github.com/kjranyone/sttts-gui/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kjranyone/sttts-gui/releases/tag/v0.1.0
