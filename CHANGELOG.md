# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/kjranyone/sttts-gui/commits/main
