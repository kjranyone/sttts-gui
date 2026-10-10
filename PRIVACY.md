# Privacy policy

sttts-gui and sttts-say run on your computer. They have no telemetry, analytics, crash reporting or update checks,
and the maintainers receive no data from them.

The programs connect to other networked systems only in the cases below.

## 1. Model downloads (Hugging Face)

When a model is not yet on your computer (typically the first run), the programs download it from Hugging Face
(`https://huggingface.co`, or `HF_ENDPOINT` if set) and store it in the Hugging Face cache on your computer.
These are ordinary file downloads: no audio, text or settings are sent. As with any download, Hugging Face receives
your IP address and the names of the requested files.

- Models: Irodori-TTS (speech synthesis) and, in sttts-gui, the speech recognition model you choose (kotoba-whisper or Nemotron).
- To prevent downloads, set the environment variable `HF_HUB_OFFLINE=1`. The programs then use only models already in the cache, and stop with an error if one is missing.
- Hugging Face privacy policy: https://huggingface.co/privacy

## 2. Cloud speech recognition (Google Gemini) — sttts-gui only, opt-in

Speech recognition is local by default. Only if you select "Cloud (Gemini)" as the recognition provider and enter your
own Gemini API key, sttts-gui sends the audio of each utterance and your API key to the Google Gemini Live API
(`generativelanguage.googleapis.com`) to transcribe it. The app shows a notice when you switch to it.
Switch back to a local provider to stop sending audio.

- Your API key is stored on your computer, encrypted with Windows DPAPI (`data/config.json`).
- Gemini API terms: https://ai.google.dev/gemini-api/terms
- Google privacy policy: https://policies.google.com/privacy

sttts-say never uses Gemini and never opens the microphone.

## Data stored on your computer

Settings (`data/`), your voice bank (`data/voices/`) and generated audio (`output/`) stay in the app folder
(see "Distributing the exe" in the README for its location). Delete the folder to remove them.
