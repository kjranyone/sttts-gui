//! 設定の既定値とマージ。
//!
//! 設定の永続化は GUI 側が担い、エンジンは `configure` で受ける。GUI に UI が無い上級設定
//! (VAD・投機的 TTS 等)は、任意の JSON ファイル
//! `<root>/data/backend.json`(環境変数 `STTTS_CONFIG` で変更可)に書くと起動時に既定値へ
//! マージされる。GUI からの configure はその上に適用される。`tts.sampling` は GUI の
//! 「合成パラメータ」欄からも編集でき、このファイルへ書き戻される([`set_user_config_value`])。
//!
//! 設定は JSON のまま持つ(セクション単位の深いマージ)。Irodori の項目名は Irodori と同じで、
//! GUI が未対応の項目も `tts.sampling` から必ず指定できる。

use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

pub const SECTIONS: [&str; 5] = ["tts", "asr", "audio", "voice", "pipeline"];

pub fn default_config() -> Value {
    json!({
        "tts": {
            "model": "v4.1-small-mf",
            // None で checkpoint 既定(MeanFlow: 4)
            "num_steps": null,
            // 起動後(初回 configure 時)にモデルをロードして短文を数回合成しておく
            "warmup": true,
            // Irodori の SamplingRequest 項目を丸ごと上書きできる(duration_scale / seconds /
            // max_ref_seconds / trim_tail ...)。GUI が対応していない Irodori の機能もここから使える。
            // text/caption/ref_*/seed/no_ref は発話ごとにアプリが決める。
            "sampling": {},
        },
        "asr": {
            // "kotoba"(Whisper) | "nemotron"(ONNX) | "gemini"(Live API) | "mock"
            "engine": "kotoba",
            "model": "kotoba-tech/kotoba-whisper-v2.0",
            "final_beam_size": 2,
            // 0 で partial(途中経過表示)を無効化
            "partial_interval_ms": 800,
            "language": "ja",
            // 起動時に ASR をロードして「マイク開始」を即座に使えるようにする
            "preload": true,
            // silero VAD: 無音がこの長さ続いたら発話終了。短いほど速いが文中の間で切れやすい
            "vad_min_silence_ms": 280,
            "vad_threshold": 0.5,
            // Nemotron 3.5 ASR(asr.engine = "nemotron")用
            "nemotron_repo": "codavidgarcia/nemotron-3.5-asr-streaming-0.6b-onnx",
            "nemotron_model_dir": null,
            "nemotron_chunk_ms": 320,
            "nemotron_precision": "fp16",
            "nemotron_threads": 4,
            // Gemini Live API(asr.engine = "gemini")用。APIキーは環境変数 GEMINI_API_KEY でも可
            "gemini_model": "gemini-3.5-transcribe-live",
            "gemini_api_key": null,
            // VERBATIM: 話し方を保持。SMART: フィラー除去・句読点整形
            "gemini_mode": "VERBATIM",
            "gemini_timeout_s": 20.0,
        },
        "audio": { "input_device_index": null },
        "voice": { "caption": null, "ref_wavs": [], "no_ref": true },
        "pipeline": {
            "auto_speak": true,
            // 2チャンク目以降の最小文字数
            "chunk_min_chars": 16,
            // これを超える塊は読点/文節境界で分割(句読点なし ASR 出力対策)
            "chunk_max_chars": 80,
            "first_chunk_min_chars": 1,
            // 先頭チャンクを読点か約 8〜12 モーラで切って初音を早める(max=0 で無効)
            "first_chunk_mora_min": 8,
            "first_chunk_mora_max": 12,
            // 投機的 TTS(既定 OFF): 同じ先頭チャンクが N 回連続した partial から先行合成し、
            // 確定文の先頭チャンクと完全一致した場合だけ再生に回す(不一致なら破棄)
            "speculative_tts": false,
            "speculative_stable_partials": 2,
            // 元音声の表現を発話単位で Irodori へ渡す。ASR の文字列は書き換えない。
            "performance_enabled": true,
            "emotion_engine": "none",
            // ASR 確定後、並行解析の完了を待つ上限。超過時は明瞭読み上げ。
            "performance_wait_ms": 150,
        },
    })
}

/// セクション単位の深いマージ。`patch` を破壊しない。
pub fn merge_config(base: &Value, patch: &Value) -> Value {
    match (base, patch) {
        (Value::Object(b), Value::Object(p)) => {
            let mut out: Map<String, Value> = b.clone();
            for (k, v) in p {
                let merged = match out.get(k) {
                    Some(cur) if cur.is_object() && v.is_object() => merge_config(cur, v),
                    _ => v.clone(),
                };
                out.insert(k.clone(), merged);
            }
            Value::Object(out)
        }
        _ => patch.clone(),
    }
}

pub fn default_user_config_path(root: &Path) -> PathBuf {
    match std::env::var_os("STTTS_CONFIG") {
        Some(p) if !p.is_empty() => PathBuf::from(p),
        _ => root.join("data").join("backend.json"),
    }
}

/// ユーザー設定 JSON を読む。無ければ `{}`。未知のセクションは無視して警告する。
pub fn load_user_config(path: &Path, warn: &dyn Fn(&str)) -> Value {
    if !path.is_file() {
        return json!({});
    }
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            warn(&format!("設定ファイルを読めません(無視します): {}: {e}", path.display()));
            return json!({});
        }
    };
    let data: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            warn(&format!("設定ファイルを読めません(無視します): {}: {e}", path.display()));
            return json!({});
        }
    };
    let Value::Object(obj) = data else {
        warn(&format!("設定ファイルの形式が不正です(オブジェクトではない): {}", path.display()));
        return json!({});
    };
    let unknown: Vec<&String> = obj.keys().filter(|k| !SECTIONS.contains(&k.as_str())).collect();
    if !unknown.is_empty() {
        warn(&format!("設定ファイルの未知のセクションを無視: {unknown:?}"));
    }
    Value::Object(obj.into_iter().filter(|(k, v)| SECTIONS.contains(&k.as_str()) && v.is_object()).collect())
}

/// ユーザー設定ファイルの `section.key` だけを書き換える(他の内容はそのまま残す)。
/// GUI から編集した上級設定(`tts.sampling` 等)を、手で書いた設定と同じ場所に記録するため。
/// 読めない・壊れたファイルは上書きせずエラーにする(手書きの設定を消さない)。
pub fn set_user_config_value(path: &Path, section: &str, key: &str, value: Value) -> anyhow::Result<()> {
    use anyhow::Context as _;
    let mut root = if path.is_file() {
        let text = std::fs::read_to_string(path).with_context(|| format!("{} を読めません", path.display()))?;
        serde_json::from_str::<Value>(&text).with_context(|| format!("{} の JSON が不正です", path.display()))?
    } else {
        json!({})
    };
    let Value::Object(obj) = &mut root else {
        anyhow::bail!("{} の形式が不正です(オブジェクトではない)", path.display());
    };
    let sec = obj.entry(section).or_insert_with(|| json!({}));
    if !sec.is_object() {
        *sec = json!({});
    }
    sec[key] = value;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(&root)? + "
").with_context(|| format!("{} に書けません", path.display()))?;
    Ok(())
}

/// `cfg[section][key]` を取り出す(無ければ Null)。
pub fn get<'a>(cfg: &'a Value, section: &str, key: &str) -> &'a Value {
    cfg.get(section).and_then(|s| s.get(key)).unwrap_or(&Value::Null)
}

pub fn get_bool(cfg: &Value, section: &str, key: &str, default: bool) -> bool {
    get(cfg, section, key).as_bool().unwrap_or(default)
}

pub fn get_i64(cfg: &Value, section: &str, key: &str, default: i64) -> i64 {
    get(cfg, section, key).as_i64().unwrap_or(default)
}

pub fn get_f64(cfg: &Value, section: &str, key: &str, default: f64) -> f64 {
    get(cfg, section, key).as_f64().unwrap_or(default)
}

pub fn get_str<'a>(cfg: &'a Value, section: &str, key: &str) -> Option<&'a str> {
    get(cfg, section, key).as_str()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_shape() {
        let cfg = default_config();
        let mut keys: Vec<&String> = cfg.as_object().unwrap().keys().collect();
        keys.sort();
        assert_eq!(keys, ["asr", "audio", "pipeline", "tts", "voice"]);
        assert_eq!(cfg["tts"]["model"], "v4.1-small-mf");
        assert_eq!(cfg["pipeline"]["auto_speak"], true);
        assert_eq!(cfg["asr"]["engine"], "kotoba"); // 既定エンジンは変えない
        assert_eq!(cfg["tts"]["sampling"], json!({}));
    }

    #[test]
    fn merge_partial_section() {
        let defaults = default_config();
        let cfg = merge_config(&defaults, &json!({"tts": {"model": "other"}}));
        assert_eq!(cfg["tts"]["model"], "other");
        assert_eq!(cfg["tts"]["warmup"], true); // 他キーは保持
        assert_eq!(cfg["asr"]["model"], defaults["asr"]["model"]);
    }

    #[test]
    fn merge_does_not_mutate() {
        let base = default_config();
        let _ = merge_config(&base, &json!({"voice": {"caption": "落ち着いた声"}}));
        assert!(base["voice"]["caption"].is_null());
    }

    #[test]
    fn sampling_merges_per_key() {
        let cfg = merge_config(&default_config(), &json!({"tts": {"sampling": {"duration_scale": 1.1, "seconds": 3.0}}}));
        let cfg = merge_config(&cfg, &json!({"tts": {"sampling": {"duration_scale": 1.2}}}));
        assert_eq!(cfg["tts"]["sampling"], json!({"duration_scale": 1.2, "seconds": 3.0}));
    }

    #[test]
    fn set_user_config_value_keeps_other_settings() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("data").join("backend.json");
        set_user_config_value(&p, "tts", "sampling", json!({"trim_tail": false})).unwrap();
        assert_eq!(load_user_config(&p, &|_| {}), json!({"tts": {"sampling": {"trim_tail": false}}}));

        std::fs::write(&p, r#"{"asr": {"engine": "nemotron"}, "tts": {"warmup": false, "sampling": {"seconds": 3.0}}}"#).unwrap();
        set_user_config_value(&p, "tts", "sampling", json!({"duration_scale": 1.2})).unwrap();
        let cfg = load_user_config(&p, &|_| {});
        assert_eq!(cfg["asr"]["engine"], "nemotron");
        assert_eq!(cfg["tts"]["warmup"], false);
        assert_eq!(cfg["tts"]["sampling"], json!({"duration_scale": 1.2}));

        // 壊れたファイルは上書きしない
        std::fs::write(&p, "{ broken").unwrap();
        assert!(set_user_config_value(&p, "tts", "sampling", json!({})).is_err());
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "{ broken");
    }

    #[test]
    fn user_config_file_is_merged() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("backend.json");
        std::fs::write(&p, r#"{"asr": {"engine": "nemotron"}, "pipeline": {"speculative_tts": true}, "bogus": {}}"#).unwrap();
        let user = load_user_config(&p, &|_| {});
        let mut keys: Vec<&String> = user.as_object().unwrap().keys().collect();
        keys.sort();
        assert_eq!(keys, ["asr", "pipeline"]);
        let cfg = merge_config(&default_config(), &user);
        assert_eq!(cfg["asr"]["engine"], "nemotron");
        assert_eq!(cfg["asr"]["vad_min_silence_ms"], 280); // 他キーは既定のまま
        assert_eq!(cfg["pipeline"]["speculative_tts"], true);
    }

    #[test]
    fn missing_or_broken_user_config_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(load_user_config(&dir.path().join("none.json"), &|_| {}), json!({}));
        let bad = dir.path().join("bad.json");
        std::fs::write(&bad, "{not json").unwrap();
        assert_eq!(load_user_config(&bad, &|_| {}), json!({}));
    }
}
