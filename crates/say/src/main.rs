//! sttts-say — GUI を開かずに Irodori-TTS でセリフを WAV にする CLI(エージェント・動画制作向け)。
//!
//! 本体は `sttts_engine::say`。ここは引数の解釈と、結果 JSON の標準出力だけを持つ。
//! 進捗とエラーは標準エラーへ(表示言語は GUI と同じ規則)。マイクは開かない。GPU は
//! GUI と同時に使わない(プロセス間ロック。GUI が起動中ならエラーで止まる)。
//!
//! 合成(`speak` / `render` / `audition`)は常駐プロセス(`sttts_engine::say_server`)に頼む。常駐プロセスは
//! モデルを読み込んだまま待つので、2 回目以降の呼び出しはモデルの読み込みとカーネルのコンパイルを省ける。
//! 最初の呼び出しで自動的に起動し、10 分使われなければ終わる。使えないときはこのプロセスで合成する。

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sttts_engine::config::{default_config, default_user_config_path, load_user_config, merge_config};
use sttts_engine::say::{Line, Studio, line_like, list_models, list_voices, load_voice, read_take, use_model};
use sttts_engine::config::get;
use sttts_i18n::trf;
use sttts_engine::say_server::{self, EngineCache, RunRequest, ServerPaths};
use sttts_engine::tts::{IrodoriTts, TtsEngine};

const USAGE: &str = "\
sttts-say — Irodori-TTS without the GUI (voices, takes, incremental scripts)

USAGE
  sttts-say speak    --text T [--voice V] [--style S] [--caption C] [--seed N|random]
                     [--sampling JSON] [--like TAKE] [--out FILE.wav]
  sttts-say render   SCRIPT.jsonl [--out-dir DIR]
  sttts-say audition --text T [--voice V] [--style S] [--caption C] [--sampling JSON]
                     [--count N] [--out-dir DIR]
  sttts-say voice list
  sttts-say voice show NAME
  sttts-say voice save NAME --from TAKE
  sttts-say model list
  sttts-say model use NAME
  sttts-say stop

  --voice    a voice in data/voices (reference audio NAME.wav and/or settings NAME.json)
  --style    how this line is delivered; appended to the voice's caption
  --caption  replaces the voice's caption (timbre / character description)
  --like     reuse a take's settings and seed (TAKE = its .wav or .json); other options override
  --seed random  draw a new seed (e.g. with --like for a retake)
  model use  records the model in data/backend.json (tts.model); later runs use it and
             re-render changed lines. v4.1-small (RF) is slower but more accurate than
             the default v4.1-small-mf (MeanFlow). Weights download on first use.

Results are JSON on stdout; progress and errors go to stderr.
Settings come from data/backend.json (tts.sampling etc.), the same as the GUI.

speak / render / audition run in a background sttts-say process that keeps the model
loaded, so later calls skip loading it. It starts by itself, exits after 10 minutes
without requests, and gives the GPU to sttts-gui when the GUI starts while it is idle.
`stop` ends it now.
";

#[derive(Debug, PartialEq, Serialize, Deserialize)]
enum Command {
    Help,
    Speak { line: Line, like: Option<PathBuf>, reseed: bool, out: Option<PathBuf> },
    Render { script: PathBuf, out_dir: Option<PathBuf> },
    Audition { line: Line, count: usize, out_dir: Option<PathBuf> },
    VoiceList,
    VoiceShow { name: String },
    VoiceSave { name: String, from: PathBuf },
    ModelList,
    ModelUse { name: String },
    Stop,
    /// 常駐プロセスとして依頼を待つ(`speak` などが自動で起動する。利用者が直接使うものではない)
    Serve,
}

fn parse(args: &[String]) -> Result<Command> {
    let Some((cmd, rest)) = args.split_first() else { return Ok(Command::Help) };
    let mut it = rest.iter();
    let mut line = Line::default();
    let (mut like, mut reseed, mut out, mut out_dir, mut from) = (None, false, None, None, None);
    let mut count = 4usize;
    let mut positional: Vec<String> = Vec::new();
    let mut text = None;
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--text" => text = Some(val()?),
            "--voice" => line.voice = Some(val()?),
            "--style" => line.style = Some(val()?),
            "--caption" => line.caption = Some(val()?),
            "--seed" => match val()?.as_str() {
                "random" => reseed = true,
                s => line.seed = Some(s.parse().with_context(|| format!("--seed {s}"))?),
            },
            "--sampling" => {
                let v: Value = serde_json::from_str(&val()?).context("--sampling")?;
                let Value::Object(m) = v else { bail!("--sampling must be a JSON object") };
                line.sampling = m;
            }
            "--like" => like = Some(PathBuf::from(val()?)),
            "--out" => out = Some(PathBuf::from(val()?)),
            "--out-dir" => out_dir = Some(PathBuf::from(val()?)),
            "--count" => count = val()?.parse().context("--count")?,
            "--from" => from = Some(PathBuf::from(val()?)),
            "-h" | "--help" => return Ok(Command::Help),
            s if s.starts_with("--") => bail!("unknown option {s}"),
            s => positional.push(s.to_string()),
        }
    }
    let no_positional = |positional: &[String]| match positional.first() {
        Some(p) => bail!("unexpected argument {p}"),
        None => Ok(()),
    };
    let cmd = match cmd.as_str() {
        "help" | "-h" | "--help" => Command::Help,
        "speak" => {
            no_positional(&positional)?;
            if text.is_none() && like.is_none() {
                bail!("speak needs --text (or --like TAKE)");
            }
            line.text = text.unwrap_or_default();
            Command::Speak { line, like, reseed, out }
        }
        "render" => {
            let [script] = positional.as_slice() else { bail!("render needs exactly one SCRIPT.jsonl") };
            Command::Render { script: PathBuf::from(script), out_dir }
        }
        "audition" => {
            no_positional(&positional)?;
            line.text = text.context("audition needs --text")?;
            if line.seed.is_some() || reseed {
                bail!("audition picks the seeds itself; drop --seed");
            }
            Command::Audition { line, count: count.max(1), out_dir }
        }
        "voice" => match positional.as_slice() {
            [sub] if sub == "list" => Command::VoiceList,
            [sub, name] if sub == "show" => Command::VoiceShow { name: name.clone() },
            [sub, name] if sub == "save" => Command::VoiceSave { name: name.clone(), from: from.context("voice save needs --from TAKE")? },
            _ => bail!("voice needs list | show NAME | save NAME --from TAKE"),
        },
        "model" => match positional.as_slice() {
            [sub] if sub == "list" => Command::ModelList,
            [sub, name] if sub == "use" => Command::ModelUse { name: name.clone() },
            _ => bail!("model needs list | use NAME"),
        },
        "stop" => {
            no_positional(&positional)?;
            Command::Stop
        }
        "serve" => Command::Serve,
        other => bail!("unknown command {other} (see sttts-say help)"),
    };
    Ok(cmd)
}

/// `--like` のテイクに、明示した指定を重ねる
fn merge_like(base: Line, given: Line, reseed: bool) -> Line {
    let mut sampling: Map<String, Value> = base.sampling;
    sampling.extend(given.sampling);
    Line {
        id: None,
        text: if given.text.is_empty() { base.text } else { given.text },
        voice: given.voice.or(base.voice),
        style: given.style.or(base.style),
        caption: given.caption.or(base.caption),
        seed: if reseed { None } else { given.seed.or(base.seed) },
        sampling,
    }
}

/// 読み込んだ TTS の指定(モデルとステップ数)。常駐プロセスはこれが変わったら読み込み直す
fn engine_key(cfg: &Value) -> (String, Option<usize>) {
    let model = get(cfg, "tts", "model").as_str().unwrap_or(sttts_engine::tts::DEFAULT_TTS_MODEL).to_string();
    let steps = get(cfg, "tts", "num_steps").as_u64().map(|n| n as usize);
    (model, steps)
}

/// 合成のコマンドを実行する(このプロセスでも常駐プロセスでも同じ)。TTS は `cache` から使い回す
fn synthesize(cmd: Command, root: PathBuf, cfg: Value, cache: Rc<RefCell<EngineCache>>, log: Rc<dyn Fn(&str)>) -> Result<Value> {
    let loader_cfg = cfg.clone();
    let loader_log = Rc::clone(&log);
    let studio = Studio::new(
        root,
        cfg,
        move || {
            let (model, steps) = engine_key(&loader_cfg);
            let progress = |m: &str| loader_log(m);
            cache.borrow_mut().get_or_load(&format!("{model}/{steps:?}"), || {
                Ok(Arc::new(IrodoriTts::load(&model, steps, &progress)?) as Arc<dyn TtsEngine>)
            })
        },
        move |m| log(m),
    );
    match cmd {
        Command::Speak { line, like, reseed, out } => {
            let line = match like {
                Some(take) => merge_like(line_like(&read_take(&take)?), line, reseed),
                None => line,
            };
            studio.speak(&line, out.as_deref())
        }
        Command::Render { script, out_dir } => studio.render(&script, out_dir.as_deref()),
        Command::Audition { line, count, out_dir } => studio.audition(&line, count, out_dir.as_deref()),
        other => bail!("not a synthesis command: {other:?}"),
    }
}

/// 常駐プロセス: 依頼を順番に処理する。モデルは依頼をまたいで使い回す
fn serve() -> Result<()> {
    let cache = Rc::new(RefCell::new(EngineCache::default()));
    say_server::serve(&ServerPaths::default(), say_server::IDLE_EXIT, |req, log| {
        let cmd: Command = serde_json::from_value(req.command.clone())?;
        synthesize(cmd, req.root.clone(), req.cfg.clone(), Rc::clone(&cache), log)
    })
}

fn run(cmd: Command) -> Result<Option<Value>> {
    let root = sttts_engine::root::app_root();
    let warn = |m: &str| eprintln!("[warn] {m}");
    let cfg = merge_config(&default_config(), &load_user_config(&default_user_config_path(&root), &warn));
    if !matches!(cmd, Command::Help | Command::Serve | Command::Stop) {
        install_presets(&root, &warn);
    }
    let value = match cmd {
        Command::Help => {
            print!("{USAGE}");
            return Ok(None);
        }
        Command::Serve => {
            serve()?;
            return Ok(None);
        }
        Command::Stop => json!({ "stopped": say_server::stop(&ServerPaths::default())? }),
        Command::VoiceList => serde_json::to_value(list_voices(&root)?)?,
        Command::VoiceShow { name } => serde_json::to_value(load_voice(&root, &name)?)?,
        Command::VoiceSave { name, from } => {
            let studio = Studio::new(root.clone(), cfg, || bail!("voice save does not synthesize"), |m| eprintln!("{m}"));
            studio.save_voice(&name, &from)?
        }
        Command::ModelList => Value::from(list_models(&cfg)),
        Command::ModelUse { name } => use_model(&root, &cfg, &name)?,
        synth => {
            let req = RunRequest {
                cwd: std::env::current_dir()?,
                root: root.clone(),
                cfg: cfg.clone(),
                lang: sttts_i18n::lang().code().to_string(),
                command: serde_json::to_value(&synth)?,
            };
            let spawn = || say_server::spawn_detached(&std::env::current_exe()?, &["serve"]);
            match say_server::run_remote(&ServerPaths::default(), &req, spawn, &|m| eprintln!("{m}"))? {
                Some(v) => v,
                // 常駐プロセスを使えないときは、このプロセスで合成する
                None => synthesize(synth, root, cfg, Rc::new(RefCell::new(EngineCache::default())), Rc::new(|m: &str| eprintln!("{m}")))?,
            }
        }
    };
    Ok(Some(value))
}

/// 同梱の声プリセットのうち未導入のものを `data/voices` へ書き出す(GUI の起動時と同じ。消したものは戻さない)。
/// 書き出せなくても声を使わないコマンドは動くので、警告にとどめる
fn install_presets(root: &Path, warn: &dyn Fn(&str)) {
    match sttts_engine::presets::install_presets(root) {
        Ok(names) if !names.is_empty() => {
            let names = names.join(", ");
            eprintln!(
                "{}",
                trf!("Added the bundled voices: {names}", "同梱の声を追加しました: {names}", "已添加内置声音:{names}")
            );
        }
        Ok(_) => {}
        Err(e) => {
            let e = format!("{e:#}");
            warn(&trf!(
            "Could not write the bundled voices: {e}",
            "同梱の声を書き出せませんでした: {e}",
            "无法写入内置声音:{e}"
            ))
        }
    }
}

fn main() {
    let root = sttts_engine::root::app_root();
    let saved = std::fs::read_to_string(root.join("data").join("config.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .and_then(|v| v.get("language").and_then(Value::as_str).map(str::to_string));
    sttts_i18n::set_lang(sttts_i18n::initial(saved.as_deref()));

    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = parse(&args).and_then(run);
    match result {
        Ok(Some(v)) => println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default()),
        Ok(None) => {}
        Err(e) => {
            eprintln!("[error] {e:#}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &[&str]) -> Vec<String> {
        s.iter().map(|a| a.to_string()).collect()
    }

    #[test]
    fn speak_collects_the_line() {
        let c = parse(&args(&["speak", "--text", "やあ", "--voice", "aoi", "--style", "眠そうに", "--seed", "9", "--sampling", "{\"duration_scale\":1.2}", "--out", "a.wav"])).unwrap();
        let Command::Speak { line, like, reseed, out } = c else { panic!() };
        assert_eq!((line.text.as_str(), line.voice.as_deref(), line.style.as_deref(), line.seed), ("やあ", Some("aoi"), Some("眠そうに"), Some(9)));
        assert_eq!(line.sampling["duration_scale"], 1.2);
        assert_eq!((like, reseed, out), (None, false, Some(PathBuf::from("a.wav"))));
    }

    #[test]
    fn mistakes_are_errors_not_ignored() {
        assert!(parse(&args(&["speak"])).is_err());
        assert!(parse(&args(&["speak", "--text", "a", "--sede", "1"])).is_err());
        assert!(parse(&args(&["speak", "--text", "a", "--sampling", "[1]"])).is_err());
        assert!(parse(&args(&["audition", "--text", "a", "--seed", "1"])).is_err());
        assert!(parse(&args(&["render"])).is_err());
        assert!(parse(&args(&["voice", "save", "x"])).is_err());
        assert!(parse(&args(&["sing"])).is_err());
        assert!(parse(&args(&["model", "use"])).is_err());
        assert_eq!(parse(&args(&[])).unwrap(), Command::Help);
    }

    #[test]
    fn like_keeps_the_take_and_applies_overrides() {
        let base = Line { text: "元のセリフ".into(), voice: Some("aoi".into()), style: Some("明るく".into()), seed: Some(5), ..Default::default() };
        // 何も変えなければ同じテイク
        assert_eq!(merge_like(base.clone(), Line::default(), false), base);
        // 話し方だけ変える(seed は保つ)
        let l = merge_like(base.clone(), Line { style: Some("悲しげに".into()), ..Default::default() }, false);
        assert_eq!((l.style.as_deref(), l.seed, l.text.as_str()), (Some("悲しげに"), Some(5), "元のセリフ"));
        // 撮り直し(seed を引き直す)
        assert_eq!(merge_like(base, Line::default(), true).seed, None);
    }

    #[test]
    fn subcommands_parse() {
        assert_eq!(parse(&args(&["render", "s.jsonl", "--out-dir", "o"])).unwrap(), Command::Render { script: "s.jsonl".into(), out_dir: Some("o".into()) });
        assert_eq!(parse(&args(&["voice", "list"])).unwrap(), Command::VoiceList);
        assert_eq!(parse(&args(&["model", "list"])).unwrap(), Command::ModelList);
        assert_eq!(parse(&args(&["stop"])).unwrap(), Command::Stop);
        assert_eq!(parse(&args(&["model", "use", "v4.1-small"])).unwrap(), Command::ModelUse { name: "v4.1-small".into() });
        assert_eq!(parse(&args(&["voice", "save", "mio", "--from", "t.wav"])).unwrap(), Command::VoiceSave { name: "mio".into(), from: "t.wav".into() });
        let Command::Audition { count, .. } = parse(&args(&["audition", "--text", "a", "--count", "6"])).unwrap() else { panic!() };
        assert_eq!(count, 6);
    }
}

#[cfg(test)]
mod transport_tests {
    use super::*;

    /// 常駐プロセスへはコマンドを JSON で渡す。往復しても同じコマンドに戻る
    #[test]
    fn commands_survive_the_trip_to_the_background_process() {
        let args: Vec<String> = ["speak", "--text", "やあ", "--voice", "aoi", "--seed", "3", "--sampling", "{\"duration_scale\":1.1}", "--out", "o.wav"]
            .iter()
            .map(|a| a.to_string())
            .collect();
        let cmd = parse(&args).unwrap();
        let back: Command = serde_json::from_value(serde_json::to_value(&cmd).unwrap()).unwrap();
        assert_eq!(back, cmd);
        let render = Command::Render { script: "ep.jsonl".into(), out_dir: None };
        assert_eq!(serde_json::from_value::<Command>(serde_json::to_value(&render).unwrap()).unwrap(), render);
    }
}
