//! 合成パラメータ(Irodori の `tts.sampling`)の編集。
//!
//! 項目の一覧と既定値はエンジンの [`sampling_fields`] から作る(Irodori 側の項目が増えたら
//! そこへ足せば GUI にも出る)。値は data/backend.json の `tts.sampling` に記録し、手で書いた
//! 設定と同じ場所を唯一の置き場にする。GUI が知らない項目はそのまま残す(塞がない)。
//! 空欄・既定値は「上書きしない」= 項目を消す。

use gpui_kit::component::button::*;
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::switch::Switch;
use gpui_kit::component::*;
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::*;
use serde_json::{Map, Value};
use sttts_engine::config;
use sttts_engine::tts::{SamplingField, SamplingKind, sampling_fields};
use sttts_protocol::{GuiMessage, TtsConfig};

use super::{StttsApp, kit};
use crate::theme::{self, c};

/// 数値欄の入力が止まってから適用するまでの待ち時間
const APPLY_DELAY: std::time::Duration = std::time::Duration::from_millis(700);

/// 合成パラメータ欄の状態
pub(super) struct SamplingEditor {
    fields: Vec<SamplingField>,
    /// いま有効な上書き(= backend.json の tts.sampling)
    values: Map<String, Value>,
    /// 数値項目の入力欄(fields と同じ順。真偽項目は None)
    inputs: Vec<Option<Entity<InputState>>>,
    pub(super) open: bool,
    error: Option<String>,
    edit_seq: u64,
}

impl SamplingEditor {
    /// backend.json の tts.sampling を読み、入力欄を作る。
    pub(super) fn new(root: &std::path::Path, window: &mut Window, cx: &mut Context<StttsApp>) -> Self {
        let user = config::load_user_config(&config::default_user_config_path(root), &|_| {});
        let values = config::get(&user, "tts", "sampling").as_object().cloned().unwrap_or_default();
        let fields = sampling_fields();
        let inputs = fields
            .iter()
            .map(|f| {
                (f.kind != SamplingKind::Bool).then(|| {
                    cx.new(|cx| {
                        let mut state = InputState::new(window, cx).placeholder(format!("既定 {}", show(&f.default)));
                        if let Some(v) = values.get(f.key).filter(|v| v.is_number()) {
                            state.set_value(show(v), window, cx);
                        }
                        state
                    })
                })
            })
            .collect();
        Self { fields, values, inputs, open: false, error: None, edit_seq: 0 }
    }

    /// 既定から変えている項目数(GUI が知らない項目も含む)
    fn changed_count(&self) -> usize {
        self.values.len()
    }

    /// GUI に欄の無い項目(backend.json に手で書かれたもの)
    fn unknown_keys(&self) -> Vec<&str> {
        self.values.keys().map(String::as_str).filter(|k| !self.fields.iter().any(|f| f.key == *k)).collect()
    }
}

/// 値の表示(null は「なし」)
fn show(v: &Value) -> String {
    match v {
        Value::Null => "なし".into(),
        other => other.to_string(),
    }
}

/// 入力欄の文字列を項目の値へ。空欄は None(上書きしない)。
fn parse(field: &SamplingField, text: &str) -> Result<Option<Value>, String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }
    match field.kind {
        SamplingKind::Int => text
            .parse::<u64>()
            .map(|n| Some(Value::from(n)))
            .map_err(|_| format!("{}: 0 以上の整数で入力してください", field.label)),
        SamplingKind::Float => match text.parse::<f64>() {
            Ok(x) if x.is_finite() => Ok(Some(Value::from(x))),
            _ => Err(format!("{}: 数値で入力してください", field.label)),
        },
        SamplingKind::Bool => Ok(None),
    }
}

impl StttsApp {
    /// 数値欄の購読(Enter / フォーカスアウトで即時、入力が止まったら少し待って適用)。
    pub(super) fn subscribe_sampling_inputs(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let inputs: Vec<Entity<InputState>> = self.sampling.inputs.iter().flatten().cloned().collect();
        for input in inputs {
            let weak = cx.weak_entity();
            let sub = window.subscribe(&input, cx, move |_, event: &InputEvent, _window, cx| match event {
                InputEvent::PressEnter { .. } | InputEvent::Blur => {
                    let _ = weak.update(cx, |app, cx| app.apply_sampling_inputs(cx));
                }
                InputEvent::Change => {
                    let _ = weak.update(cx, |app, cx| app.schedule_sampling_apply(cx));
                }
                _ => {}
            });
            self.subscriptions.push(sub);
        }
    }

    fn schedule_sampling_apply(&mut self, cx: &mut Context<Self>) {
        self.sampling.edit_seq += 1;
        let seq = self.sampling.edit_seq;
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(APPLY_DELAY).await;
            let _ = this.update(cx, |app, cx| {
                if app.sampling.edit_seq == seq {
                    app.apply_sampling_inputs(cx);
                }
            });
        })
        .detach();
    }

    /// 数値欄の内容を上書きの組へ反映する。不正な値があれば何も変えずに知らせる。
    fn apply_sampling_inputs(&mut self, cx: &mut Context<Self>) {
        let mut next = self.sampling.values.clone();
        let mut errors = Vec::new();
        for (field, input) in self.sampling.fields.iter().zip(&self.sampling.inputs) {
            let Some(input) = input else { continue };
            // 「なし」を明示している項目(正規化しない 等)は欄を出していない
            if field.null_label.is_some() && next.get(field.key).is_some_and(Value::is_null) {
                continue;
            }
            match parse(field, &input.read(cx).value()) {
                Ok(Some(v)) => {
                    next.insert(field.key.into(), v);
                }
                Ok(None) => {
                    next.remove(field.key);
                }
                Err(e) => errors.push(e),
            }
        }
        if errors.is_empty() {
            self.sampling.error = None;
            self.commit_sampling(next, cx);
        } else {
            self.sampling.error = Some(errors.join("\n"));
            cx.notify();
        }
    }

    fn set_sampling_bool(&mut self, key: &'static str, on: bool, cx: &mut Context<Self>) {
        let Some(field) = self.sampling.fields.iter().find(|f| f.key == key) else { return };
        let mut next = self.sampling.values.clone();
        if field.default.as_bool() == Some(on) {
            next.remove(key);
        } else {
            next.insert(key.into(), Value::Bool(on));
        }
        self.commit_sampling(next, cx);
    }

    /// null の明示(例: 参照音声を正規化しない)の切替。外したら数値欄の内容に戻す。
    fn set_sampling_null(&mut self, key: &'static str, null: bool, cx: &mut Context<Self>) {
        let mut next = self.sampling.values.clone();
        next.remove(key);
        if null {
            next.insert(key.into(), Value::Null);
        } else if let Some((field, Some(input))) =
            self.sampling.fields.iter().zip(&self.sampling.inputs).find(|(f, _)| f.key == key)
        {
            // 数値欄に値が残っていればそれに戻す(不正なら既定)
            if let Ok(Some(v)) = parse(field, &input.read(cx).value()) {
                next.insert(key.into(), v);
            }
        }
        self.commit_sampling(next, cx);
    }

    /// GUI に欄のある項目をすべて Irodori の既定に戻す(手書きの未知の項目は残す)。
    fn reset_sampling(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        for input in self.sampling.inputs.iter().flatten() {
            input.update(cx, |state, cx| state.set_value("", window, cx));
        }
        let fields = &self.sampling.fields;
        let next: Map<String, Value> = self
            .sampling
            .values
            .iter()
            .filter(|(k, _)| !fields.iter().any(|f| f.key == k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        self.sampling.error = None;
        self.commit_sampling(next, cx);
    }

    /// 上書きの組を確定する: backend.json に記録し、エンジンへ送る。
    fn commit_sampling(&mut self, next: Map<String, Value>, cx: &mut Context<Self>) {
        if next == self.sampling.values {
            cx.notify();
            return;
        }
        let path = config::default_user_config_path(&self.root);
        if let Err(e) = config::set_user_config_value(&path, "tts", "sampling", Value::Object(next.clone())) {
            // 保存できなくても、この起動中は効かせる
            self.sampling.error = Some(format!("保存できませんでした(この起動中のみ有効): {e:#}"));
        }
        self.sampling.values = next;
        self.send(GuiMessage::Configure {
            tts: Some(TtsConfig { sampling: Some(self.sampling.values.clone()), ..Default::default() }),
            asr: None,
            audio: None,
            voice: None,
            pipeline: None,
        });
        self.push_log(format!("合成パラメータ: {}", Value::Object(self.sampling.values.clone())));
        cx.notify();
    }

    /// 右レール「声」の中の折りたたみ欄
    pub(super) fn render_sampling(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let editor = &self.sampling;
        let changed = editor.changed_count();
        let toggle = Button::new("sampling-toggle")
            .xsmall()
            .ghost()
            .icon(if editor.open { IconName::ChevronDown } else { IconName::ChevronRight })
            .label(if changed > 0 { format!("合成パラメータ · {changed} 項目を変更") } else { "合成パラメータ".to_string() })
            .on_click(cx.listener(|this, _, _, cx| {
                this.sampling.open = !this.sampling.open;
                cx.notify();
            }));

        v_flex().gap_2().child(h_flex().child(toggle)).when(editor.open, |col| {
            let rows: Vec<AnyElement> = editor
                .fields
                .iter()
                .zip(&editor.inputs)
                .map(|(field, input)| self.render_sampling_row(field, input.as_ref(), cx).into_any_element())
                .collect();
            let unknown = editor.unknown_keys();
            col.child(v_flex().gap_3().pl_2().children(rows))
                .when_some(editor.error.clone(), |col, e| {
                    col.child(div().text_xs().text_color(c(theme::ERROR)).child(e))
                })
                .when(!unknown.is_empty(), |col| {
                    col.child(kit::hint(format!("GUI に欄の無い項目(backend.json の値を使用): {}", unknown.join(", "))))
                })
                .child(
                    h_flex()
                        .gap_1()
                        .justify_end()
                        .child(
                            Button::new("sampling-reset")
                                .xsmall()
                                .ghost()
                                .icon(IconName::Undo2)
                                .label("既定に戻す")
                                .disabled(changed == unknown.len())
                                .on_click(cx.listener(|this, _, window, cx| this.reset_sampling(window, cx))),
                        )
                        .child(
                            Button::new("open-data")
                                .xsmall()
                                .ghost()
                                .icon(IconName::FolderOpen)
                                .tooltip("data/backend.json の tts.sampling に保存されます")
                                .on_click(cx.listener(|this, _, _, _| this.open_data_folder())),
                        ),
                )
        })
    }

    fn render_sampling_row(
        &self,
        field: &SamplingField,
        input: Option<&Entity<InputState>>,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let key = field.key;
        let value = self.sampling.values.get(key);
        let is_null = field.null_label.is_some() && value.is_some_and(Value::is_null);
        let control: AnyElement = match (field.kind, input) {
            (SamplingKind::Bool, _) => {
                let weak = cx.weak_entity();
                let checked = value.and_then(Value::as_bool).or(field.default.as_bool()).unwrap_or(false);
                Switch::new(SharedString::from(format!("sampling-{key}")))
                    .checked(checked)
                    .on_click(move |on, _, cx| {
                        let _ = weak.update(cx, |this, cx| this.set_sampling_bool(key, *on, cx));
                    })
                    .into_any_element()
            }
            (_, Some(input)) if !is_null => div().w(px(96.)).child(Input::new(input).small()).into_any_element(),
            _ => div().text_xs().text_color(c(theme::TEXT_MUTED)).child("—").into_any_element(),
        };
        v_flex()
            .gap_0p5()
            .child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .min_w_0()
                            .text_xs()
                            .text_color(c(if value.is_some() { theme::TEXT } else { theme::TEXT_MUTED }))
                            .child(field.label),
                    )
                    .child(control),
            )
            .child(kit::hint(format!("{} · {key}", field.help)))
            .when_some(field.null_label, |col, label| {
                let weak = cx.weak_entity();
                col.child(
                    Switch::new(SharedString::from(format!("sampling-null-{key}")))
                        .xsmall()
                        .checked(is_null)
                        .label(label)
                        .on_click(move |on, _, cx| {
                            let _ = weak.update(cx, |this, cx| this.set_sampling_null(key, *on, cx));
                        }),
                )
            })
    }
}

#[cfg(test)]
mod tests {
    // super::* は gpui の #[test] を持ち込むので個別に使う
    use super::{parse, show};
    use serde_json::Value;
    use sttts_engine::tts::{SamplingField, sampling_fields};

    fn field(key: &str) -> SamplingField {
        sampling_fields().into_iter().find(|f| f.key == key).unwrap()
    }

    #[test]
    fn parse_by_kind() {
        assert_eq!(parse(&field("num_steps"), " 8 "), Ok(Some(Value::from(8u64))));
        assert!(parse(&field("num_steps"), "1.5").is_err());
        assert_eq!(parse(&field("duration_scale"), "1.25"), Ok(Some(Value::from(1.25))));
        assert!(parse(&field("duration_scale"), "fast").is_err());
        assert!(parse(&field("duration_scale"), "NaN").is_err());
        // 空欄 = 上書きしない
        assert_eq!(parse(&field("seconds"), "  "), Ok(None));
    }

    #[test]
    fn defaults_are_shown_readably() {
        assert_eq!(show(&field("tail_std_threshold").default), "0.05");
        assert_eq!(show(&field("seconds").default), "なし");
        assert_eq!(field("ref_normalize_db").null_label, Some("正規化しない"));
    }
}
