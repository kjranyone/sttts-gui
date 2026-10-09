//! 「?」から開く解説。普段の画面には説明文を置かず(プロ向けの道具なので)、
//! 知りたいときだけモーダルで読めるようにする。

use gpui_kit::component::button::*;
use gpui_kit::component::*;
use gpui_kit::*;
use sttts_i18n::{tr, trf};

use super::StttsApp;
use crate::theme::{self, c, ca};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HelpTopic {
    AutoPlay,
    Tempo,
    Voice,
    Style,
    Recognition,
    Acting,
    Seed,
}

impl HelpTopic {
    fn title(self) -> &'static str {
        match self {
            Self::AutoPlay => tr!("Auto play", "自動再生", "自动播放"),
            Self::Tempo => tr!("Match tempo and pauses", "テンポと間を再現", "还原语速与停顿"),
            Self::Voice => tr!("Voice", "声", "声音"),
            Self::Style => tr!("Speaking style", "話し方", "说话方式"),
            Self::Recognition => tr!("Recognition", "認識", "识别"),
            Self::Acting => tr!("Acting", "演技", "表演"),
            Self::Seed => tr!("Random seed", "乱数(seed)", "随机种子(seed)"),
        }
    }

    /// 段落の並び。「・」で始まる行は箇条書きとして表示する。
    fn body(self) -> &'static [&'static str] {
        match self {
            Self::AutoPlay => tr!(
                &[
                    "ON: sentences are read aloud as soon as recognition confirms them.",
                    "OFF: confirmed sentences stop at their card. Choose \"Speak this\", \"Correct\" or \"Don't speak\".",
                    "If you play through speakers, the mic may pick up the voice and read the same sentence again and again. Use headphones.",
                ],
                &[
                    "ON: 認識が確定した文を、そのまま読み上げます。",
                    "OFF: 確定した文はカードで止まります。「この内容で話す」「訂正する」「話さない」から選んでください。",
                    "スピーカーで鳴らすと、その音をマイクが拾って同じ文を繰り返し読み上げることがあります。ヘッドホンを使ってください。",
                ],
                &[
                    "开启:识别确定的句子会直接朗读出来。",
                    "关闭:确定的句子会停在卡片上。请选择「按此内容发话」「修改」或「不发话」。",
                    "如果用扬声器播放,麦克风可能会收录声音,导致同一句话被反复朗读。请使用耳机。",
                ],
            ),
            Self::Tempo => tr!(
                &[
                    "Measures the speed and pauses of what you said into the mic and applies them to the synthesized voice.",
                    "・Speak faster than usual and it reads a little faster; speak slowly and it reads a little slower (Irodori's ⏩ / 🐢 and length scaling)",
                    "・When you pause a lot, it asks Irodori to read with pauses",
                    "・The first 3 utterances keep the normal speed, since there is nothing to compare against yet",
                    "・It does not apply to typed text. Add emoji from \"Acting\" above the input box instead",
                    "When ON, reading may start up to 0.15 s after recognition is confirmed.",
                ],
                &[
                    "マイクで話したときの速さと間を測り、合成音声に反映します。",
                    "・普段より速く話すと少し速く、ゆっくり話すと少しゆっくり読み上げます(Irodori の ⏩ / 🐢 と長さの調整)",
                    "・途中の間が多いときは、間を取りながら読むよう指示します",
                    "・最初の3回は比べる基準がないため、速さは変えません",
                    "・文字入力には効きません。入力欄の「演技」で絵文字を付けてください",
                    "ON のときは、認識の確定から読み上げ開始まで最大 0.15 秒待つことがあります。",
                ],
                &[
                    "测量你对麦克风说话时的语速和停顿,并反映到合成语音中。",
                    "・说得比平时快,朗读也会稍快;说得慢,朗读也会稍慢(使用 Irodori 的 ⏩ / 🐢 和时长调整)",
                    "・中途停顿较多时,会指示边停顿边朗读",
                    "・前 3 次没有可比较的基准,因此不改变语速",
                    "・对文字输入无效。请用输入框上方的「表演」添加表情符号",
                    "开启时,从识别确定到开始朗读最多可能等待 0.15 秒。",
                ],
            ),
            Self::Voice => tr!(
                &[
                    "The voice used for reading aloud. \"Default voice\" synthesizes without a voice sample.",
                    "Drag and drop a wav / flac file (about 10 seconds of speech) onto the window, or pick one with \"+\", to add it as a voice. Drop an image to make it the icon of the selected voice.",
                    "Only use voice samples from people who have given their consent.",
                ],
                &[
                    "読み上げに使う声です。「既定の声」は、声の見本を使わずに合成します。",
                    "wav / flac(10 秒ほどの話し声)をウィンドウへドラッグ&ドロップするか「＋」から選ぶと、声として追加されます。画像を落とすと、選んでいる声のアイコンになります。",
                    "見本にする声は、本人の同意があるものだけを使ってください。",
                ],
                &[
                    "用于朗读的声音。「默认声音」不使用声音样本进行合成。",
                    "将 wav / flac(约 10 秒的说话声)拖放到窗口,或通过「＋」选择,即可添加为声音。拖放图片会将其设为当前所选声音的图标。",
                    "请只使用已获得本人同意的声音作为样本。",
                ],
            ),
            Self::Style => tr!(
                &[
                    "A speaking-style instruction passed to Irodori. Irodori expects Japanese, e.g. 落ち着いて、近い距離感で (calm, close distance)",
                    "Voice quality and gender come from the \"Voice\" selection. Describing the scene or how to deliver the line works best here.",
                ],
                &[
                    "Irodori に渡す話し方の指示です。例: 落ち着いて、近い距離感で",
                    "声の質や性別は「声」の選択で決まります。ここには場面や伝え方を書くと安定します。",
                ],
                &[
                    "传给 Irodori 的说话方式提示。Irodori 以日语为前提,例如:落ち着いて、近い距離感で(沉稳、近距离感)",
                    "音色和性别由「声音」的选择决定。这里写场景或表达方式效果更稳定。",
                ],
            ),
            Self::Recognition => tr!(
                &[
                    "Cloud (Gemini): sends your audio to Google while you speak and turns it into text. Requires an API key. The key is stored encrypted so that only this user on this PC can read it.",
                    "Local: transcribes on this PC. Audio is not sent anywhere.",
                    "A change takes effect the next time you start live.",
                ],
                &[
                    "クラウド(Gemini): 話している間の音声を Google に送り、文字にします。API キーが必要です。キーは、この PC のこのユーザーだけが読める形で暗号化して保存します。",
                    "ローカル: この PC の中で文字にします。音声は外部に送りません。",
                    "切り替えは、次にライブを開始したときから使われます。",
                ],
                &[
                    "云端(Gemini):说话时将音频发送到 Google 转成文字。需要 API 密钥。密钥会加密保存,只有本电脑的当前用户才能读取。",
                    "本地:在本电脑内转成文字。音频不会发送到外部。",
                    "切换将在下次开始直播时生效。",
                ],
            ),
            Self::Acting => tr!(
                &[
                    "Inserts the emoji you press into the input box. Irodori reads emoji in the text as delivery instructions.",
                    "The palette is the 45 emoji supported by Irodori-TTS v4, in the official order. Hover over a button to see its meaning.",
                    "The emoji goes in at the cursor in the input box. If text is selected, the selection is replaced by the emoji.",
                    "Repeating the same emoji can strengthen the effect. How well it works depends on the voice and the surrounding text.",
                    "The full list is in docs/irodori-annotations.md.",
                ],
                &[
                    "押した絵文字を入力欄に入れます。Irodori は文中の絵文字を話し方の指示として読みます。",
                    "パレットは Irodori-TTS v4 の対応絵文字 45 種で、公式と同じ順です。意味はボタンにカーソルを合わせると出ます。",
                    "押した絵文字は、入力欄のカーソル位置に入ります。範囲を選んでいるときは、その範囲が絵文字に置き換わります。",
                    "同じ絵文字を重ねると効果が強くなることがあります。効き方は声や前後の文で変わります。",
                    "一覧は docs/irodori-annotations.md にあります。",
                ],
                &[
                    "将按下的表情符号插入输入框。Irodori 会把文中的表情符号当作说话方式的提示来读。",
                    "调色板是 Irodori-TTS v4 支持的 45 种表情符号,顺序与官方相同。将鼠标悬停在按钮上可查看含义。",
                    "表情符号会插入到输入框的光标位置。选中了文字时,选中的部分会被替换为表情符号。",
                    "叠加同一个表情符号有时会增强效果。效果因声音和前后文而异。",
                    "完整列表见 docs/irodori-annotations.md。",
                ],
            ),
            Self::Seed => tr!(
                &[
                    "The value that decides the variation in synthesis.",
                    "Random: the same sentence is read a little differently each time.",
                    "Fixed: the same sentence with the same voice is read the same way every time.",
                ],
                &[
                    "合成のゆらぎを決める値です。",
                    "ランダム: 同じ文でも毎回少し違う読み方になります。",
                    "固定: 同じ文・同じ声なら、毎回同じ読み方になります。",
                ],
                &[
                    "决定合成随机变化的值。",
                    "随机:同一句话每次的读法都会略有不同。",
                    "固定:同一句话、同一个声音,每次读法都相同。",
                ],
            ),
        }
    }
}

impl StttsApp {
    /// 項目名の横に置く「?」。押すと解説モーダルを開く(Button なのでキーボード・読み上げでも使える)。
    pub(super) fn help_icon(&self, topic: HelpTopic, cx: &mut Context<Self>) -> Button {
        Button::new(SharedString::from(format!("help-{topic:?}")))
            .xsmall()
            .ghost()
            .label("?")
            .size(px(18.))
            .rounded_full()
            .border_1()
            .border_color(c(theme::BORDER_STRONG))
            .text_color(c(theme::TEXT_FAINT))
            .on_click(cx.listener(move |this, _, _, cx| {
                this.help_topic = Some(topic);
                cx.notify();
            }))
    }

    pub(super) fn render_help_modal(&self, cx: &mut Context<Self>) -> Option<AnyElement> {
        let topic = self.help_topic?;
        let mut lines: Vec<String> = topic.body().iter().map(|s| s.to_string()).collect();
        if topic == HelpTopic::Recognition {
            // Gemini の段落の直後に、実際に使うモデル名を添える
            let model = self.gemini_model_name();
            lines.insert(
                1,
                trf!(
                    "・Model: {model} (change it with asr.gemini_model in data/backend.json)",
                    "・使用モデル: {model}(data/backend.json の asr.gemini_model で変更できます)",
                    "・使用模型:{model}(可在 data/backend.json 的 asr.gemini_model 中更改)"
                ),
            );
        }
        let paragraphs = lines.into_iter().map(|line| match line.strip_prefix('・') {
            Some(item) => h_flex()
                .gap_2()
                .items_start()
                .child(div().text_color(c(theme::TEXT_FAINT)).child("・"))
                // min_w_0 が無いと flex 子が内容幅より縮まず、長い行が折り返さずにはみ出す
                .child(div().flex_1().min_w_0().child(item.to_string()))
                .into_any_element(),
            None => div().child(line).into_any_element(),
        });
        Some(
            div()
                .absolute()
                .top_0()
                .left_0()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .child(
                    // 背景の幕(クリックで閉じる)
                    div()
                        .id("help-scrim")
                        .absolute()
                        .top_0()
                        .left_0()
                        .size_full()
                        .bg(ca(0x07060f, 0x99))
                        .on_click(cx.listener(|this, _, _, cx| this.close_help(cx))),
                )
                .child(
                    v_flex()
                        .id("help-card")
                        .occlude()
                        .w(px(480.))
                        .max_w(relative(0.9))
                        .rounded_lg()
                        .bg(c(theme::CARD))
                        .border_1()
                        .border_color(c(theme::BORDER_STRONG))
                        .shadow_lg()
                        .child(
                            h_flex()
                                .px_5()
                                .pt_4()
                                .pb_2()
                                .justify_between()
                                .items_center()
                                .child(
                                    div()
                                        .text_base()
                                        .font_weight(FontWeight::SEMIBOLD)
                                        .text_color(c(theme::TEXT))
                                        .child(topic.title()),
                                )
                                .child(
                                    Button::new("close-help")
                                        .small()
                                        .ghost()
                                        .icon(IconName::Close)
                                        .on_click(
                                            cx.listener(|this, _, _, cx| this.close_help(cx)),
                                        ),
                                ),
                        )
                        .child(
                            v_flex()
                                .px_5()
                                .pb_5()
                                .gap_2()
                                .text_sm()
                                .line_height(rems(1.6))
                                .text_color(c(theme::TEXT_MUTED))
                                .children(paragraphs),
                        ),
                )
                .into_any_element(),
        )
    }

    /// Gemini のモデル名。読み込み済みならエンジンが報告した名前、まだなら設定
    /// (data/backend.json → 既定値)から引く。
    fn gemini_model_name(&self) -> String {
        if self.selected_asr_engine == "gemini" {
            // 切替直後は前のエンジン(ローカル)の名前が残っていることがあるので Gemini のものだけ使う
            if let Some(model) = self.asr_state.model.as_deref().filter(|m| m.contains("gemini")) {
                return model.to_string();
            }
        }
        use sttts_engine::config;
        let user = config::load_user_config(&config::default_user_config_path(&self.root), &|_| {});
        let cfg = config::merge_config(&config::default_config(), &user);
        config::get_str(&cfg, "asr", "gemini_model").unwrap_or(tr!("unknown", "不明", "未知")).to_string()
    }

    fn close_help(&mut self, cx: &mut Context<Self>) {
        self.help_topic = None;
        cx.notify();
    }
}
