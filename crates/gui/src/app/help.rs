//! 「?」から開く解説。普段の画面には説明文を置かず(プロ向けの道具なので)、
//! 知りたいときだけモーダルで読めるようにする。

use gpui_kit::component::button::*;
use gpui_kit::component::*;
use gpui_kit::*;

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
            Self::AutoPlay => "自動再生",
            Self::Tempo => "テンポと間を再現",
            Self::Voice => "声",
            Self::Style => "話し方",
            Self::Recognition => "認識",
            Self::Acting => "演技",
            Self::Seed => "乱数(seed)",
        }
    }

    /// 段落の並び。「・」で始まる行は箇条書きとして表示する。
    fn body(self) -> &'static [&'static str] {
        match self {
            Self::AutoPlay => &[
                "ON: 認識が確定した文を、そのまま読み上げます。",
                "OFF: 確定した文はカードで止まります。「この内容で話す」「訂正する」「話さない」から選んでください。",
                "スピーカーで鳴らすと、その音をマイクが拾って同じ文を繰り返し読み上げることがあります。ヘッドホンを使ってください。",
            ],
            Self::Tempo => &[
                "マイクで話したときの速さと間を測り、合成音声に反映します。",
                "・普段より速く話すと少し速く、ゆっくり話すと少しゆっくり読み上げます(Irodori の ⏩ / 🐢 と長さの調整)",
                "・途中の間が多いときは、間を取りながら読むよう指示します",
                "・最初の3回は比べる基準がないため、速さは変えません",
                "・文字入力には効きません。入力欄の「演技」で絵文字を付けてください",
                "ON のときは、認識の確定から読み上げ開始まで最大 0.15 秒待つことがあります。",
            ],
            Self::Voice => &[
                "読み上げに使う声です。「既定の声」は、声の見本を使わずに合成します。",
                "wav / flac(10 秒ほどの話し声)をウィンドウへドラッグ&ドロップするか「＋」から選ぶと、声として追加されます。画像を落とすと、選んでいる声のアイコンになります。",
                "見本にする声は、本人の同意があるものだけを使ってください。",
            ],
            Self::Style => &[
                "Irodori に渡す話し方の指示です。例: 落ち着いて、近い距離感で",
                "声の質や性別は「声」の選択で決まります。ここには場面や伝え方を書くと安定します。",
            ],
            Self::Recognition => &[
                "クラウド(Gemini): 話している間の音声を Google に送り、文字にします。API キーが必要です。キーは、この PC のこのユーザーだけが読める形で暗号化して保存します。",
                "ローカル: この PC の中で文字にします。音声は外部に送りません。",
                "切り替えは、次にライブを開始したときから使われます。",
            ],
            Self::Acting => &[
                "押した絵文字を入力欄に入れます。Irodori は文中の絵文字を話し方の指示として読みます。",
                "パレットは Irodori-TTS v4 の対応絵文字 45 種で、公式と同じ順です。意味はボタンにカーソルを合わせると出ます。",
                "⏸️ は文の末尾に入ります。それ以外は文頭に入ります。効かせたい位置へ移してください。",
                "同じ絵文字を重ねると効果が強くなることがあります。効き方は声や前後の文で変わります。",
                "一覧は docs/irodori-annotations.md にあります。",
            ],
            Self::Seed => &[
                "合成のゆらぎを決める値です。",
                "ランダム: 同じ文でも毎回少し違う読み方になります。",
                "固定: 同じ文・同じ声なら、毎回同じ読み方になります。",
            ],
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
        let paragraphs = topic
            .body()
            .iter()
            .map(|line| match line.strip_prefix('・') {
                Some(item) => h_flex()
                    .gap_2()
                    .items_start()
                    .child(div().text_color(c(theme::TEXT_FAINT)).child("・"))
                    .child(div().flex_1().child(item))
                    .into_any_element(),
                None => div().child(*line).into_any_element(),
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

    fn close_help(&mut self, cx: &mut Context<Self>) {
        self.help_topic = None;
        cx.notify();
    }
}
