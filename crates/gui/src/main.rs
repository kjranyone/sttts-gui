//! sttts-gui — GPUI(gpui-kit)クライアントのエントリポイント。
//!
//! - エンジン(`sttts-engine`)を同じプロセスで起動し、チャネルで往復
//! - マイク → ASR → 確定文を選んだ声で合成 → rodio で逐次再生(疑似ストリーミング)
//! - 画面の責務と構成は app.rs 冒頭を参照

mod app;
mod audio;
mod backend;
mod device_picker;
mod locale;
mod secret;
mod settings;
mod sysmon;
mod theme;
mod turns;

use gpui_kit::component::TitleBar;
use gpui_kit::*;

fn main() {
    // --mock / --real で明示。無指定なら設定ファイルの mock を踏襲(初回は実エンジン)。
    let args: Vec<String> = std::env::args().collect();
    let saved = settings::AppSettings::load(&backend::repo_root());
    let lang = locale::initial(saved.language.as_deref());
    let mock = if args.iter().any(|a| a == "--mock") {
        true
    } else if args.iter().any(|a| a == "--real") {
        false
    } else {
        saved.mock.unwrap_or(false)
    };

    // コンポーネントのアイコン(タイトルバーのウィンドウ操作ボタン等)は SVG アセットとして同梱する
    gpui_kit::application().with_assets(gpui_kit::assets::Assets).run(move |cx: &mut App| {
        gpui_kit::init(cx);
        locale::apply(lang);
        theme::install(cx);
        let bounds = Bounds::centered(None, size(px(1240.), px(800.)), cx);
        gpui_kit::open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                window_min_size: Some(size(px(980.), px(620.))),
                // OS のタイトルバーを隠し、TitleBar コンポーネントが描画とドラッグを受け持つ
                ..TitleBar::window_options()
            },
            cx,
            |window, cx| cx.new(|cx| app::StttsApp::new(mock, window, cx)),
        )
        .expect("failed to open the window");
        cx.activate(true);
    });
}
