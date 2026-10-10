//! ログ: ログ欄(直近 300 行)と data/gui.log、エラー行の判定。

use std::io::Write as _;

use gpui_kit::*;

use super::StttsApp;

impl StttsApp {
    pub(super) fn push_log(&mut self, line: impl Into<String>) {
        let line = line.into();
        if let Some(f) = self.log_file.as_mut() {
            let _ = writeln!(f, "{line}");
        }
        if !self.log_open && is_error_line(&line) {
            self.unread_errors += 1;
        }
        self.logs.push_back(line);
        while self.logs.len() > 300 {
            self.logs.pop_front();
        }
        self.log_scroll.scroll_to_bottom();
    }

    pub(super) fn toggle_log(&mut self, cx: &mut Context<Self>) {
        self.log_open = !self.log_open;
        if self.log_open {
            self.unread_errors = 0;
            self.log_scroll.scroll_to_bottom();
        }
        cx.notify();
    }
}

/// エラー表示(赤字・未読バッジ)の対象行。GUI が出す行は `[error:…]` / `[warn]` を付ける。
/// 下位クレートのエラー文(日本語)が info で届くこともあるので、その語も見る。
pub(crate) fn is_error_line(line: &str) -> bool {
    let head = line.get(..6).unwrap_or(line).to_ascii_lowercase();
    head.starts_with("[error")
        || head.starts_with("[warn")
        || line.contains("エラー")
        || line.contains("失敗")
        || line.contains("切れました")
}

/// 起動時に data/gui.log を空にして開く(古いログは残さない)。
pub(super) fn open_log_file(root: &std::path::Path) -> Option<std::fs::File> {
    let dir = root.join("data");
    std::fs::create_dir_all(&dir).ok()?;
    std::fs::File::create(dir.join("gui.log")).ok()
}
