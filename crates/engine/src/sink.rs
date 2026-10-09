//! エンジン → GUI のメッセージ送出口。

use std::sync::Arc;

use sttts_protocol::BackendMessage;

#[derive(Clone)]
pub struct Sink(Arc<dyn Fn(BackendMessage) + Send + Sync>);

impl Sink {
    pub fn new(f: impl Fn(BackendMessage) + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }

    pub fn send(&self, msg: BackendMessage) {
        (self.0)(msg);
    }

    pub fn log(&self, level: &str, message: impl Into<String>) {
        self.send(BackendMessage::Log { level: level.into(), message: message.into() });
    }

    pub fn info(&self, message: impl Into<String>) {
        self.log("info", message);
    }

    pub fn warn(&self, message: impl Into<String>) {
        self.log("warn", message);
    }

    pub fn debug(&self, message: impl Into<String>) {
        self.log("debug", message);
    }

    pub fn error(&self, scope: &str, message: impl Into<String>, recoverable: bool) {
        self.send(BackendMessage::Error { scope: scope.into(), message: message.into(), recoverable });
    }
}
