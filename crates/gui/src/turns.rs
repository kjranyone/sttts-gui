//! 発話ターン: 「話した内容」と「届けた声」の対応。
//!
//! 1ターン = 入力1件(マイクの1発話 or テキスト入力1件)と、それを合成・再生した結果。
//! backend のイベント(asr_partial / asr_final / speak_accepted / tts_chunk_start /
//! tts_audio / speak_done)をターンへ集約する。対応付けは
//! - マイク発話: asr_* の `utterance` と speak_accepted の `utterance`
//! - GUI から送った発話: Speak の `tag`(= [`tag_for`])と speak_accepted の `tag`
//! - 合成チャンク: speak_accepted で得た `request`
//!
//! 描画や音声再生はここで扱わない(状態遷移だけを持ち、単体テストで固める)。

use std::collections::VecDeque;
use sttts_protocol::DeliveryInfo;

/// 保持するターン数の上限(古いものから捨てる)
const MAX_TURNS: usize = 300;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnSource {
    Mic { utterance: u64 },
    Typed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnStatus {
    /// 認識の途中経過を受信中
    Listening,
    /// 確定済み。自動再生 OFF のため、利用者の操作待ち
    AwaitingConfirm,
    /// 発話を依頼済み(自動発話 or 送信)、backend の受付待ち
    Queued,
    /// 受付済み。合成・再生中
    Speaking,
    Done,
    Cancelled,
    Failed,
    /// 確定文が空だった(聞き取れなかった)
    Unheard,
    /// 発話しなかった(破棄、または後続の発話が先に受け付けられた)
    Skipped,
    /// 確定前にライブが止まった
    Interrupted,
}

impl TurnStatus {
    pub fn is_active(self) -> bool {
        matches!(self, Self::Queued | Self::Speaking)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub index: u32,
    pub text: String,
    /// 合成済みか(tts_audio を受信した)
    pub ready: bool,
    /// 生成 WAV(再生し直し用)
    pub path: Option<String>,
    pub duration_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub id: u64,
    pub source: TurnSource,
    /// 話した内容(認識文 or 入力文)
    pub text: String,
    pub status: TurnStatus,
    /// 発話した声の表示名(受付時点の選択。None = 既定の声)
    pub voice: Option<String>,
    pub request: Option<u64>,
    pub chunks: Vec<Chunk>,
    /// 話し終え → 初音(マイク発話の先頭チャンクのみ)
    pub e2e_ms: Option<u64>,
    /// 確定文の認識時間
    pub asr_ms: Option<u64>,
    /// 元音声から推定した表現。転写文とは別に保持する。
    pub delivery: Option<DeliveryInfo>,
    /// ターンを作った時刻(待ち時間の表示用)
    pub created_at: std::time::Instant,
}

impl Turn {
    /// 作成からの経過秒(整数)
    pub fn waited_secs(&self) -> u64 {
        self.created_at.elapsed().as_secs()
    }

    pub fn ready_chunks(&self) -> usize {
        self.chunks.iter().filter(|c| c.ready).count()
    }

    pub fn audio_paths(&self) -> Vec<&str> {
        self.chunks.iter().filter_map(|c| c.path.as_deref()).collect()
    }
}

/// Speak の tag(speak_accepted でターンへ戻すための識別子)
pub fn tag_for(id: u64) -> String {
    format!("turn-{id}")
}

fn id_from_tag(tag: &str) -> Option<u64> {
    tag.strip_prefix("turn-")?.parse().ok()
}

#[derive(Debug, Default)]
pub struct Turns {
    items: VecDeque<Turn>,
    next_id: u64,
}

impl Turns {
    pub fn iter(&self) -> impl Iterator<Item = &Turn> {
        self.items.iter()
    }

    /// (発話待ち, 合成・再生中) のターン数
    pub fn queue_counts(&self) -> (usize, usize) {
        let queued = self.items.iter().filter(|t| t.status == TurnStatus::Queued).count();
        let speaking = self.items.iter().filter(|t| t.status == TurnStatus::Speaking).count();
        (queued, speaking)
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn get(&self, id: u64) -> Option<&Turn> {
        self.items.iter().find(|t| t.id == id)
    }

    /// 合成 request に対応するターンの id
    pub fn id_for_request(&self, request: u64) -> Option<u64> {
        self.items.iter().rev().find(|t| t.request == Some(request)).map(|t| t.id)
    }

    fn get_mut(&mut self, id: u64) -> Option<&mut Turn> {
        self.items.iter_mut().find(|t| t.id == id)
    }

    fn by_utterance(&mut self, utterance: u64) -> Option<&mut Turn> {
        self.items
            .iter_mut()
            .rev()
            .find(|t| t.source == TurnSource::Mic { utterance })
    }

    fn by_request(&mut self, request: u64) -> Option<&mut Turn> {
        self.items.iter_mut().rev().find(|t| t.request == Some(request))
    }

    fn push(&mut self, source: TurnSource, text: String, status: TurnStatus) -> u64 {
        self.next_id += 1;
        let id = self.next_id;
        self.items.push_back(Turn {
            id,
            source,
            text,
            status,
            voice: None,
            request: None,
            chunks: Vec::new(),
            e2e_ms: None,
            asr_ms: None,
            delivery: None,
            created_at: std::time::Instant::now(),
        });
        while self.items.len() > MAX_TURNS {
            self.items.pop_front();
        }
        id
    }

    /// 認識の途中経過。同じ発話のターンがあれば更新、無ければ作る。
    pub fn asr_partial(&mut self, utterance: u64, text: String) {
        match self.by_utterance(utterance) {
            // 確定後に遅れて届いた partial で確定文を戻さない
            Some(t) if t.status == TurnStatus::Listening => t.text = text,
            Some(_) => {}
            None => {
                self.push(TurnSource::Mic { utterance }, text, TurnStatus::Listening);
            }
        }
    }

    /// 認識の確定。`auto_speak` なら backend が自動で発話するので受付待ちにする。
    pub fn asr_final(&mut self, utterance: u64, text: String, asr_ms: Option<u64>, auto_speak: bool) {
        let status = if text.trim().is_empty() {
            TurnStatus::Unheard
        } else if auto_speak {
            TurnStatus::Queued
        } else {
            TurnStatus::AwaitingConfirm
        };
        let turn = match self.by_utterance(utterance) {
            Some(t) => t,
            None => {
                let id = self.push(TurnSource::Mic { utterance }, String::new(), status);
                self.get_mut(id).expect("just pushed")
            }
        };
        turn.text = text;
        turn.asr_ms = asr_ms;
        // speak_accepted が先に届いていた場合(投機的 TTS 等)は Speaking を保つ
        if turn.status != TurnStatus::Speaking {
            turn.status = status;
        }
    }

    pub fn set_delivery(&mut self, utterance: u64, delivery: Option<DeliveryInfo>) {
        if let Some(turn) = self.by_utterance(utterance) {
            turn.delivery = delivery;
        }
    }

    pub fn set_delivery_for_id(&mut self, id: u64, delivery: Option<DeliveryInfo>) {
        if let Some(turn) = self.get_mut(id) {
            turn.delivery = delivery;
        }
    }

    /// テキスト入力の発話。返した id を Speak の tag([`tag_for`])に使う。
    pub fn push_typed(&mut self, text: String) -> u64 {
        self.push(TurnSource::Typed, text, TurnStatus::Queued)
    }

    /// 確認待ちのターンを(訂正なしで)発話依頼した。
    pub fn mark_queued(&mut self, id: u64) {
        if let Some(t) = self.get_mut(id) {
            t.status = TurnStatus::Queued;
        }
    }

    /// 確認待ちのターンを発話せずに閉じる。
    pub fn dismiss(&mut self, id: u64) {
        if let Some(t) = self.get_mut(id) {
            if t.status == TurnStatus::AwaitingConfirm {
                t.status = TurnStatus::Skipped;
            }
        }
    }

    /// backend が発話を受け付けた。tag(GUI 送信)→ utterance(自動発話)の順でターンを探す。
    pub fn speak_accepted(
        &mut self,
        request: u64,
        tag: Option<&str>,
        utterance: Option<u64>,
        voice: Option<String>,
    ) {
        let id = tag
            .and_then(id_from_tag)
            .filter(|id| self.get(*id).is_some())
            .or_else(|| utterance.and_then(|u| self.by_utterance(u).map(|t| t.id)));
        let id = match id {
            Some(id) => id,
            // GUI が送っていない発話(外部ツール等)も1ターンとして見せる
            None => self.push(TurnSource::Typed, String::new(), TurnStatus::Queued),
        };
        // これより前に自動発話待ちのまま受け付けられなかったマイク発話は、発話されない
        for t in self.items.iter_mut() {
            if t.id >= id {
                break;
            }
            if t.status == TurnStatus::Queued && t.request.is_none() && matches!(t.source, TurnSource::Mic { .. }) {
                t.status = TurnStatus::Skipped;
            }
        }
        let turn = self.get_mut(id).expect("resolved above");
        turn.request = Some(request);
        turn.status = TurnStatus::Speaking;
        turn.voice = voice;
        turn.chunks.clear();
    }

    pub fn chunk_start(&mut self, request: u64, index: u32, text: String) {
        if let Some(t) = self.by_request(request) {
            match t.chunks.iter_mut().find(|c| c.index == index) {
                Some(c) => c.text = text,
                None => t.chunks.push(Chunk { index, text, ready: false, path: None, duration_ms: None }),
            }
        }
    }

    pub fn chunk_audio(
        &mut self,
        request: u64,
        index: u32,
        path: Option<String>,
        duration_ms: u64,
        e2e_ms: Option<u64>,
    ) {
        if let Some(t) = self.by_request(request) {
            if let Some(ms) = e2e_ms {
                t.e2e_ms = Some(ms);
            }
            match t.chunks.iter_mut().find(|c| c.index == index) {
                Some(c) => {
                    c.ready = true;
                    c.path = path;
                    c.duration_ms = Some(duration_ms);
                }
                None => t.chunks.push(Chunk {
                    index,
                    text: String::new(),
                    ready: true,
                    path,
                    duration_ms: Some(duration_ms),
                }),
            }
            t.chunks.sort_by_key(|c| c.index);
        }
    }

    pub fn speak_done(&mut self, request: u64, cancelled: bool, failed: bool) {
        if let Some(t) = self.by_request(request) {
            t.status = if failed {
                TurnStatus::Failed
            } else if cancelled {
                TurnStatus::Cancelled
            } else {
                TurnStatus::Done
            };
        }
    }

    /// 利用者が発話を止めた。受付待ち・発話中のターンをすべて中止にする。
    pub fn cancel_active(&mut self) {
        for t in self.items.iter_mut().filter(|t| t.status.is_active()) {
            t.status = TurnStatus::Cancelled;
        }
    }

    /// backend の(再)接続。request id は振り直されるので、進行中のターンを切り離す。
    /// ライブが止まった。確定前の途中経過は確定しないので閉じる。
    pub fn mic_stopped(&mut self) {
        for t in self.items.iter_mut().filter(|t| t.status == TurnStatus::Listening) {
            t.status = TurnStatus::Interrupted;
        }
    }

    pub fn backend_restarted(&mut self) {
        for t in self.items.iter_mut() {
            if t.status.is_active() || t.status == TurnStatus::Listening {
                t.status = TurnStatus::Failed;
            }
            t.request = None;
        }
    }
}

/// 再生キューの中身(どのターンのどのチャンクか)を、Sink に積んだ順に覚える。
///
/// rodio の Sink は「残り何個か」しか教えてくれないため、積んだ順の記録と
/// 残数([`Playback::sync`])を突き合わせて、いま鳴っているチャンクを割り出す。
#[derive(Debug, Default)]
pub struct Playback {
    /// Sink に積んだ順。先頭が再生中(または再生済みで未同期)
    queue: VecDeque<(u64, u32)>,
}

impl Playback {
    pub fn push(&mut self, turn: u64, chunk: u32) {
        self.queue.push_back((turn, chunk));
    }

    /// Sink の残数(再生中を含む)に合わせて再生済みを捨てる。いま鳴っているものが変わったら true。
    pub fn sync(&mut self, remaining: usize) -> bool {
        let before = self.current();
        while self.queue.len() > remaining {
            self.queue.pop_front();
        }
        self.current() != before
    }

    /// いま鳴っている (ターン, チャンク)
    pub fn current(&self) -> Option<(u64, u32)> {
        self.queue.front().copied()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Sink を空にしたとき(キャンセル・出力デバイス切替)
    pub fn clear(&mut self) {
        self.queue.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn playback_follows_sink_remaining_count() {
        let mut p = Playback::default();
        assert_eq!(p.current(), None);
        p.push(1, 0);
        p.push(1, 1);
        p.push(2, 0);
        assert!(!p.sync(3));
        assert_eq!(p.current(), Some((1, 0)));
        assert!(p.sync(2));
        assert_eq!(p.current(), Some((1, 1)));
        // 再生中に後続が積まれても先頭は変わらない
        p.push(2, 1);
        assert!(!p.sync(3));
        assert_eq!(p.current(), Some((1, 1)));
        assert!(p.sync(0));
        assert_eq!(p.current(), None);
        assert!(p.is_empty());
    }

    #[test]
    fn playback_clear_forgets_queue() {
        let mut p = Playback::default();
        p.push(3, 0);
        p.clear();
        assert_eq!(p.current(), None);
    }

    #[test]
    fn mic_utterance_flows_from_partial_to_done() {
        let mut t = Turns::default();
        t.asr_partial(7, "こんに".into());
        t.asr_partial(7, "こんにちは".into());
        assert_eq!(t.len(), 1);
        t.asr_final(7, "こんにちは。".into(), Some(120), true);
        let turn = t.iter().next().unwrap();
        assert_eq!(turn.status, TurnStatus::Queued);
        assert_eq!(turn.asr_ms, Some(120));

        t.speak_accepted(3, None, Some(7), Some("さくら".into()));
        t.chunk_start(3, 0, "こんにちは。".into());
        t.chunk_audio(3, 0, Some("out/a.wav".into()), 900, Some(820));
        let turn = t.iter().next().unwrap();
        assert_eq!(turn.status, TurnStatus::Speaking);
        assert_eq!(turn.voice.as_deref(), Some("さくら"));
        assert_eq!(turn.ready_chunks(), 1);
        assert_eq!(turn.e2e_ms, Some(820));
        assert_eq!(turn.audio_paths(), vec!["out/a.wav"]);

        t.speak_done(3, false, false);
        assert_eq!(t.iter().next().unwrap().status, TurnStatus::Done);
    }

    #[test]
    fn mic_stop_closes_unfinished_partial() {
        let mut t = Turns::default();
        t.asr_final(1, "確定".into(), None, true);
        t.asr_partial(2, "途中".into());
        t.mic_stopped();
        let statuses: Vec<_> = t.iter().map(|x| x.status).collect();
        assert_eq!(statuses, vec![TurnStatus::Queued, TurnStatus::Interrupted]);
    }

    #[test]
    fn late_partial_does_not_overwrite_final() {
        let mut t = Turns::default();
        t.asr_final(1, "確定".into(), None, true);
        t.asr_partial(1, "途中".into());
        assert_eq!(t.iter().next().unwrap().text, "確定");
    }

    #[test]
    fn confirm_mode_waits_then_typed_tag_maps_back() {
        let mut t = Turns::default();
        t.asr_final(1, "確認して".into(), None, false);
        let id = t.iter().next().unwrap().id;
        assert_eq!(t.get(id).unwrap().status, TurnStatus::AwaitingConfirm);

        t.mark_queued(id);
        t.speak_accepted(10, Some(&tag_for(id)), None, None);
        assert_eq!(t.get(id).unwrap().status, TurnStatus::Speaking);
        assert_eq!(t.get(id).unwrap().request, Some(10));
    }

    #[test]
    fn typed_turn_maps_by_tag_and_chunks_keep_order() {
        let mut t = Turns::default();
        let id = t.push_typed("一。二。".into());
        t.speak_accepted(5, Some(&tag_for(id)), None, None);
        t.chunk_start(5, 0, "一。".into());
        t.chunk_start(5, 1, "二。".into());
        t.chunk_audio(5, 1, None, 500, None);
        t.chunk_audio(5, 0, None, 500, None);
        let turn = t.get(id).unwrap();
        assert_eq!(turn.chunks.iter().map(|c| c.index).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(turn.ready_chunks(), 2);
    }

    #[test]
    fn empty_final_is_unheard_and_not_spoken() {
        let mut t = Turns::default();
        t.asr_partial(2, "あ".into());
        t.asr_final(2, "  ".into(), None, true);
        assert_eq!(t.iter().next().unwrap().status, TurnStatus::Unheard);
    }

    #[test]
    fn earlier_unaccepted_auto_speak_is_skipped() {
        let mut t = Turns::default();
        t.asr_final(1, "一つ目".into(), None, true);
        t.asr_final(2, "二つ目".into(), None, true);
        t.speak_accepted(1, None, Some(2), None);
        let statuses: Vec<_> = t.iter().map(|x| x.status).collect();
        assert_eq!(statuses, vec![TurnStatus::Skipped, TurnStatus::Speaking]);
    }

    #[test]
    fn cancel_and_backend_restart_settle_active_turns() {
        let mut t = Turns::default();
        let a = t.push_typed("a".into());
        t.speak_accepted(1, Some(&tag_for(a)), None, None);
        let b = t.push_typed("b".into());
        t.cancel_active();
        assert_eq!(t.get(a).unwrap().status, TurnStatus::Cancelled);
        assert_eq!(t.get(b).unwrap().status, TurnStatus::Cancelled);

        let c = t.push_typed("c".into());
        t.speak_accepted(2, Some(&tag_for(c)), None, None);
        t.backend_restarted();
        assert_eq!(t.get(c).unwrap().status, TurnStatus::Failed);
        // 再接続後の request=2 は別の発話。古いターンに紐付かない
        t.chunk_audio(2, 0, None, 100, None);
        assert!(t.get(c).unwrap().chunks.is_empty());
    }

    #[test]
    fn unknown_accept_creates_turn_and_old_turns_are_trimmed() {
        let mut t = Turns::default();
        t.speak_accepted(9, Some("external"), None, None);
        assert_eq!(t.len(), 1);
        for i in 0..(MAX_TURNS as u64 + 5) {
            t.push_typed(format!("{i}"));
        }
        assert_eq!(t.len(), MAX_TURNS);
    }
}
