//! Silero VAD(ONNX, silero-vad 6.x)。pip の `silero_vad.VADIterator` + `OnnxWrapper` の移植。
//!
//! モデルは `assets/silero_vad.onnx`(MIT)を埋め込み、ort の CPU EP・単一スレッドで動かす。
//! 512 サンプル(16kHz で 32ms)単位で `process` に渡す。

use anyhow::{Result, anyhow};
use ort::session::Session;
use ort::value::Tensor;

use crate::FRAME;

static MODEL: &[u8] = include_bytes!("../assets/silero_vad.onnx");

const SR: i64 = 16000;
/// 16kHz のコンテキスト長(各呼び出しの先頭に直前の末尾 64 サンプルを付ける)
const CONTEXT: usize = 64;
const STATE_LEN: usize = 2 * 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    /// 発話開始(サンプル位置、先頭から)
    Start(i64),
    /// 発話終了(サンプル位置)
    End(i64),
}

pub struct SileroVad {
    session: Session,
    threshold: f64,
    min_silence_samples: i64,
    speech_pad_samples: i64,
    // OnnxWrapper の状態
    state: Vec<f32>,
    context: Vec<f32>,
    // VADIterator の状態
    triggered: bool,
    temp_end: i64,
    current_sample: i64,
}

impl SileroVad {
    /// `min_silence_ms`: 発話終了と判定するまでの無音長。speech_pad_ms は VADIterator 既定の 30。
    pub fn new(threshold: f32, min_silence_ms: u32) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow!("ort セッションビルダ: {e}"))?
            .with_intra_threads(1)
            .map_err(|e| anyhow!("ort スレッド設定: {e}"))?
            .with_inter_threads(1)
            .map_err(|e| anyhow!("ort スレッド設定: {e}"))?
            .commit_from_memory(MODEL)
            .map_err(|e| anyhow!("silero_vad.onnx のロード失敗: {e}"))?;
        Ok(Self {
            session,
            threshold: threshold as f64,
            min_silence_samples: SR * min_silence_ms as i64 / 1000,
            speech_pad_samples: SR * 30 / 1000,
            state: vec![0.0; STATE_LEN],
            context: vec![0.0; CONTEXT],
            triggered: false,
            temp_end: 0,
            current_sample: 0,
        })
    }

    /// モデル状態と VADIterator 状態を初期化する(`reset_states` 相当)。
    pub fn reset(&mut self) {
        self.state.iter_mut().for_each(|v| *v = 0.0);
        self.context.iter_mut().for_each(|v| *v = 0.0);
        self.triggered = false;
        self.temp_end = 0;
        self.current_sample = 0;
    }

    /// 1 フレーム(`FRAME` サンプル)の発話確率。モデル状態(state/context)を進める。
    pub fn probability(&mut self, frame: &[f32]) -> Result<f32> {
        if frame.len() != FRAME {
            return Err(anyhow!(
                "フレーム長は {FRAME} サンプルが必要です(受け取り: {})",
                frame.len()
            ));
        }
        let mut x = Vec::with_capacity(CONTEXT + FRAME);
        x.extend_from_slice(&self.context);
        x.extend_from_slice(frame);

        let input = Tensor::from_array(([1usize, CONTEXT + FRAME], x.clone().into_boxed_slice()))
            .map_err(|e| anyhow!("{e}"))?;
        let state = Tensor::from_array(([2usize, 1, 128], self.state.clone().into_boxed_slice()))
            .map_err(|e| anyhow!("{e}"))?;
        let sr = Tensor::from_array((Vec::<usize>::new(), vec![SR].into_boxed_slice()))
            .map_err(|e| anyhow!("{e}"))?;
        let outputs = self
            .session
            .run(ort::inputs!["input" => input, "state" => state, "sr" => sr])
            .map_err(|e| anyhow!("silero 推論失敗: {e}"))?;
        let (_, prob) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("{e}"))?;
        let (_, new_state) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("{e}"))?;
        let p = *prob.first().ok_or_else(|| anyhow!("出力が空です"))?;
        if new_state.len() != STATE_LEN {
            return Err(anyhow!("state の形が不正です: {}", new_state.len()));
        }
        self.state.copy_from_slice(new_state);
        drop(outputs);
        self.context.copy_from_slice(&x[x.len() - CONTEXT..]);
        Ok(p)
    }

    /// `VADIterator.__call__` と同じ。確率から開始/終了イベントを判定する。
    pub fn try_process(&mut self, frame: &[f32]) -> Result<Option<VadEvent>> {
        let p = self.probability(frame)? as f64;
        Ok(self.step(p))
    }

    /// 推論に失敗したフレームは無かったものとして扱う(エラーは stderr へ)。
    pub fn process(&mut self, frame: &[f32]) -> Option<VadEvent> {
        match self.try_process(frame) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("[vad] {e}");
                None
            }
        }
    }

    /// 確率だけ与えて VADIterator の状態遷移を進める(推論を介さないテスト用にも公開)。
    pub fn step(&mut self, prob: f64) -> Option<VadEvent> {
        let window = FRAME as i64;
        self.current_sample += window;

        if prob >= self.threshold && self.temp_end != 0 {
            self.temp_end = 0;
        }

        if prob >= self.threshold && !self.triggered {
            self.triggered = true;
            let start = (self.current_sample - self.speech_pad_samples - window).max(0);
            return Some(VadEvent::Start(start));
        }

        if prob < self.threshold - 0.15 && self.triggered {
            if self.temp_end == 0 {
                self.temp_end = self.current_sample;
            }
            if self.current_sample - self.temp_end < self.min_silence_samples {
                return None;
            }
            let end = self.temp_end + self.speech_pad_samples - window;
            self.temp_end = 0;
            self.triggered = false;
            return Some(VadEvent::End(end));
        }
        None
    }
}
