from concurrent.futures import Future

import numpy as np

from sttts_server.performance import AcousticObservation, Delivery, Emotion2VecClassifier, observe, plan_delivery


def test_acoustic_observation_and_conservative_delivery():
    audio = np.concatenate(
        [np.full(16000, 0.08, dtype=np.float32), np.zeros(4000, dtype=np.float32),
         np.full(16000, 0.08, dtype=np.float32)]
    )
    obs = observe(audio, "おはようございます")
    assert obs.audio_ms == 2250
    assert obs.pause_ms >= 200
    assert obs.active_ms >= 1900
    assert plan_delivery(obs).emoji == ""
    assert plan_delivery(obs, emotion="neutral").emoji == ""
    assert plan_delivery(obs, emotion="happy").emoji == "😊"
    slow_pauses = AcousticObservation(3500, 2600, 700, 0.08)
    assert plan_delivery(slow_pauses).style == "間を取りながら"
    assert plan_delivery(slow_pauses, emotion="happy").style == "楽しげに、間を取りながら"


def test_delivery_preserves_text_and_voice_caption():
    delivery = Delivery(emoji="😠", style="不満げに", duration_scale=0.9)
    assert delivery.annotated_text("やめてください") == "😠やめてください"
    assert delivery.annotated_text("😠やめてください") == "😠やめてください"
    assert delivery.caption("落ち着いた声") == "落ち着いた声。話し方は不満げに。"
    assert Delivery.from_mapping({"emoji": "🤐", "duration_scale": 9}).emoji == ""
    assert Delivery.from_mapping({"duration_scale": 9}).duration_scale == 1.15
    assert Delivery.from_mapping({"duration_scale": float("nan")}).duration_scale is None


def test_emotion2vec_bilingual_labels_and_numpy_scores():
    class FakeModel:
        def generate(self, **_):
            return [{"labels": ["中立/neutral", "开心/happy"],
                     "scores": np.array([0.08, 0.92])}]

    classifier = object.__new__(Emotion2VecClassifier)
    classifier._model = FakeModel()
    assert classifier.classify(np.zeros(1600, dtype=np.float32)) == "happy"


def test_auto_speak_sends_separate_delivery_to_irodori(mock_app):
    app = mock_app
    app.config["voice"]["caption"] = "落ち着いた声"
    app.config["voice"]["ref_wavs"] = ["target.wav"]
    app.config["pipeline"]["performance_wait_ms"] = 0
    future = Future()
    future.set_result((AcousticObservation(1200, 1100, 0, 0.08), "happy"))
    app._performance_pending[7] = future

    app.on_asr_final(7, "こんにちは。")
    final = app.of_type("asr_final")[-1]
    accepted = app.of_type("speak_accepted")[-1]
    job = app._tts_queue.get_nowait()

    assert final["text"] == "こんにちは。"
    assert final["delivery"]["emoji"] == "😊"
    assert accepted["delivery"]["emoji"] == "😊"
    assert job.text == "😊こんにちは。"
    assert job.caption == "落ち着いた声。話し方は楽しげに。"
    assert job.ref_wavs == ["target.wav"]
    app._performance_pool.shutdown(wait=True)


def test_expression_deadline_does_not_delay_speech(mock_app):
    app = mock_app
    app.config["pipeline"]["performance_wait_ms"] = 0
    app._performance_pending[8] = Future()
    app.on_asr_final(8, "大丈夫です。")
    assert app.of_type("asr_final")[-1]["delivery"] is None
    assert app._tts_queue.get_nowait().text == "大丈夫です。"
    app._performance_pool.shutdown(wait=True, cancel_futures=True)
