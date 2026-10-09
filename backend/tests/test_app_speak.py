import queue


def _drain(app):
    jobs = []
    while True:
        try:
            jobs.append(app._tts_queue.get_nowait())
        except queue.Empty:
            return jobs


def test_random_seed_is_shared_by_all_chunks_of_a_request(mock_app):
    mock_app.speak({"text": "こんにちは、今日はいい天気ですね。明日も晴れるといいですね。"})
    jobs = _drain(mock_app)
    assert len(jobs) >= 2
    seeds = {j.seed for j in jobs}
    assert len(seeds) == 1 and None not in seeds


def test_explicit_seed_is_kept(mock_app):
    mock_app.speak({"text": "一文目です。二文目もあります。", "seed": 1234})
    assert {j.seed for j in _drain(mock_app)} == {1234}


def test_each_request_gets_its_own_random_seed(mock_app):
    seeds = set()
    for _ in range(5):
        mock_app.speak({"text": "テストです。"})
        seeds |= {j.seed for j in _drain(mock_app)}
    assert len(seeds) >= 4  # 31bit 乱数なので衝突はまず起きない


def test_speak_uses_pipeline_chunk_config(mock_app):
    mock_app.config["pipeline"]["first_chunk_mora_max"] = 0
    mock_app.speak({"text": "こんにちは、今日はいい天気ですね。"})
    assert [j.text for j in _drain(mock_app)] == ["こんにちは、今日はいい天気ですね。"]
    mock_app.config["pipeline"]["first_chunk_mora_max"] = 12
    mock_app.speak({"text": "こんにちは、今日はいい天気ですね。"})
    assert [j.text for j in _drain(mock_app)] == ["こんにちは、", "今日はいい天気ですね。"]


def test_device_lost_marks_tts_error_and_rejects_further_speaks(mock_app):
    from sttts_server.protocol import ERROR

    def boom(job):
        raise RuntimeError("level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)")

    mock_app._synthesize = boom
    mock_app.speak({"text": "テストです。"})
    for job in _drain(mock_app):
        mock_app._run_job(job)

    assert mock_app._tts_phase == ERROR
    errors = mock_app.of_type("error")
    assert errors and errors[-1]["recoverable"] is False
    states = mock_app.of_type("state")
    assert states[-1]["tts"]["phase"] == "error"

    # 以降の発話は受け付けず(speak_accepted を出さず)、キューにも積まない
    before = len(mock_app.of_type("speak_accepted"))
    mock_app.speak({"text": "もう一度です。"})
    assert len(mock_app.of_type("speak_accepted")) == before
    assert _drain(mock_app) == []


def test_ordinary_synthesis_failure_stays_recoverable(mock_app):
    from sttts_server.protocol import ERROR

    def boom(job):
        raise RuntimeError("something odd")

    mock_app._synthesize = boom
    mock_app.speak({"text": "テストです。"})
    for job in _drain(mock_app):
        mock_app._run_job(job)
    assert mock_app._tts_phase != ERROR
    assert mock_app.of_type("error")[-1]["recoverable"] is True
