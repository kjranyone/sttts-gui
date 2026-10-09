"""ASR プリロードの作り直し/流用判定のテスト(実モデル不要。create_asr を fake に差し替える)。

GUI から API キーを変えたときにエンジンが作り直されること、切替後のプリロードが失敗・
ロード中のときに旧エンジンがマイク開始で流用されないことを確認する。
"""

import threading
import time

import pytest

import sttts_server.app as app_mod

from .conftest import RecordingApp


class FakeAsr:
    def __init__(self, cfg: dict, gate: threading.Event | None, fail: bool) -> None:
        self.cfg = dict(cfg)
        self.gate = gate
        self.fail = fail
        self.unloaded = False
        self.model_id = f"fake-{cfg.get('engine')}-{cfg.get('gemini_api_key')}"

    def load(self, progress=None) -> None:
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail:
            raise RuntimeError("Gemini API キーがありません")

    def unload(self) -> None:
        self.unloaded = True


class FakeFactory:
    """create_asr の代替。生成したエンジンを記録し、キー単位でゲート/失敗を仕込める。"""

    def __init__(self) -> None:
        self.created: list[FakeAsr] = []
        self.gates: dict[str | None, threading.Event] = {}
        self.fail_keys: set[str | None] = set()

    def __call__(self, cfg: dict) -> FakeAsr:
        key = cfg.get("gemini_api_key")
        engine = FakeAsr(cfg, self.gates.get(key), key in self.fail_keys)
        self.created.append(engine)
        return engine


@pytest.fixture
def real_app(tmp_path, monkeypatch):
    factory = FakeFactory()
    monkeypatch.setattr(app_mod, "create_asr", factory)
    app = RecordingApp(mock=False, output_dir=str(tmp_path / "out"), load_user_file=False)
    yield app, factory
    app._stop.set()


def _wait(cond, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.01)
    raise AssertionError("timeout")


def _configure_asr(app, **asr) -> None:
    app.configure({"type": "configure", "asr": asr})


def _injectable(app):
    """start_session が流用するエンジン(現設定で作られたもののみ)。"""
    with app._asr_lock:
        return app._asr_engine if app._asr_engine_key == app._asr_config_key() else None


def test_api_key_change_rebuilds_gemini_engine(real_app):
    app, factory = real_app
    _configure_asr(app, engine="gemini", gemini_api_key="key-a")
    _wait(lambda: _injectable(app) is not None)
    first = _injectable(app)
    assert first.cfg["gemini_api_key"] == "key-a"

    _configure_asr(app, gemini_api_key="key-b")
    _wait(lambda: _injectable(app) is not None and _injectable(app) is not first)
    assert _injectable(app).cfg["gemini_api_key"] == "key-b"
    assert first.unloaded

    # 同じキーの再送では作り直さない
    count = len(factory.created)
    _configure_asr(app, gemini_api_key="key-b")
    time.sleep(0.1)
    assert len(factory.created) == count


def test_failed_switch_does_not_reuse_previous_engine(real_app):
    app, factory = real_app
    _configure_asr(app, engine="nemotron")
    _wait(lambda: _injectable(app) is not None)

    # キー未設定の Gemini へ切替 → プリロード失敗。旧(nemotron)は流用されない
    factory.fail_keys.add(None)
    _configure_asr(app, engine="gemini")
    _wait(lambda: app._asr_phase == app_mod.ERROR)
    assert _injectable(app) is None

    # キーを入れると再試行され、使えるようになる
    _configure_asr(app, gemini_api_key="key-a")
    _wait(lambda: _injectable(app) is not None)
    assert _injectable(app).cfg["engine"] == "gemini"


def test_stale_preload_result_is_discarded(real_app):
    app, factory = real_app
    slow = threading.Event()
    factory.gates["key-slow"] = slow
    _configure_asr(app, engine="gemini", gemini_api_key="key-slow")
    _wait(lambda: len(factory.created) == 1)

    # ロード中にキーが変わる。旧設定(key-slow)のエンジンは流用されない
    # (新設定のプリロードは並行に走るので、完了していれば key-fast のものだけが使える)
    _configure_asr(app, gemini_api_key="key-fast")
    current = _injectable(app)
    assert current is None or current.cfg["gemini_api_key"] == "key-fast"

    # 先発が完了しても採用されず解放され、後発(現設定)が採用される。
    slow.set()
    _wait(lambda: _injectable(app) is not None)
    assert _injectable(app).cfg["gemini_api_key"] == "key-fast"
    _wait(lambda: factory.created[0].unloaded)
