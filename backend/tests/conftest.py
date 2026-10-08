import pytest

from sttts_server.app import BackendApp


class RecordingApp(BackendApp):
    """send() を記録する BackendApp(stdout へは書かない)。"""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.sent: list[dict] = []

    def send(self, msg: dict) -> None:
        self.sent.append(msg)

    def of_type(self, mtype: str) -> list[dict]:
        return [m for m in self.sent if m.get("type") == mtype]


@pytest.fixture
def mock_app(tmp_path):
    app = RecordingApp(mock=True, output_dir=str(tmp_path / "out"), load_user_file=False)
    yield app
    app._stop.set()
