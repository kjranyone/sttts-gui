"""TTS オーバーヘッド削減(precision 自動選択 / メモ化 / 参照 latent キャッシュ / ウォームアップ)。"""

import os
import queue

import pytest

from sttts_server.engines.tts_irodori import ModuleMemo, ref_cache_key, resolve_precision


@pytest.mark.parametrize(
    ("device", "requested", "cap", "expected"),
    [
        ("cuda", "auto", (7, 5), "fp32"),  # RTX 2080Ti (Turing)
        ("cuda", "auto", (8, 6), "bf16"),  # Ampere 以降
        ("cuda", "auto", None, "bf16"),
        ("cuda", "bf16", (7, 5), "bf16"),  # 明示指定は尊重
        ("xpu", "auto", None, "bf16"),  # 従来の XPU 挙動を維持
        ("cpu", "auto", None, "fp32"),
        ("cpu", "bf16", None, "fp32"),  # Irodori は CPU bf16 非対応
        ("mps", "auto", None, "fp32"),
    ],
)
def test_resolve_precision(device, requested, cap, expected):
    assert resolve_precision(device, requested, cap) == expected


torch = pytest.importorskip("torch")


class Enc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 3)
        self.calls = 0

    def forward(self, backbone, ids, mask):
        self.calls += 1
        x = torch.nn.functional.one_hot(ids, 4).float()
        return self.lin(x) * mask.unsqueeze(-1)


def test_module_memo_dedupes_identical_inputs():
    enc = Enc().eval()
    backbone = torch.nn.Identity()
    memo = ModuleMemo(enc, "text_encoder").install()
    ids = torch.tensor([[1, 2, 3, 0]])
    mask = torch.tensor([[True, True, True, False]])
    with torch.inference_mode():
        a = enc(backbone, ids, mask)
        b = enc(backbone, ids.clone(), mask.clone())  # 別オブジェクトでも内容が同じならヒット
        c = enc(backbone, torch.tensor([[1, 2, 2, 0]]), mask)
    assert enc.calls == 2
    assert a is b
    assert not torch.equal(a, c)
    assert (memo.hits, memo.misses) == (1, 2)
    memo.uninstall()
    with torch.inference_mode():
        enc(backbone, ids, mask)
    assert enc.calls == 3  # 解除後は素通し


def test_module_memo_matches_uncached_output_and_is_bounded():
    enc = Enc().eval()
    backbone = torch.nn.Identity()
    ids_list = [torch.tensor([[i % 4, (i + 1) % 4]]) for i in range(12)]
    mask = torch.ones(1, 2, dtype=torch.bool)
    with torch.inference_mode():
        ref = [enc(backbone, ids, mask) for ids in ids_list]
    memo = ModuleMemo(enc, "x", maxsize=3).install()
    with torch.inference_mode():
        got = [enc(backbone, ids, mask) for ids in ids_list]
    assert all(torch.equal(r, g) for r, g in zip(ref, got))
    assert len(memo.cache) <= 3


def test_module_memo_distinguishes_dtype_and_module_args():
    enc = Enc().eval()
    memo = ModuleMemo(enc, "x").install()
    ids = torch.tensor([[1, 2]])
    b1, b2 = torch.nn.Identity(), torch.nn.Identity()
    with torch.inference_mode():
        enc(b1, ids, torch.ones(1, 2, dtype=torch.bool))
        enc(b2, ids, torch.ones(1, 2, dtype=torch.bool))  # backbone が別インスタンス
        enc(b1, ids, torch.ones(1, 2, dtype=torch.float32))  # mask の dtype が違う
    assert memo.misses == 3


def test_ref_cache_key_changes_with_file_and_settings(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF0000")
    k1 = ref_cache_key(str(wav), codec_repo="c", trim_seconds=30.0)
    assert k1 == ref_cache_key(str(wav), codec_repo="c", trim_seconds=30.0)
    assert k1 != ref_cache_key(str(wav), codec_repo="c", trim_seconds=None)
    assert k1 != ref_cache_key(str(wav), codec_repo="other", trim_seconds=30.0)
    st = os.stat(wav)
    os.utime(wav, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    assert k1 != ref_cache_key(str(wav), codec_repo="c", trim_seconds=30.0)


def _drain(app):
    jobs = []
    while True:
        try:
            jobs.append(app._tts_queue.get_nowait())
        except queue.Empty:
            return jobs


def test_warmup_is_scheduled_once_per_model(mock_app):
    mock_app.configure({"type": "configure", "tts": {"model": "v4.1-small-mf"}})
    jobs = _drain(mock_app)
    assert len(jobs) == 1 and jobs[0].warmup
    mock_app.configure({"type": "configure", "tts": {"model": "v4.1-small-mf"}})
    assert _drain(mock_app) == []
    mock_app.configure({"type": "configure", "tts": {"model": "v4-large"}})
    assert [j.warmup for j in _drain(mock_app)] == [True]


def test_warmup_can_be_disabled(mock_app):
    mock_app.configure({"type": "configure", "tts": {"model": "v4.1-small-mf", "warmup": False}})
    assert _drain(mock_app) == []


def test_warmup_runs_without_emitting_audio(mock_app):
    import threading
    import time

    t = threading.Thread(target=mock_app._tts_worker, daemon=True)
    t.start()
    mock_app.configure({"type": "configure", "tts": {"model": "v4.1-small-mf"}})
    end = time.time() + 5
    while time.time() < end and not any("ウォームアップ完了" in m.get("message", "") for m in mock_app.of_type("log")):
        time.sleep(0.02)
    mock_app._tts_queue.put(None)
    t.join(2)
    assert any("ウォームアップ完了" in m.get("message", "") for m in mock_app.of_type("log"))
    assert mock_app.of_type("tts_audio") == []
    assert mock_app._tts_loaded_model == "v4.1-small-mf"
