"""HF snapshot の symlink を実ファイルへ展開する(onnxruntime の外部データ検証対策)。"""

import os

import pytest

from sttts_server.engines.asr_nemotron_onnx import _materialize_snapshot


def test_materialize_snapshot_resolves_symlinks(tmp_path):
    blobs = tmp_path / "blobs"
    snap = tmp_path / "snapshots" / "rev1"
    blobs.mkdir()
    snap.mkdir(parents=True)
    (blobs / "aaa").write_bytes(b"graph")
    (blobs / "bbb").write_bytes(b"weights")
    try:
        os.symlink(blobs / "aaa", snap / "enc.onnx")
        os.symlink(blobs / "bbb", snap / "enc.onnx.data")
    except OSError:
        pytest.skip("symlink 権限なし")

    out = _materialize_snapshot(snap)

    from pathlib import Path

    for name, data in (("enc.onnx", b"graph"), ("enc.onnx.data", b"weights")):
        f = Path(out) / name
        assert f.read_bytes() == data
        assert not f.is_symlink()
        assert f.resolve().parent == Path(out).resolve()

    # 冪等
    assert _materialize_snapshot(snap) == out
