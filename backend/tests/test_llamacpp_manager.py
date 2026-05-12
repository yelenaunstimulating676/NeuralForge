"""
Test del LlamaCppManager.

Mockiamo urllib.request.urlopen così non scarichiamo davvero da internet.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.export.llamacpp_manager import (
    LlamaCppError,
    LlamaCppManager,
    LlamaCppPaths,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_fake_zip_response(files: dict[str, bytes]) -> MagicMock:
    """Crea un mock di urlopen() che ritorna un ZIP in memoria."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    zip_bytes = buf.getvalue()

    resp = MagicMock()
    resp.headers = {"Content-Length": str(len(zip_bytes))}
    resp.read = io.BytesIO(zip_bytes).read
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=None)
    return resp


def make_fake_text_response(content: bytes) -> MagicMock:
    resp = MagicMock()
    resp.headers = {"Content-Length": str(len(content))}
    resp.read = io.BytesIO(content).read
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=None)
    return resp


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


class TestPaths:
    def test_root_dir_uses_home(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        mgr = LlamaCppManager(version="b9999")
        assert mgr.root_dir == tmp_path / ".neuralforge" / "llamacpp" / "b9999"

    def test_get_paths_windows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr("platform.system", lambda: "Windows")
        mgr = LlamaCppManager(version="b1")
        paths = mgr.get_paths()
        assert paths.quantize_bin.name == "llama-quantize.exe"
        assert paths.convert_script.name == "convert_hf_to_gguf.py"

    def test_get_paths_linux(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr("platform.system", lambda: "Linux")
        mgr = LlamaCppManager(version="b1")
        paths = mgr.get_paths()
        assert paths.quantize_bin.name == "llama-quantize"


# ---------------------------------------------------------------------------
# is_installed
# ---------------------------------------------------------------------------


class TestIsInstalled:
    def test_false_when_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        mgr = LlamaCppManager(version="b1")
        assert mgr.is_installed() is False

    def test_true_when_all_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr("platform.system", lambda: "Windows")
        mgr = LlamaCppManager(version="b1")
        # Crea i due file fittizi
        paths = mgr.get_paths()
        paths.root_dir.mkdir(parents=True)
        paths.quantize_bin.write_bytes(b"fake binary")
        paths.convert_script.write_text("# fake script")
        assert mgr.is_installed() is True


# ---------------------------------------------------------------------------
# ensure_installed (cache hit)
# ---------------------------------------------------------------------------


class TestEnsureInstalledCacheHit:
    def test_already_installed_no_download(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr("platform.system", lambda: "Windows")
        mgr = LlamaCppManager(version="b1")
        # Pre-installa tutto
        paths = mgr.get_paths()
        paths.root_dir.mkdir(parents=True)
        paths.quantize_bin.write_bytes(b"x")
        paths.convert_script.write_text("x")

        # Spia urlopen: NON deve essere chiamato
        with patch("urllib.request.urlopen") as mock_url:
            result = mgr.ensure_installed()
            mock_url.assert_not_called()
        assert result.all_exist()


# ---------------------------------------------------------------------------
# ensure_installed (download)
# ---------------------------------------------------------------------------


class TestEnsureInstalledDownload:
    def test_full_download_flow(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr("platform.system", lambda: "Windows")
        mgr = LlamaCppManager(version="b1")

        # Fake ZIP che contiene llama-quantize.exe in subdirectory tipica
        fake_zip = make_fake_zip_response({
            "build/bin/llama-quantize.exe": b"FAKE BINARY DATA",
            "build/bin/main.exe": b"OTHER",
        })
        fake_script = make_fake_text_response(b"# convert script")

        with patch(
            "urllib.request.urlopen",
            side_effect=[fake_zip, fake_script],
        ):
            paths = mgr.ensure_installed()

        assert paths.all_exist()
        assert paths.quantize_bin.read_bytes() == b"FAKE BINARY DATA"
        assert paths.convert_script.read_text() == "# convert script"

    def test_download_only_script_if_binary_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr("platform.system", lambda: "Windows")
        mgr = LlamaCppManager(version="b1")

        # Pre-installa solo il binario
        paths_before = mgr.get_paths()
        paths_before.root_dir.mkdir(parents=True)
        paths_before.quantize_bin.write_bytes(b"BINARY")

        fake_script = make_fake_text_response(b"# script")

        with patch("urllib.request.urlopen", return_value=fake_script) as mock_url:
            mgr.ensure_installed()
            # Una sola chiamata (per lo script)
            assert mock_url.call_count == 1

    def test_zip_missing_quantize_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr("platform.system", lambda: "Windows")
        mgr = LlamaCppManager(version="b1")

        # ZIP senza llama-quantize.exe
        fake_zip = make_fake_zip_response({"build/bin/main.exe": b"OTHER"})

        with patch("urllib.request.urlopen", return_value=fake_zip):
            with pytest.raises(LlamaCppError, match="non trovato"):
                mgr.ensure_installed()

    def test_progress_callback_called(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr("platform.system", lambda: "Windows")
        mgr = LlamaCppManager(version="b1")

        fake_zip = make_fake_zip_response({"bin/llama-quantize.exe": b"X"})
        fake_script = make_fake_text_response(b"x")

        calls = []
        def cb(stage, pct):
            calls.append((stage, pct))

        with patch(
            "urllib.request.urlopen",
            side_effect=[fake_zip, fake_script],
        ):
            mgr.ensure_installed(progress_callback=cb)

        # Almeno una chiamata per ogni stage
        stages = {c[0] for c in calls}
        assert "downloading_binaries" in stages
        assert "downloading_script" in stages

    def test_download_failure_no_partial_file(self, tmp_path, monkeypatch):
        """Se download fallisce a metà, NON deve restare file parziale."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        mgr = LlamaCppManager(version="b1")

        def fail_urlopen(*args, **kwargs):
            raise ConnectionError("network down")

        with patch("urllib.request.urlopen", side_effect=fail_urlopen):
            with pytest.raises(LlamaCppError):
                mgr.ensure_installed()

        # Nessun file .tmp residuo
        if mgr.root_dir.exists():
            assert not any(p.suffix == ".tmp" for p in mgr.root_dir.iterdir())


# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------


class TestUninstall:
    def test_uninstall_existing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        mgr = LlamaCppManager(version="b1")
        mgr.root_dir.mkdir(parents=True)
        (mgr.root_dir / "test.txt").write_text("x")

        assert mgr.uninstall() is True
        assert not mgr.root_dir.exists()

    def test_uninstall_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        mgr = LlamaCppManager(version="b1")
        assert mgr.uninstall() is False