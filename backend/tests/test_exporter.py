"""
Test della pipeline di export (mockata, no GPU, no subprocess reale).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.export.exporter import (
    DEFAULT_QUANTIZATION,
    VALID_QUANTIZATIONS,
    ExportError,
    check_disk_space,
    validate_quantization,
)
from core.export.llamacpp_manager import LlamaCppPaths


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


class TestValidateQuantization:
    def test_default_valid(self):
        assert validate_quantization(DEFAULT_QUANTIZATION) == DEFAULT_QUANTIZATION

    def test_lowercase_normalized(self):
        assert validate_quantization("q4_k_m") == "Q4_K_M"

    def test_strip(self):
        assert validate_quantization("  Q5_K_M  ") == "Q5_K_M"

    def test_invalid_raises(self):
        with pytest.raises(ExportError, match="invalida"):
            validate_quantization("Q99_X")

    def test_all_documented_are_valid(self):
        """Sanity: assicura che tutti i formati elencati siano accettati."""
        for q in VALID_QUANTIZATIONS:
            assert validate_quantization(q) == q


# ---------------------------------------------------------------------------
# Disk space check
# ---------------------------------------------------------------------------


class TestDiskSpace:
    def test_enough_space(self, tmp_path):
        # Niente sollevato per 1 byte richiesto
        check_disk_space(needed_bytes=1, target_dir=tmp_path)

    def test_not_enough_raises(self, tmp_path):
        # Richiedi 10 PB (10^16 byte) → sicuramente non c'è
        with pytest.raises(ExportError, match="Spazio insufficiente"):
            check_disk_space(
                needed_bytes=10 ** 16, target_dir=tmp_path
            )


# ---------------------------------------------------------------------------
# Subprocess calls (mocked)
# ---------------------------------------------------------------------------


def make_fake_llamacpp_paths(tmp_path: Path) -> LlamaCppPaths:
    """Crea path fittizi con file marker."""
    root = tmp_path / "llamacpp"
    root.mkdir()
    quantize = root / "llama-quantize.exe"
    quantize.write_bytes(b"FAKE")
    script = root / "convert_hf_to_gguf.py"
    script.write_text("# fake")
    return LlamaCppPaths(
        root_dir=root, quantize_bin=quantize, convert_script=script
    )


class TestConvertToGguf:
    def test_success(self, tmp_path):
        from core.export.exporter import convert_to_gguf

        paths = make_fake_llamacpp_paths(tmp_path)
        merged_dir = tmp_path / "merged"
        merged_dir.mkdir()
        output = tmp_path / "out.gguf"

        # Mock subprocess.run con success E creazione file output
        def fake_run(cmd, **kwargs):
            output.write_bytes(b"GGUF FAKE DATA")
            result = MagicMock()
            result.returncode = 0
            result.stdout = "ok"
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=fake_run):
            result = convert_to_gguf(merged_dir, output, paths)
        assert result == output
        assert output.exists()

    def test_subprocess_failure(self, tmp_path):
        from core.export.exporter import convert_to_gguf

        paths = make_fake_llamacpp_paths(tmp_path)
        merged_dir = tmp_path / "merged"
        merged_dir.mkdir()
        output = tmp_path / "out.gguf"

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 1
            result.stdout = ""
            result.stderr = "boom"
            return result

        with patch("subprocess.run", side_effect=fake_run):
            with pytest.raises(ExportError, match="exit 1"):
                convert_to_gguf(merged_dir, output, paths)

    def test_missing_script_raises(self, tmp_path):
        from core.export.exporter import convert_to_gguf

        # Path che NON esistono
        paths = LlamaCppPaths(
            root_dir=tmp_path,
            quantize_bin=tmp_path / "missing.exe",
            convert_script=tmp_path / "missing.py",
        )
        with pytest.raises(ExportError, match="non trovato"):
            convert_to_gguf(
                tmp_path / "merged", tmp_path / "out.gguf", paths
            )


class TestQuantizeGguf:
    def test_success(self, tmp_path):
        from core.export.exporter import quantize_gguf

        paths = make_fake_llamacpp_paths(tmp_path)
        input_gguf = tmp_path / "input.gguf"
        input_gguf.write_bytes(b"F16 DATA")
        output = tmp_path / "out_q4.gguf"

        def fake_run(cmd, **kwargs):
            output.write_bytes(b"Q4 DATA")
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=fake_run):
            result = quantize_gguf(
                input_gguf, output, "Q4_K_M", paths
            )
        assert result == output

    def test_invalid_quant_raises(self, tmp_path):
        from core.export.exporter import quantize_gguf

        paths = make_fake_llamacpp_paths(tmp_path)
        with pytest.raises(ExportError, match="invalida"):
            quantize_gguf(
                tmp_path / "a.gguf",
                tmp_path / "b.gguf",
                "Q99_BOGUS",
                paths,
            )


# ---------------------------------------------------------------------------
# Pipeline completa (heavy mock)
# ---------------------------------------------------------------------------


class TestExportPipeline:
    def test_end_to_end_with_mocks(self, tmp_path, monkeypatch):
        from core.export import exporter

        # Mock llamacpp_manager.ensure_installed
        fake_paths = make_fake_llamacpp_paths(tmp_path)
        monkeypatch.setattr(
            exporter.llamacpp_manager,
            "ensure_installed",
            lambda progress_callback=None: fake_paths,
        )

        base_path = tmp_path / "base"
        base_path.mkdir()
        adapter_path = tmp_path / "adapter"
        adapter_path.mkdir()
        output_path = tmp_path / "out" / "model.gguf"

        # Mock merge_lora per non caricare modelli reali
        def fake_merge(base_model_path, adapter_path, output_dir, progress_callback=None):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "config.json").write_text("{}")
            return output_dir

        monkeypatch.setattr(exporter, "merge_lora", fake_merge)

        # Mock subprocess.run per convert E quantize.
        # Devono creare i file output dichiarati.
        call_count = {"n": 0}
        def fake_run(cmd, **kwargs):
            call_count["n"] += 1
            # Trova il path output dal cmd
            for i, arg in enumerate(cmd):
                if str(arg).endswith(".gguf"):
                    Path(arg).parent.mkdir(parents=True, exist_ok=True)
                    Path(arg).write_bytes(b"GGUF FAKE")
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=fake_run):
            result = exporter.export_ft_to_gguf(
                base_model_path=base_path,
                adapter_path=adapter_path,
                output_path=output_path,
                quantization="Q4_K_M",
            )

        assert result.output_path == output_path
        assert result.quantization == "Q4_K_M"
        assert result.size_bytes > 0
        assert result.elapsed_seconds >= 0
        # subprocess.run chiamato due volte (convert + quantize)
        assert call_count["n"] == 2

    def test_workdir_cleaned_up(self, tmp_path, monkeypatch):
        from core.export import exporter

        fake_paths = make_fake_llamacpp_paths(tmp_path)
        monkeypatch.setattr(
            exporter.llamacpp_manager,
            "ensure_installed",
            lambda progress_callback=None: fake_paths,
        )

        workdir = tmp_path / "workdir"

        def fake_merge(base_model_path, adapter_path, output_dir, progress_callback=None):
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir

        monkeypatch.setattr(exporter, "merge_lora", fake_merge)

        def fake_run(cmd, **kwargs):
            for arg in cmd:
                if str(arg).endswith(".gguf"):
                    Path(arg).parent.mkdir(parents=True, exist_ok=True)
                    Path(arg).write_bytes(b"x")
            r = MagicMock()
            r.returncode = 0; r.stdout = ""; r.stderr = ""
            return r

        with patch("subprocess.run", side_effect=fake_run):
            exporter.export_ft_to_gguf(
                base_model_path=tmp_path / "base",
                adapter_path=tmp_path / "adapter",
                output_path=tmp_path / "out.gguf",
                workdir=workdir,
            )

        # Workdir rimosso dopo il run
        assert not workdir.exists()