import os
import tempfile
import shutil
import pytest
from dia.ctranslate2 import DiaCTranslate2

@pytest.mark.skipif(os.getenv("CI") is not None, reason="Test benötigt lokale Ressourcen")
def test_ctranslate2_invalid_model_name():
    """
    Testet defensiv, dass bei ungültigem Modellnamen ein sauberer Fehler geloggt/geworfen wird.
    """
    tmp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(tmp_dir, "ct2-invalid")
    invalid_hf_model = "nari-labs/NonExistentModelXYZ"
    try:
        with pytest.raises(Exception):
            DiaCTranslate2.from_pretrained(
                model_dir,
                backend="cpu",
                hf_model_name=invalid_hf_model
            )
    finally:
        shutil.rmtree(tmp_dir)

@pytest.mark.skipif(os.getenv("CI") is not None, reason="Test benötigt lokale Ressourcen")
def test_ctranslate2_missing_converter(monkeypatch):
    """
    Testet defensiv, dass ein sauberer Fehler geworfen wird, wenn ct2-transformers-converter fehlt.
    """
    tmp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(tmp_dir, "ct2-missing-tool")
    def fake_run(*args, **kwargs):
        if "ct2-transformers-converter" in str(args):
            raise FileNotFoundError("ct2-transformers-converter not found!")
        import subprocess
        return subprocess.run(*args, **kwargs)
    monkeypatch.setattr("subprocess.run", fake_run)
    try:
        with pytest.raises(Exception):
            DiaCTranslate2.from_pretrained(
                model_dir,
                backend="cpu",
                hf_model_name="nari-labs/Dia-1.6B"
            )
    finally:
        shutil.rmtree(tmp_dir)
