import os
import tempfile
import shutil
import pytest
from dia.ctranslate2 import DiaCTranslate2

@pytest.mark.skipif(os.getenv("CI") is not None, reason="Test benötigt Internetzugang und lokale Ressourcen")
def test_ctranslate2_auto_download_and_conversion():
    """
    End-to-End-Test für die automatische Modellbereitstellung:
    - Löscht ggf. das Zielverzeichnis
    - Prüft, ob Download, ONNX-Export und CTranslate2-Konvertierung automatisch erfolgen
    """
    tmp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(tmp_dir, "ct2-test-auto")
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    # Test: Modellverzeichnis existiert nicht -> Automatisierung sollte greifen
    model = DiaCTranslate2.from_pretrained(
        model_dir,
        backend="cpu",
        hf_model_name="nari-labs/Dia-1.6B"
    )
    assert os.path.exists(model_dir), "CTranslate2-Modellverzeichnis wurde nicht erzeugt!"
    # Optional: weitere Checks, z.B. ob model.model geladen ist
    shutil.rmtree(tmp_dir)
