import os
import numpy as np
import pytest

# Annahme: Es gibt eine Factory, die das Modell mit CTranslate2-Backend lädt (Platzhalter)
# from dia.model import DiaCTranslate2

@pytest.mark.skipif(
    os.getenv("CI") is not None, reason="CTranslate2-Backend noch nicht integriert"
)
def test_ctranslate2_inference_smoke(tmp_path):
    """
    Integrationstest: Kann das Modell mit CTranslate2 ein Audio erzeugen?
    """
    # Beispieltext
    text = "[S1] Test für CTranslate2-Inferenz. [S2] Das Backend wurde erfolgreich migriert."

    # Dummy-Integrationstest mit dem Dummy-Backend
    from dia.ctranslate2 import DiaCTranslate2
    model = DiaCTranslate2.from_pretrained("nari-labs/Dia-1.6B", backend="ctranslate2-metal")
    output = model.generate(text)
    output_path = tmp_path / "test_output.wav"
    model.save_audio(output_path, output)
    assert output_path.exists(), "Dummy-Audio-Datei wurde nicht erzeugt"
