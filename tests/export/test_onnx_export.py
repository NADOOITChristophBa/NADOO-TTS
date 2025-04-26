import sys
import os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if root not in sys.path:
    sys.path.insert(0, root)

import os
import pytest
import torch

from dia.model import Dia

@pytest.mark.skipif(
    os.getenv("CI") is not None, reason="ONNX-Export benötigt lokale Ressourcen"
)
def test_onnx_export(tmp_path):
    """
    Testet, ob das Dia-Modell nach ONNX exportiert werden kann.
    """
    model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")
    dummy_text = "[S1] ONNX Export Test. [S2]"
    dummy_input = (dummy_text,)
    onnx_path = tmp_path / "dia_test.onnx"

    try:
        torch.onnx.export(
            model.model,
            dummy_input,
            str(onnx_path),
            input_names=["text"],
            output_names=["output"],
            opset_version=17,
        )
    except Exception as e:
        pytest.fail(f"ONNX-Export fehlgeschlagen: {e}")
    assert onnx_path.exists(), "ONNX-Datei wurde nicht erstellt"
