import sys
import os

def test_import_dia():
    # Projekt-Root zum sys.path hinzufügen
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from dia.model import Dia
    except Exception as e:
        assert False, f"Import von dia.model.Dia fehlgeschlagen: {e}"
