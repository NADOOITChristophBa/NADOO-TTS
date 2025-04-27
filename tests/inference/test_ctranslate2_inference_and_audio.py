import os
import tempfile
import shutil
import pytest
from dia.ctranslate2 import DiaCTranslate2

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

def dummy_tokenizer(text):
    # Platzhalter: Ersetze durch echten Tokenizer, falls verfügbar
    return text.split()

@pytest.mark.skipif(os.getenv("CI") is not None, reason="Test benötigt Internetzugang und lokale Ressourcen")
@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio nicht verfügbar")
def test_ctranslate2_inference_and_audio():
    """
    End-to-End-Test: Lädt/konvertiert Modell, führt Inferenz durch, speichert und prüft Audio.
    """
    tmp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(tmp_dir, "ct2-dia-infer")
    audio_path = os.path.join(tmp_dir, "test_audio.wav")
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    # Modell laden (inkl. Auto-Download/-Konvertierung)
    model = DiaCTranslate2.from_pretrained(
        model_dir,
        backend="cpu",
        hf_model_name="nari-labs/Dia-1.6B"
    )
    # Inferenz (Dummy-Tokenizer, kurzer Testtext)
    audio = model.generate("Hello world!", tokenizer=dummy_tokenizer)
    assert audio is not None, "Audio darf nicht None sein!"
    # Audio speichern
    model.save_audio(audio_path, audio)
    assert os.path.exists(audio_path), "Audio-Datei wurde nicht erzeugt!"
    # Prüfe, ob die Datei mit torchaudio geladen werden kann und ein paar Samples enthält
    waveform, sr = torchaudio.load(audio_path)
    assert waveform.numel() > 0, "Audio-Datei ist leer!"
    shutil.rmtree(tmp_dir)
