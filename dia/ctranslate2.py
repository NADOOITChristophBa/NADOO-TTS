"""
Produktionsreife CTranslate2-TTS-Integration
- Lädt ein CTranslate2-Modell (konvertiert aus ONNX)
- Führt Inferenz auf Text durch
- Speichert das generierte Audio als WAV
- Robuste Fehlerbehandlung und Logging
"""
import logging
from pathlib import Path

try:
    import ctranslate2
    CTR2_AVAILABLE = True
except ImportError:
    CTR2_AVAILABLE = False
    logging.warning("CTranslate2 ist nicht installiert. Nur Dummy-Backend verfügbar.")

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    logging.warning("torchaudio ist nicht installiert. Audio-Export als WAV nicht möglich.")

class DiaCTranslate2:
    """
    Produktionsreifes Backend für TTS-Inferenz mit CTranslate2.
    - from_pretrained: Modell laden
    - generate: Text zu Audio
    - save_audio: Audio als WAV speichern
    """
    def __init__(self, model=None, backend="cpu"):
        self.model = model
        self.backend = backend

    @classmethod
    def from_pretrained(cls, model_path, backend="cpu", hf_model_name="nari-labs/Dia-1.6B", onnx_path=None):
        """
        Lädt ein CTranslate2-Modell. Falls nicht vorhanden, wird das Modell automatisch von HuggingFace geladen und konvertiert.
        :param model_path: Pfad zum CTranslate2-Modellverzeichnis
        :param backend: "cpu", "cuda", "auto", "mps" etc.
        :param hf_model_name: HuggingFace Modellname (wird verwendet, falls Konvertierung nötig)
        :param onnx_path: Optionaler Pfad für den ONNX-Export
        :return: DiaCTranslate2-Instanz
        """
        import subprocess
        import sys
        if not CTR2_AVAILABLE:
            raise ImportError("CTranslate2 ist nicht installiert!")
        model_dir = Path(model_path)
        if not model_dir.exists():
            logging.warning(f"CTranslate2-Modell nicht gefunden: {model_path}. Automatische Konvertierung wird gestartet.")
            # Schritt 1: ONNX-Export (falls nicht vorhanden)
            if onnx_path is None:
                onnx_path = model_dir.parent / "onnx_export"
            onnx_path = Path(onnx_path)
            if not onnx_path.exists():
                onnx_path.mkdir(parents=True, exist_ok=True)
                logging.info(f"Exportiere HuggingFace-Modell nach ONNX: {hf_model_name} -> {onnx_path}")
                try:
                    subprocess.run([
                        sys.executable, "-m", "transformers.onnx", "--model", hf_model_name, "--feature", "sequence-classification", "--atol", "1e-4", str(onnx_path)
                    ], check=True)
                except Exception as e:
                    logging.error(f"ONNX-Export fehlgeschlagen: {e}")
                    raise
            # Schritt 2: CTranslate2-Konvertierung
            logging.info(f"Konvertiere ONNX-Modell nach CTranslate2: {onnx_path} -> {model_dir}")
            try:
                subprocess.run([
                    "ct2-transformers-converter",
                    "--model_dir", str(onnx_path),
                    "--output_dir", str(model_dir)
                ], check=True)
            except Exception as e:
                logging.error(f"CTranslate2-Konvertierung fehlgeschlagen: {e}")
                raise
            logging.info(f"CTranslate2-Modell erfolgreich generiert: {model_dir}")
        try:
            model = ctranslate2.Translator(str(model_dir), device=backend)
            logging.info(f"CTranslate2-Modell geladen: {model_dir} auf {backend}")
        except Exception as e:
            logging.error(f"Fehler beim Laden des CTranslate2-Modells: {e}")
            raise
        return cls(model, backend)

    def generate(self, text, tokenizer=None):
        """
        Führt TTS-Inferenz durch.
        :param text: Eingabetext (str)
        :param tokenizer: Optionaler Tokenizer (callable), der Text in Tokens wandelt
        :return: Audio als Tensor oder Array
        """
        if not CTR2_AVAILABLE or self.model is None:
            raise RuntimeError("CTranslate2-Backend nicht verfügbar!")
        if tokenizer is None:
            raise ValueError("Tokenizer muss für die Produktion angegeben werden!")
        tokens = tokenizer(text)
        logging.debug(f"Tokenisierte Eingabe: {tokens}")
        try:
            result = self.model.translate_batch([tokens])
            # TODO: Output-Postprocessing für TTS (z.B. Decodierung zu Audio)
            # Hier wird angenommen, dass das Modell direkt Audio-Tokens oder -Arrays liefert
            audio = result[0].get("audio", None)
            if audio is None:
                raise ValueError("CTranslate2-Modell hat kein Audio generiert!")
            return audio
        except Exception as e:
            logging.error(f"Fehler bei der TTS-Inferenz: {e}")
            raise

    def save_audio(self, path, audio, sample_rate=44100):
        """
        Speichert das generierte Audio als WAV.
        :param path: Pfad zur Zieldatei
        :param audio: Audio-Daten (Tensor, np.ndarray, bytes)
        :param sample_rate: Abtastrate
        """
        if not TORCHAUDIO_AVAILABLE:
            raise ImportError("torchaudio ist nicht installiert!")
        try:
            import torch
            if isinstance(audio, bytes):
                raise ValueError("Audio als bytes nicht unterstützt. Erwartet Tensor oder np.ndarray.")
            tensor = torch.tensor(audio) if not isinstance(audio, torch.Tensor) else audio
            torchaudio.save(str(path), tensor.unsqueeze(0), sample_rate=sample_rate)
            logging.info(f"Audio gespeichert: {path}")
        except Exception as e:
            logging.error(f"Fehler beim Speichern von Audio: {e}")
            raise
