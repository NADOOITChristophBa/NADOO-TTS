# Dummy-Modul für dac (Descript Audio Codec)
# Ermöglicht testgetriebenes Arbeiten, wenn das echte Paket nicht verfügbar ist

class DummyUtils:
    @staticmethod
    def download():
        # Gibt einen Dummy-Pfad zurück
        return "/tmp/dummy_dac_model.pt"

class DummyDAC:
    def __init__(self, *args, **kwargs):
        pass
    def encode(self, x):
        return x
    def decode(self, x):
        return x
    def to(self, device):
        return self
    @staticmethod
    def load(path):
        return DummyDAC()

Model = DummyDAC
DAC = DummyDAC
utils = DummyUtils
