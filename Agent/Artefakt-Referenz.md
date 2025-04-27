# Artefakt-Referenz

| Dateiname                                 | Zweck                                               | Zusammenhang zum Ziel                                 |
|--------------------------------------------|-----------------------------------------------------|-------------------------------------------------------|
| tests/inference/test_ctranslate2_integration.py | Integrationstest für CTranslate2-Inferenz-Backend   | Sicherstellen, dass die Migration zu CTranslate2 funktioniert |
| tests/inference/test_ctranslate2_auto_download.py | End-to-End-Test für automatische Modellbereitstellung (Download, ONNX-Export, Konvertierung) | Stellt sicher, dass die Automatisierung robust funktioniert |
| tests/inference/test_ctranslate2_inference_and_audio.py | End-to-End-Test für vollständige TTS-Pipeline inkl. Audio-Validierung | Prüft, dass Modell, Inferenz und Audioausgabe produktionsreif funktionieren |
| tests/inference/test_ctranslate2_error_handling.py | Defensive Tests für Fehlerfälle (ungültiges Modell, fehlender Konverter) | Stellt sicher, dass das System robust auf Fehlerquellen reagiert |
| dia/ctranslate2.py                         | Dummy-Backend für CTranslate2-Integration. Erlaubt TDD und Testausführung ohne echtes Backend. | Integrationstest, TDD, CTranslate2 |
| dac.py                                     | Dummy-Modul für DAC (Descript Audio Codec)           | Ermöglicht TDD auf ARM/Mac, solange echtes Paket nicht verfügbar ist |
