# Artefakt-Referenz

| Dateiname                                 | Zweck                                               | Zusammenhang zum Ziel                                 |
|--------------------------------------------|-----------------------------------------------------|-------------------------------------------------------|
| tests/inference/test_ctranslate2_integration.py | Integrationstest für CTranslate2-Inferenz-Backend   | Sicherstellen, dass die Migration zu CTranslate2 funktioniert |
| dia/ctranslate2.py                         | Dummy-Backend für CTranslate2-Integration. Erlaubt TDD und Testausführung ohne echtes Backend. | Integrationstest, TDD, CTranslate2 |
| dac.py                                     | Dummy-Modul für DAC (Descript Audio Codec)           | Ermöglicht TDD auf ARM/Mac, solange echtes Paket nicht verfügbar ist |
