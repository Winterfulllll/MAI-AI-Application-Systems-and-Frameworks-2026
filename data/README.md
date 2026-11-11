# Данные

Файлы датасетов не хранятся в репозитории. Они скачиваются автоматически при запуске ноутбука (`src.data.download_data()`), либо вручную:

| Файл                    | Источник                                                                                                                                            |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `features_3_sec.csv`    | [GTZAN Music Genre Classification (Kaggle)](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), папка `Data/` |
| `YearPredictionMSD.txt` | [YearPredictionMSD (UCI)](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd)                                                                |
| `msd_sample.npz`        | создаётся автоматически: фиксированная подвыборка MSD 50 000 train / 10 000 test                                                                    |
