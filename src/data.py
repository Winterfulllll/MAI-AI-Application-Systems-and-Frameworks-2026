from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RANDOM_STATE = 42

GTZAN_HANDLE = "andradaolteanu/gtzan-dataset-music-genre-classification"
MSD_URL = "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip"

# Официальное разбиение UCI: первые 463 715 строк — train, остальные 51 630 — test.
MSD_TRAIN_ROWS = 463_715
MSD_TRAIN_SAMPLE = 50_000
MSD_TEST_SAMPLE = 10_000


def download_data() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    gtzan = DATA_DIR / "features_3_sec.csv"
    if not gtzan.exists():
        import kagglehub

        src = kagglehub.dataset_download(GTZAN_HANDLE, path="Data/features_3_sec.csv")
        gtzan.write_bytes(Path(src).read_bytes())
    msd_txt = DATA_DIR / "YearPredictionMSD.txt"
    if not msd_txt.exists():
        zip_path = DATA_DIR / "yearpredictionmsd.zip"
        urllib.request.urlretrieve(MSD_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_DIR)


def load_gtzan_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "features_3_sec.csv")
    df["track"] = df["filename"].str.rsplit(".", n=2).str[0]
    return df


def gtzan_split():
    df = load_gtzan_raw()
    X = df.drop(columns=["filename", "length", "label", "track"])
    y = df["label"].to_numpy()
    groups = df["track"].to_numpy()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
                  .split(X, y, groups))
    return (X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True),
            y[tr], y[te], groups[tr], groups[te])


MSD_COLUMNS = [f"timbre_avg_{i}" for i in range(1, 13)] + [f"timbre_cov_{i}" for i in range(1, 79)]


def msd_split():
    cache = DATA_DIR / "msd_sample.npz"
    if not cache.exists():
        df = pd.read_csv(DATA_DIR / "YearPredictionMSD.txt", header=None, dtype=np.float32)
        train = df.iloc[:MSD_TRAIN_ROWS].sample(MSD_TRAIN_SAMPLE, random_state=RANDOM_STATE)
        test = df.iloc[MSD_TRAIN_ROWS:].sample(MSD_TEST_SAMPLE, random_state=RANDOM_STATE)
        np.savez_compressed(cache, train=train.to_numpy(), test=test.to_numpy())
    z = np.load(cache)
    train, test = z["train"], z["test"]
    to_df = lambda a: pd.DataFrame(a[:, 1:], columns=MSD_COLUMNS)
    return to_df(train), to_df(test), train[:, 0].astype(int), test[:, 0].astype(int)
