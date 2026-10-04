import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_PATH, FEATURES, TARGET, RANDOM_STATE, TEST_SIZE


def load_and_clean_data(path=DATA_PATH):
    df = pd.read_csv(path)

    required = FEATURES + [TARGET]
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[required].copy()

    before = len(df)
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)

    return df, {
        "rows_before": before,
        "duplicates_removed": duplicate_count,
        "rows_after": after,
        "missing_values": int(df.isna().sum().sum()),
    }


def split_data(df):
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
