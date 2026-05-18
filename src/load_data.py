from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/diabetes.csv")


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    return df


if __name__ == "__main__":
    df = load_dataset()

    print("\n✅ Dataset loaded successfully")

    print("\nShape:")
    print(df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)