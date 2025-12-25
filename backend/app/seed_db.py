from pathlib import Path
import pandas as pd
from sqlalchemy import select, func

from app.db import engine, SessionLocal, Base
from app.models import IrisRow

# Point to your CSV file here
CSV_PATH = Path(__file__).resolve().parents[2] / "sampledata" / "iris.csv"

# Map CSV headers -> DB column names (edit if needed)
COLUMN_MAP = {
    "SepalLengthCm": "sepal_length",
    "SepalWidthCm": "sepal_width",
    "PetalLengthCm": "petal_length",
    "PetalWidthCm": "petal_width",
    "Species": "target",
}

REQUIRED_DB_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]

def seed_from_csv() -> None:
    # 1) create tables if missing
    Base.metadata.create_all(bind=engine)

    # 2) validate CSV exists
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    # 3) read CSV
    df = pd.read_csv(CSV_PATH)

    # 4) rename columns according to mapping
    rename_dict = {col: COLUMN_MAP[col] for col in df.columns if col in COLUMN_MAP}
    df = df.rename(columns=rename_dict)

    # 5) check required columns exist
    missing = [c for c in REQUIRED_DB_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns after mapping: {missing}\n"
            f"CSV columns found: {list(df.columns)}\n"
            f"Fix COLUMN_MAP in seed_db.py to map your CSV headers."
        )

    # 6) keep needed columns only
    df = df[REQUIRED_DB_COLUMNS].copy()

    # 7) convert target labels -> numbers if needed
    if df["target"].dtype == object:
        # Normalize strings (strip spaces, unify case)
        df["target"] = df["target"].astype(str).str.strip()

        # Common Kaggle Iris labels are usually: Iris-setosa, Iris-versicolor, Iris-virginica
        # But sometimes they appear without "Iris-" or with different casing.
        df["target_norm"] = (
            df["target"]
            .str.replace("Iris-", "", regex=False)
            .str.replace("iris-", "", regex=False)
            .str.lower()
        )

        label_map = {"setosa": 0, "versicolor": 1, "virginica": 2}
        df["target"] = df["target_norm"].map(label_map)

        # If mapping failed for any row, stop with a clear error
        unmapped = df[df["target"].isna()]
        if not unmapped.empty:
            sample = unmapped["target_norm"].value_counts().head(10).to_dict()
            raise ValueError(
                f"Species->target mapping failed for {len(unmapped)} rows. "
                f"Examples (normalized): {sample}. "
                f"Check your Species values in the CSV."
            )

        df = df.drop(columns=["target_norm"])

    # 8) enforce types and drop bad rows
    df = df.dropna()
    df["target"] = df["target"].astype(int)
    for c in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        df[c] = df[c].astype(float)

    # 9) avoid double seeding (simple approach)
    with SessionLocal() as db:
        existing = db.execute(select(func.count(IrisRow.id))).scalar_one()
        if existing and existing > 0:
            print(f"DB already has {existing} rows. Skipping seed.")
            return

        objects = [
            IrisRow(
                sepal_length=row.sepal_length,
                sepal_width=row.sepal_width,
                petal_length=row.petal_length,
                petal_width=row.petal_width,
                target=row.target,
            )
            for row in df.itertuples(index=False)
        ]

        db.add_all(objects)
        db.commit()

    print(f"Seeded {len(df)} rows into DB: {engine.url}")

if __name__ == "__main__":
    seed_from_csv()
