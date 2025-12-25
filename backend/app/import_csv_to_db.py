from pathlib import Path
import pandas as pd
from sqlalchemy import select, func
from app.db import engine, SessionLocal, Base
from app.models import IrisRow

# Adjust this path if your CSV is elsewhere
CSV_PATH = Path(__file__).resolve().parents[2] / "sample_data" / "iris.csv"

# If your CSV has different column names, map them here:
# Keys = what your CSV has, Values = what DB expects
COLUMN_MAP = {
    "sepal_length": "sepal_length",
    "sepal_width": "sepal_width",
    "petal_length": "petal_length",
    "petal_width": "petal_width",
    "target": "target",
    # Common iris dataset column variants you might have:
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width",
}

REQUIRED_DB_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]

def import_csv():
    # 1) Create tables if not exist
    Base.metadata.create_all(bind=engine)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    # 2) Read CSV
    df = pd.read_csv(CSV_PATH)

    # 3) Rename columns to match DB model
    rename_dict = {}
    for col in df.columns:
        if col in COLUMN_MAP:
            rename_dict[col] = COLUMN_MAP[col]
    df = df.rename(columns=rename_dict)

    # 4) Validate required columns
    missing = [c for c in REQUIRED_DB_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns after mapping: {missing}\n"
            f"CSV columns are: {list(df.columns)}\n"
            f"Edit COLUMN_MAP in import_csv_to_db.py to map your column names."
        )

    # 5) Keep only needed columns and coerce types
    df = df[REQUIRED_DB_COLUMNS].copy()

    # If your CSV has target as string labels, convert to numeric (optional):
    # setosa/versicolor/virginica -> 0/1/2
    if df["target"].dtype == object:
        label_map = {"setosa": 0, "versicolor": 1, "virginica": 2}
        df["target"] = df["target"].map(label_map)

    df = df.dropna()
    df["target"] = df["target"].astype(int)

    for c in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        df[c] = df[c].astype(float)

    # 6) Optional: avoid double-insert by checking if table already has rows
    with SessionLocal() as db:
        existing = db.execute(select(func.count(IrisRow.id))).scalar_one()
        if existing and existing > 0:
            print(f"DB already has {existing} rows. Skipping import.")
            return

        # 7) Insert rows
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

    print(f"Imported {len(df)} rows from CSV into SQLite DB at: {engine.url}")

if __name__ == "__main__":
    import_csv()
