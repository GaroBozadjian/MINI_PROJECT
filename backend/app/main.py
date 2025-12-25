from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from app.db import get_db
from app.models import IrisRow
from app.ml.model import predict, ModelNotTrainedError

app = FastAPI(title="ML Dashboard Demo", version="1.0.0")

class PredictRequest(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)

class AnalyzeResponse(BaseModel):
    rows: int
    columns: int
    describe: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/db-info")
def db_info(db: Session = Depends(get_db)):
    total = db.execute(select(func.count(IrisRow.id))).scalar_one()
    return {"rows_in_iris_table": int(total)}


@app.get("/analyze", response_model=AnalyzeResponse)
def analyze(db: Session = Depends(get_db)):
    # Pull data directly from DB (no helper)
    rows = db.execute(select(IrisRow)).scalars().all()
    if not rows:
        raise HTTPException(
            status_code=400,
            detail="No data found in DB. Run: python -m app.seed_db"
        )

    # Convert DB rows -> DataFrame
    df = pd.DataFrame([{
        "sepal_length": r.sepal_length,
        "sepal_width": r.sepal_width,
        "petal_length": r.petal_length,
        "petal_width": r.petal_width,
        "target": r.target,
    } for r in rows])

    desc = df.describe(include="all").to_dict()
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "describe": desc,
    }


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    # Prediction uses the trained model artifact; inputs come from dashboard
    try:
        return predict([
            req.sepal_length,
            req.sepal_width,
            req.petal_length,
            req.petal_width
        ])
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
