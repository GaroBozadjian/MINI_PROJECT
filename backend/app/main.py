from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from sklearn.datasets import load_iris

from app.ml.model import predict, ModelNotTrainedError

app = FastAPI(title="ML Dashboard Demo", version="1.0.0")

# --- Pydantic schemas ---
class PredictRequest(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)

class AnalyzeResponse(BaseModel):
    rows: int
    columns: int
    describe: dict  # pandas describe() to dict


# --- Helpers ---
def iris_dataframe() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    # df includes target; keep it for analysis
    return df


# --- Routes ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/analyze", response_model=AnalyzeResponse)
def analyze():
    df = iris_dataframe()
    desc = df.describe(include="all").to_dict()
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "describe": desc,
    }

@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    try:
        result = predict([
            req.sepal_length,
            req.sepal_width,
            req.petal_length,
            req.petal_width
        ])
        return result
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
