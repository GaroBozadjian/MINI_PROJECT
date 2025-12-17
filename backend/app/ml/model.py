from pathlib import Path
import joblib
import numpy as np

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"

TARGET_NAMES = ["setosa", "versicolor", "virginica"]

class ModelNotTrainedError(RuntimeError):
    pass

def load_model():
    if not MODEL_PATH.exists():
        raise ModelNotTrainedError(
            f"Model not found at {MODEL_PATH}. Run: python -m app.ml.train"
        )
    return joblib.load(MODEL_PATH)

def predict(features: list[float]) -> dict:
    """
    features: [sepal_length, sepal_width, petal_length, petal_width]
    """
    model = load_model()
    X = np.array(features, dtype=float).reshape(1, -1)
    probs = model.predict_proba(X)[0]
    idx = int(np.argmax(probs))
    return {
        "class_id": idx,
        "class_name": TARGET_NAMES[idx],
        "probabilities": {TARGET_NAMES[i]: float(probs[i]) for i in range(len(TARGET_NAMES))}
    }
