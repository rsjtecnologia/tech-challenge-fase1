"""Inferência dos modelos persistidos de SOP."""
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path("models/pcos")


class PCOSPredictor:
    """Carrega modelos de SOP e produz classificação probabilística."""

    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.feature_names = []
        self._load_models()

    def _load_models(self) -> None:
        if not MODEL_DIR.exists():
            return
        for path in sorted(MODEL_DIR.glob("*.joblib")):
            if path.name not in {"preprocessor.joblib", "feature_names.joblib"}:
                self.models[path.stem] = joblib.load(path)
        preprocessor_path = MODEL_DIR / "preprocessor.joblib"
        features_path = MODEL_DIR / "feature_names.joblib"
        if preprocessor_path.exists() and features_path.exists():
            self.preprocessor = joblib.load(preprocessor_path)
            self.feature_names = joblib.load(features_path)

    def predict(self, features: list[float], model_name: str = "random_forest") -> dict:
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} indisponível.")
        if len(features) != len(self.feature_names):
            raise ValueError(f"São esperadas {len(self.feature_names)} features.")
        frame = pd.DataFrame([features], columns=self.feature_names)
        transformed = self.preprocessor.transform(frame)
        model = self.models[model_name]
        probability = float(model.predict_proba(transformed)[0, 1])
        return {
            "prediction": "PCOS" if probability >= 0.5 else "No PCOS",
            "probability_positive": probability,
            "probability_negative": 1 - probability,
            "model_used": model_name,
            "features": features,
        }

    @property
    def is_loaded(self) -> bool:
        return bool(self.models) and self.preprocessor is not None and bool(self.feature_names)

    @property
    def available_models(self) -> list[str]:
        return list(self.models)
