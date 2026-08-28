"""Servico de predicao de diabetes."""
import logging
import os
import joblib
import numpy as np

logger = logging.getLogger(__name__)


class DiabetesPredictor:
    """Carrega modelos treinados e faz predicoes de diabetes."""

    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self._load_models()

    def _load_models(self):
        model_dir = "models/diabetes"
        if not os.path.exists(model_dir):
            logger.warning("Diretório %s não encontrado.", model_dir)
            return
        for filename in sorted(os.listdir(model_dir)):
            if not filename.endswith(".joblib"):
                continue
            if filename in {"scaler.joblib", "feature_names.joblib"}:
                continue
            name = filename[:-7]
            path = os.path.join(model_dir, filename)
            self.models[name] = joblib.load(path)
            logger.info("Modelo diabetes carregado: %s", name)
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        feat_path = os.path.join(model_dir, "feature_names.joblib")
        if os.path.exists(feat_path):
            self.feature_names = joblib.load(feat_path)

    def predict(self, features: list, model_name: str = "random_forest") -> dict:
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} indisponivel. Opcoes: {list(self.models.keys())}")
        model = self.models[model_name]
        X = np.array(features).reshape(1, -1)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        proba = model.predict_proba(X)[0]
        pred = model.predict(X)[0]
        return {
            "prediction": "Positive" if pred == 1 else "Negative",
            "probability_positive": float(proba[1]),
            "probability_negative": float(proba[0]),
            "model_used": model_name,
            "features": features,
        }

    @property
    def is_loaded(self) -> bool:
        return len(self.models) > 0

    @property
    def available_models(self) -> list:
        return list(self.models.keys())
