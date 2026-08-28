"""Serviço de predição tabular."""
import logging
import os
import joblib
import numpy as np
import shap
from src.api.cache import PredictionCache
from src.api.schemas import TabularPredictionResponse

logger = logging.getLogger(__name__)


class TabularService:
    """Gerencia modelos tabulares e faz predições."""

    def __init__(self):
        self.models = {}
        self.scaler = None
        self.model_scalers = {}
        self.feature_names = None
        self.explainers = {}
        self._load_models()
        self.cache = PredictionCache(maxsize=512, ttl=300)

    def _load_models(self):
        """Carrega modelos treinados do disco."""
        model_dir = "models/tabular"
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
            try:
                self.explainers[name] = shap.Explainer(self.models[name])
            except Exception:
                pass
            logger.info("Modelo carregado: %s", name)

        scaler_path = os.path.join(model_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            for model_name in self.models:
                self.model_scalers[model_name] = self.scaler

        feat_path = os.path.join(model_dir, "feature_names.joblib")
        if os.path.exists(feat_path):
            self.feature_names = joblib.load(feat_path)

        self._load_legacy_model()

    def _load_legacy_model(self):
        """Carrega modelo legado (.pkl) da fase inicial para compatibilidade."""
        legacy_dir = "models/legacy"
        legacy_model_path = os.path.join(legacy_dir, "modelo_breast_cancer.pkl")
        if os.path.exists(legacy_model_path):
            self.models["legacy_logistic_regression"] = joblib.load(legacy_model_path)
            logger.info("Modelo legado carregado: legacy_logistic_regression")

        legacy_scaler_path = os.path.join(legacy_dir, "scaler.pkl")
        if os.path.exists(legacy_scaler_path):
            legacy_scaler = joblib.load(legacy_scaler_path)
            self.model_scalers["legacy_logistic_regression"] = legacy_scaler

    def predict(self, features: list, model_name: str = "random_forest") -> TabularPredictionResponse:
        """Faz predição + explicabilidade SHAP com cache."""
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} não disponível. Opções: {list(self.models.keys())}")

        # Verifica cache
        cached = self.cache.get(features, model_name)
        if cached is not None:
            logger.debug("Tabular cache hit: %s", model_name)
            return cached

        model = self.models[model_name]
        X = np.array(features).reshape(1, -1)

        model_scaler = self.model_scalers.get(model_name, self.scaler)
        if model_scaler is not None:
            X_scaled = model_scaler.transform(X)
        else:
            X_scaled = X

        proba = model.predict_proba(X_scaled)[0]
        pred_class = model.predict(X_scaled)[0]

        classes = list(getattr(model, "classes_", []))
        if 1 in classes and 0 in classes:
            malignant_idx = classes.index(1)
            benign_idx = classes.index(0)
        elif len(proba) >= 2:
            malignant_idx = 1
            benign_idx = 0
        else:
            malignant_idx = 0
            benign_idx = 0

        diagnosis = "M" if int(pred_class) == 1 else "B"
        probability_malignant = float(proba[malignant_idx])
        probability_benign = float(proba[benign_idx])

        # SHAP explicabilidade
        shap_top = None
        if model_name in self.explainers and self.feature_names is not None:
            try:
                shap_values = self.explainers[model_name](X_scaled)
                vals = shap_values.values
                if vals.ndim == 3:
                    vals = vals[0, :, 1]
                elif vals.ndim == 2:
                    vals = vals[0]
                top_idx = np.argsort(np.abs(vals))[-5:][::-1]
                shap_top = {self.feature_names[i]: float(vals[i]) for i in top_idx}
            except Exception as e:
                logger.warning("SHAP falhou para %s: %s", model_name, e)

        result = TabularPredictionResponse(
            diagnosis=diagnosis,
            probability_malignant=probability_malignant,
            probability_benign=probability_benign,
            model_used=model_name,
            shap_top_features=shap_top
        )

        # Armazena no cache
        self.cache.set(features, model_name, result)
        return result

    @property
    def is_loaded(self) -> bool:
        return len(self.models) > 0