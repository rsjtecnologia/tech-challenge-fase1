"""Serviço da API para predição de síndrome dos ovários policísticos."""
import logging

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.api.cache import PredictionCache
from src.api.schemas import PCOSPredictionResponse
from src.pcos.dataset import get_pcos_info, load_pcos_dataframe, load_pcos_data
from src.pcos.predict import PCOSPredictor
from src.cancer.reporting import (
    build_eda_payload,
    build_metrics_payload,
    shap_bar_graph,
    top_features_table,
)

logger = logging.getLogger(__name__)

# Rótulos das classes de SOP (índice 0 = sem SOP, 1 = com SOP)
PCOS_CLASS_LABELS = ("Sem SOP", "Com SOP")


class PCOSService:
    """Gerencia o preditor de SOP e seu cache."""

    def __init__(self):
        self.predictor = None
        self._load_model()
        self.cache = PredictionCache(maxsize=256, ttl=300)
        self._metrics_cache = None

    def _load_model(self):
        try:
            self.predictor = PCOSPredictor()
            logger.info("Modelo de SOP carregado")
        except Exception as error:
            logger.warning("Falha ao carregar SOP: %s", error)

    def predict(self, features: list[float], model_name: str) -> PCOSPredictionResponse:
        if self.predictor is None or not self.predictor.is_loaded:
            raise RuntimeError("Modelos de SOP não carregados")
        if model_name not in self.predictor.models:
            raise ValueError(
                f"Modelo {model_name} indisponível. Opções: {self.predictor.available_models}"
            )
        cached = self.cache.get(features, model_name)
        if cached is not None:
            return cached
        result_raw = self.predictor.predict(features, model_name)
        model = self.predictor.models[model_name]
        contributions = self._feature_contributions(model, features)
        graphs = tables = None
        if contributions is not None:
            graphs = {
                "feature_importance": shap_bar_graph(
                    self.predictor.feature_names,
                    contributions,
                    title=f"Variáveis clínicas relevantes - {model_name}",
                )
            }
            tables = {
                "top_features": top_features_table(
                    model,
                    self.predictor.feature_names,
                    n=10,
                    values=contributions,
                )
            }
        result = PCOSPredictionResponse(
            **result_raw,
            graphs=graphs,
            tables=tables,
        )
        self.cache.set(features, model_name, result)
        return result

    def _feature_contributions(self, model, features: list[float]) -> list[float] | None:
        """Retorna importâncias por variável para os modelos que as expõem."""
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_.tolist()
        if hasattr(model, "coef_"):
            transformed = self.predictor.preprocessor.transform(
                np.asarray(features, dtype=float).reshape(1, -1)
            )
            return (model.coef_[0] * transformed[0]).tolist()
        return None

    @property
    def is_loaded(self) -> bool:
        return self.predictor.is_loaded

    @property
    def available_models(self) -> list[str]:
        return self.predictor.available_models

    def get_eda(self) -> dict:
        """Retorna tabelas e gráficos exploratórios no padrão do câncer (com rótulos de SOP)."""
        dataframe = load_pcos_dataframe().rename(columns={"PCOS (Y/N)": "target"})
        payload = build_eda_payload(dataframe, class_labels=PCOS_CLASS_LABELS)
        payload["info"] = get_pcos_info()
        return payload

    def get_metrics(self) -> dict:
        """Reavalia todos os modelos de SOP no split determinístico de teste."""
        if self._metrics_cache is not None:
            return self._metrics_cache
        if self.predictor is None or not self.predictor.is_loaded:
            self._metrics_cache = build_metrics_payload({})
            return self._metrics_cache
        X_train, X_test, _, y_test, _ = load_pcos_data()
        transformed = self.predictor.preprocessor.transform(X_test)
        results = {}
        for name, model in self.predictor.models.items():
            prediction = model.predict(transformed)
            probability = model.predict_proba(transformed)[:, 1]
            results[name] = {
                "accuracy": accuracy_score(y_test, prediction),
                "auc": roc_auc_score(y_test, probability),
                "recall": recall_score(y_test, prediction, zero_division=0),
                "precision": precision_score(y_test, prediction, zero_division=0),
                "f1": f1_score(y_test, prediction, zero_division=0),
            }
        self._metrics_cache = build_metrics_payload(results)
        return self._metrics_cache
