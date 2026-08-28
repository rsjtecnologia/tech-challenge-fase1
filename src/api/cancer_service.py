"""Servico da API para predicao de câncer de mama."""
import logging

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from sklearn.metrics import precision_score, f1_score

from src.cancer.predict import CancerPredictor
from src.cancer.dataset import load_cancer_dataframe, load_cancer_data, get_cancer_info
from src.cancer.reporting import (
    build_eda_payload,
    build_metrics_payload,
    shap_bar_graph,
    top_features_table,
)
from src.api.cache import PredictionCache
from src.api.schemas import CancerPredictionResponse

logger = logging.getLogger(__name__)


class CancerService:
    """Gerencia o predictor de câncer de mama + relatórios gráficos."""

    def __init__(self):
        self.predictor = None
        self._load_model()
        self.cache = PredictionCache(maxsize=256, ttl=300)
        self._metrics_cache = None
        self._shap_bg = None

    def _load_model(self):
        try:
            self.predictor = CancerPredictor()
            logger.info("Modelo de câncer de mama carregado")
        except Exception as e:
            logger.warning("Falha ao carregar câncer: %s", e)

    def predict(self, features: list, model_name: str = "random_forest") -> CancerPredictionResponse:
        if self.predictor is None:
            raise RuntimeError("Modelo de câncer nao carregado")

        cached = self.cache.get(features, model_name)
        if cached is not None:
            logger.debug("Cancer cache hit: %s", model_name)
            return cached

        result_raw = self.predictor.predict(features, model_name)

        # Grafico + tabela de explicabilidade (SHAP real, como o notebook)
        graphs = None
        tables = None
        model = self.predictor.models.get(model_name)
        if model is not None:
            try:
                X = np.array(features).reshape(1, -1)
                if self.predictor.scaler is not None:
                    X = self.predictor.scaler.transform(X)
                feats = self.predictor.feature_names or \
                    [f"feature_{i}" for i in range(len(features))]

                import shap
                explainer = self._build_shap_explainer(model, model_name)
                shap_values = explainer(X)
                vals = shap_values.values
                if vals.ndim == 3:
                    vals = vals[0, :, 1]  # classe M
                elif vals.ndim == 2:
                    vals = vals[0]

                graphs = {"shap": shap_bar_graph(
                    feats, vals.tolist(),
                    title=f"Importância das Features (SHAP) - {model_name}")}
                tables = {"top_features": top_features_table(
                    model, feats, n=10, values=vals.tolist())}
            except Exception as e:
                logger.warning("Falha ao gerar explicabilidade: %s", e)

        result = CancerPredictionResponse(
            **result_raw,
            graphs=graphs,
            tables=tables,
        )

        self.cache.set(features, model_name, result)
        return result

    def get_eda(self) -> dict:
        """Tabelas e graficos de EDA do dataset (como o notebook)."""
        df = load_cancer_dataframe()
        payload = build_eda_payload(df)
        payload["info"] = get_cancer_info()
        return payload

    def get_metrics(self) -> dict:
        """
        Tabela e grafico comparativo dos modelos.

        As métricas são calculadas em runtime a partir dos modelos
        carregados e do split de teste do dataset (com cache).
        """
        if self._metrics_cache is not None:
            return self._metrics_cache

        if self.predictor is None or not self.predictor.is_loaded:
            self._metrics_cache = build_metrics_payload({})
            return self._metrics_cache

        # Recalcula o split de teste (sem escala; a escala e aplicada com o scaler carregado)
        X_train, X_test, y_train, y_test, _, feature_names = load_cancer_data(scale=False)
        X_test_s = X_test
        if self.predictor.scaler is not None:
            X_test_s = self.predictor.scaler.transform(X_test)

        resultados = {}
        for name, model in self.predictor.models.items():
            try:
                y_pred = model.predict(X_test_s)
                y_proba = model.predict_proba(X_test_s)[:, 1]
                resultados[name] = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "auc": roc_auc_score(y_test, y_proba),
                    "recall": recall_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred),
                }
            except Exception as e:
                logger.warning("Falha ao avaliar %s: %s", name, e)

        self._metrics_cache = build_metrics_payload(resultados)
        return self._metrics_cache

    def get_samples(self) -> dict:
        """Retorna amostras reais da base para uso no frontend."""
        df = load_cancer_dataframe()
        feature_names = [c for c in df.columns if c not in ("diagnosis", "target")]

        def _row_payload(row) -> dict:
            return {
                "diagnosis": row["diagnosis"],
                "label": int(row["target"]),
                "features": [float(row[col]) for col in feature_names],
            }

        benign_rows = df[df["diagnosis"] == "B"]
        malignant_rows = df[df["diagnosis"] == "M"]
        if benign_rows.empty or malignant_rows.empty:
            raise ValueError("N?o foi poss?vel localizar amostras benignas e malignas na base.")

        return {
            "source": "data/breast_cancer_wisconsin.csv",
            "feature_names": feature_names,
            "default_sample": "benign",
            "samples": {
                "benign": _row_payload(benign_rows.iloc[0]),
                "malignant": _row_payload(malignant_rows.iloc[0]),
            },
        }

    def _build_shap_explainer(self, model, model_name: str):
        """
        Cria o explainer SHAP adequado ao tipo de modelo.

        - Tree-based (Random Forest / Gradient Boosting): TreeExplainer.
        - Linear (Logistic Regression): LinearExplainer com background
          dos dados de treino (não é callable diretamente).
        """
        import shap
        if model_name == "logistic_regression":
            X_bg = self._shap_background()
            return shap.LinearExplainer(model, masker=X_bg)
        return shap.TreeExplainer(model)

    def _shap_background(self, n: int = 100) -> np.ndarray:
        """Amostra dos dados de treino (escalados) como background do SHAP."""
        if self._shap_bg is None:
            X_train, _, _, _, _, _ = load_cancer_data(scale=False)
            X_bg = X_train[:n]
            if self.predictor is not None and self.predictor.scaler is not None:
                X_bg = self.predictor.scaler.transform(X_bg)
            self._shap_bg = X_bg
        return self._shap_bg

    @property
    def is_loaded(self) -> bool:
        return self.predictor is not None and self.predictor.is_loaded

    @property
    def available_models(self) -> list:
        if self.predictor is None:
            return []
        return self.predictor.available_models
