"""Servico da API para predicao de diabetes."""
import logging
from src.diabetes.predict import DiabetesPredictor
from src.api.cache import PredictionCache
from src.api.schemas import DiabetesPredictionResponse

logger = logging.getLogger(__name__)


class DiabetesService:
    """Gerencia o predictor de diabetes."""

    def __init__(self):
        self.predictor = None
        self._load_model()
        self.cache = PredictionCache(maxsize=256, ttl=300)

    def _load_model(self):
        try:
            self.predictor = DiabetesPredictor()
            logger.info("Modelo de diabetes carregado")
        except Exception as e:
            logger.warning("Falha ao carregar diabetes: %s", e)

    def predict(self, features: list, model_name: str = "random_forest") -> DiabetesPredictionResponse:
        if self.predictor is None:
            raise RuntimeError("Modelo de diabetes nao carregado")

        # Verifica cache
        cached = self.cache.get(features, model_name)
        if cached is not None:
            logger.debug("Diabetes cache hit: %s", model_name)
            return cached

        result_raw = self.predictor.predict(features, model_name)
        result = DiabetesPredictionResponse(**result_raw)

        # Armazena no cache
        self.cache.set(features, model_name, result)
        return result

    @property
    def is_loaded(self) -> bool:
        return self.predictor is not None and self.predictor.is_loaded

    @property
    def available_models(self) -> list:
        if self.predictor is None:
            return []
        return self.predictor.available_models
