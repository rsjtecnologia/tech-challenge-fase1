"""Testes do modulo Diabetes."""
import pytest
import numpy as np


class TestDataset:
    def test_info(self):
        from src.diabetes.dataset import get_diabetes_info
        info = get_diabetes_info()
        assert info["samples"] == 768
        assert info["features"] == 8
        assert info["task"] == "classification"

    def test_load_data(self):
        from src.diabetes.dataset import load_diabetes_data
        X_train, X_test, y_train, y_test, scaler, feature_names = load_diabetes_data()
        assert len(feature_names) == 8
        assert X_train.shape[1] == 8
        assert len(X_train) > 0


class TestModel:
    def test_create_models(self):
        from src.diabetes.model import create_diabetes_models
        models = create_diabetes_models()
        assert len(models) == 6
        assert "random_forest" in models


class TestPredictor:
    def test_import(self):
        from src.diabetes.predict import DiabetesPredictor
        assert DiabetesPredictor is not None

    def test_init_without_models(self):
        """DiabetesPredictor deve inicializar sem falhar mesmo sem modelos no disco."""
        from src.diabetes.predict import DiabetesPredictor
        predictor = DiabetesPredictor()
        # Pode ou não ter modelos carregados, mas não deve lançar exceção
        assert hasattr(predictor, 'is_loaded')

    def test_available_models_empty_when_none_loaded(self):
        from src.diabetes.predict import DiabetesPredictor
        predictor = DiabetesPredictor()
        if not predictor.is_loaded:
            assert predictor.available_models == []
        else:
            assert len(predictor.available_models) > 0

    def test_predict_with_mocked_model(self):
        """Testa predict() com modelos mockados."""
        from src.diabetes.predict import DiabetesPredictor
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        import tempfile
        import os

        # Cria modelo mock e salva temporariamente
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_dummy = np.random.randn(100, 8)
        y_dummy = (X_dummy[:, 0] + X_dummy[:, 1] > 0).astype(int)
        model.fit(X_dummy, y_dummy)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Salva modelo mock
            joblib.dump(model, os.path.join(tmpdir, "random_forest.joblib"))
            
            # Cria predictor e injeta modelos manualmente
            predictor = DiabetesPredictor()
            predictor.models = {"random_forest": model}

            # Testa predição
            features = [6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0]
            result = predictor.predict(features, "random_forest")
            
            assert "prediction" in result
            assert result["prediction"] in ["Positive", "Negative"]
            assert 0 <= result["probability_positive"] <= 1
            assert 0 <= result["probability_negative"] <= 1
            assert result["model_used"] == "random_forest"
            assert result["features"] == features


class TestService:
    def test_import(self):
        from src.api.diabetes_service import DiabetesService
        assert DiabetesService is not None

    def test_is_loaded_when_none(self):
        from src.api.diabetes_service import DiabetesService
        service = DiabetesService()
        # Predictor pode ou não estar carregado, mas não deve lançar exceção
        assert hasattr(service, 'is_loaded')

    def test_available_models_property(self):
        from src.api.diabetes_service import DiabetesService
        service = DiabetesService()
        if service.predictor is None:
            assert service.available_models == []
        else:
            assert isinstance(service.available_models, list)


class TestRoutes:
    def test_routes_exist(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app)
        r = client.get("/predict/diabetes/info")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "Pima Indians Diabetes Database"

    def test_predict_invalid_features(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app)
        r = client.post("/predict/diabetes/", json={"features": [1, 2, 3]})
        assert r.status_code == 422

    def test_models_list(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app)
        r = client.get("/predict/diabetes/models")
        assert r.status_code == 200
