"""Testes do modulo Cancer de Mama."""
import pytest
import numpy as np


class TestDataset:
    def test_info(self):
        from src.cancer.dataset import get_cancer_info
        info = get_cancer_info()
        assert info["samples"] == 569
        assert info["features"] == 30
        assert info["task"] == "classification"
        assert info["encoding"] == {"M": 1, "B": 0}

    def test_load_data(self):
        from src.cancer.dataset import load_cancer_data
        X_train, X_test, y_train, y_test, scaler, feature_names = load_cancer_data()
        assert len(feature_names) == 30
        assert X_train.shape[1] == 30
        assert len(X_train) > 0
        # Target deve ser M=1, B=0
        assert set(np.unique(y_train)).issubset({0, 1})

    def test_load_dataframe(self):
        from src.cancer.dataset import load_cancer_dataframe
        df = load_cancer_dataframe()
        assert df.shape == (569, 32)  # diagnosis + 30 features + target
        assert "diagnosis" in df.columns
        assert "target" in df.columns
        assert df["target"].value_counts().get(1, 0) == 212  # M
        assert df["target"].value_counts().get(0, 0) == 357  # B


class TestModel:
    def test_create_models(self):
        from src.cancer.model import create_cancer_models
        models = create_cancer_models()
        assert len(models) == 6
        assert "random_forest" in models


class TestPredictor:
    def test_import(self):
        from src.cancer.predict import CancerPredictor
        assert CancerPredictor is not None

    def test_init_without_models(self):
        from src.cancer.predict import CancerPredictor
        predictor = CancerPredictor(model_dir="models/nao_existe")
        assert hasattr(predictor, 'is_loaded')

    def test_predict_with_mocked_model(self):
        from src.cancer.predict import CancerPredictor
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_dummy = np.random.randn(100, 30)
        y_dummy = (X_dummy[:, 0] > 0).astype(int)  # 1 = M, 0 = B
        model.fit(X_dummy, y_dummy)

        predictor = CancerPredictor(model_dir="models/nao_existe")
        predictor.models = {"random_forest": model}

        features = list(np.random.randn(30))
        result = predictor.predict(features, "random_forest")

        assert "diagnosis" in result
        assert result["diagnosis"] in ["M", "B"]
        assert 0 <= result["probability_malignant"] <= 1
        assert 0 <= result["probability_benign"] <= 1
        assert result["model_used"] == "random_forest"


class TestTrain:
    def test_import(self):
        from src.cancer.train import train_cancer_models
        assert callable(train_cancer_models)


class TestReporting:
    def test_eda_payload(self):
        from src.cancer.reporting import build_eda_payload
        from src.cancer.dataset import load_cancer_dataframe
        df = load_cancer_dataframe()
        payload = build_eda_payload(df)
        assert "tables" in payload and "graphs" in payload
        assert len(payload["tables"]["classes"]) == 2
        assert payload["tables"]["classes"][0]["label"] == 0  # B
        assert payload["tables"]["classes"][1]["label"] == 1  # M
        assert payload["graphs"]["classes"].startswith("data:image/png;base64,")

    def test_metrics_payload(self):
        from src.cancer.reporting import build_metrics_payload, metrics_table
        resultados = {
            "random_forest": {"accuracy": 0.97, "recall": 0.92, "precision": 1.0,
                              "f1": 0.96, "auc": 0.99},
        }
        payload = build_metrics_payload(resultados)
        assert len(payload["tables"]["metricas"]) == 1
        assert payload["tables"]["metricas"][0]["acuracia"] == 97.0
        assert payload["graphs"]["comparativo"].startswith("data:image/png;base64,")


class TestServiceIntegration:
    def test_models_listed_in_main(self):
        """O app principal deve incluir as rotas existentes de tabular."""
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app)
        r = client.get("/predict/tabular/models")
        assert r.status_code == 200

    def test_cancer_models_list(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app)
        r = client.get("/predict/cancer/models")
        assert r.status_code == 200

    def test_cancer_info(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app)
        r = client.get("/predict/cancer/info")
        assert r.status_code == 200
        data = r.json()
        assert data["encoding"] == {"M": 1, "B": 0}

    def test_cancer_samples_endpoint(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app)
        r = client.get("/predict/cancer/samples")
        assert r.status_code == 200
        data = r.json()
        assert data["source"].endswith("breast_cancer_wisconsin.csv")
        assert len(data["feature_names"]) == 30
        assert data["samples"]["benign"]["diagnosis"] == "B"
        assert data["samples"]["malignant"]["diagnosis"] == "M"

    def test_cancer_metrics(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app)
        r = client.get("/predict/cancer/metrics")
        assert r.status_code == 200
        data = r.json()
        assert "tables" in data and "graphs" in data
        assert len(data["tables"]["metricas"]) == 6

    def test_cancer_eda(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app)
        r = client.get("/predict/cancer/eda")
        assert r.status_code == 200
        data = r.json()
        assert "tables" in data and "graphs" in data
        assert data["tables"]["classes"][1]["label"] == 1  # M

    def test_cancer_predict_with_graphs(self):
        """Predicao de câncer deve incluir grafico (base64) e tabela."""
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app)
        features = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
                    0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
                    0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
                    0.2654, 0.4601, 0.1189]
        r = client.post("/predict/cancer/", json={"features": features})
        # Requer models/cancer/ treinados (o CI treina via train_cancer_models)
        if r.status_code == 503:
            pytest.skip("Modelos de câncer não treinados")
        assert r.status_code == 200
        data = r.json()
        assert data["diagnosis"] in ["M", "B"]
        assert 0 <= data["probability_malignant"] <= 1
        assert data["graphs"]["shap"].startswith("data:image/png;base64,")
        assert len(data["tables"]["top_features"]) > 0

    def test_cancer_predict_invalid_features(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app)
        r = client.post("/predict/cancer/", json={"features": [1.0] * 10})
        assert r.status_code == 422

    def test_cancer_predict_all_models_with_shap(self):
        """Todos os modelos (incl. linear) devem gerar SHAP + tabela."""
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app)
        features = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
                    0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
                    0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
                    0.2654, 0.4601, 0.1189]
        for m in ["random_forest", "gradient_boosting", "logistic_regression"]:
            r = client.post("/predict/cancer/", json={"features": features, "model_name": m})
            if r.status_code == 503:
                pytest.skip("Modelos de câncer não treinados")
            assert r.status_code == 200, f"{m}: {r.text}"
            data = r.json()
            assert data["graphs"]["shap"].startswith("data:image/png;base64,"), m
            assert len(data["tables"]["top_features"]) > 0, m

    def test_tabular_service_preserves_cancer_labels(self):
        """A predição tabular deve manter M=1 e B=0 na resposta."""
        from src.api.cache import PredictionCache
        from src.api.tabular_service import TabularService

        class DummyModel:
            classes_ = np.array([0, 1])

            def predict_proba(self, X):
                return np.array([[0.25, 0.75]])

            def predict(self, X):
                return np.array([1])

        service = TabularService.__new__(TabularService)
        service.models = {"dummy": DummyModel()}
        service.scaler = None
        service.model_scalers = {}
        service.feature_names = [f"f{i}" for i in range(30)]
        service.explainers = {}
        service.cache = PredictionCache(maxsize=8, ttl=60)

        result = service.predict([0.0] * 30, "dummy")
        assert result.diagnosis == "M"
        assert result.probability_malignant == pytest.approx(0.75)
        assert result.probability_benign == pytest.approx(0.25)

    def test_cancer_service_samples_are_real(self):
        """As amostras expostas ao front devem vir do CSV principal."""
        from src.api.cancer_service import CancerService

        service = CancerService.__new__(CancerService)
        samples = CancerService.get_samples(service)

        assert samples["source"].endswith("breast_cancer_wisconsin.csv")
        assert samples["feature_names"][0] == "mean radius"
        assert samples["samples"]["benign"]["diagnosis"] == "B"
        assert samples["samples"]["malignant"]["diagnosis"] == "M"
        assert len(samples["samples"]["benign"]["features"]) == 30

    def test_health_includes_cancer(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app)
        r = client.get("/health")
        data = r.json()
        assert "cancer_model_loaded" in data
        assert "cancer" in data["cache"]
