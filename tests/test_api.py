"""Testes da API com pytest."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "tabular_model_loaded" in data


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "message" in data
    assert "documentation" in data
    assert data["documentation"]["swagger"] == "/docs"


def test_tabular_models_list():
    r = client.get("/predict/tabular/models")
    assert r.status_code == 200


def test_tabular_models_list_includes_legacy():
    r = client.get("/predict/tabular/models")
    assert r.status_code == 200
    data = r.json()
    assert "legacy_logistic_regression" in data["available_models"]


def test_tabular_predict_invalid_features():
    r = client.post("/predict/tabular/", json={"features": [1.0] * 10})
    assert r.status_code == 422  # validação Pydantic


def test_tabular_predict_valid():
    features = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
                0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
                0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
                0.2654, 0.4601, 0.1189]
    r = client.post("/predict/tabular/", json={"features": features})
    if r.status_code == 200:
        data = r.json()
        assert data["diagnosis"] in ["M", "B"]
        assert 0 <= data["probability_malignant"] <= 1


def test_tabular_predict_valid_legacy_model():
    features = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
                0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
                0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
                0.2654, 0.4601, 0.1189]
    r = client.post(
        "/predict/tabular/",
        json={"features": features, "model_name": "legacy_logistic_regression"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["model_used"] == "legacy_logistic_regression"
    assert data["diagnosis"] in ["M", "B"]
    assert 0 <= data["probability_malignant"] <= 1


def test_mammography_invalid_file():
    r = client.post("/predict/image/mammography",
                    files={"file": ("test.txt", b"not an image", "text/plain")})
    assert r.status_code == 400

def test_mammography_gradcam_invalid_file():
    r = client.post("/predict/image/mammography/gradcam",
                    files={"file": ("test.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_mammography_local_sample_prediction():
    samples = client.get("/predict/image/samples")
    assert samples.status_code == 200
    sample_id = samples.json()["samples"][0]["id"]

    response = client.post(f"/predict/image/mammography/sample/{sample_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in {"benign", "malignant"}
    assert 0 <= data["probability_malignant"] <= 1


def test_health_includes_cache_stats():
    """Health check deve incluir estatísticas do cache."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "cache" in data
    assert "tabular" in data["cache"]
    assert "image" in data["cache"]
    assert "diabetes" in data["cache"]
    assert data["cache"]["tabular"]["type"] == "memory"
    assert data["cache"]["diabetes"]["type"] == "memory"


def test_admin_cache_status():
    """GET /admin/cache deve retornar status de todos os caches."""
    r = client.get("/admin/cache")
    assert r.status_code == 200
    data = r.json()
    assert "services" in data
    assert "global_hit_rate" in data
    assert "total_entries" in data
    assert "tabular" in data["services"]
    assert "image" in data["services"]
    assert "diabetes" in data["services"]
    assert data["services"]["tabular"]["stats"]["type"] == "memory"
    assert 0.0 <= data["global_hit_rate"] <= 1.0


def test_admin_cache_delete_all():
    """DELETE /admin/cache deve limpar todos os caches."""
    # Primeiro faz uma predição tabular para popular o cache
    features = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
                0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
                0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
                0.2654, 0.4601, 0.1189]
    client.post("/predict/tabular/", json={"features": features})

    # Verifica que o cache foi populado
    r1 = client.get("/admin/cache")
    status_before = r1.json()

    # Limpa todos os caches
    r = client.delete("/admin/cache")
    assert r.status_code == 200
    data = r.json()
    assert "message" in data
    assert "tabular" in data["cleared"]
    assert "image" in data["cleared"]
    assert "diabetes" in data["cleared"]

    # Verifica que os caches foram limpos
    r2 = client.get("/admin/cache")
    status_after = r2.json()
    assert status_after["total_entries"] == 0


def test_admin_cache_delete_specific():
    """DELETE /admin/cache/{service} deve limpar cache específico."""
    r = client.delete("/admin/cache/tabular")
    assert r.status_code == 200
    data = r.json()
    assert data["cleared"] == ["tabular"]

    r = client.delete("/admin/cache/invalid_service")
    assert r.status_code == 404


def test_cache_decorator_hit():
    """Predições repetidas devem retornar o mesmo resultado (cache funcional)."""
    features = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
                0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
                0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
                0.2654, 0.4601, 0.1189]

    # Primeira chamada (miss, computa e cacheia)
    r1 = client.post("/predict/tabular/", json={"features": features})
    assert r1.status_code == 200
    data1 = r1.json()

    # Segunda chamada com mesmas features (deve vir do cache)
    r2 = client.post("/predict/tabular/", json={"features": features})
    assert r2.status_code == 200
    data2 = r2.json()

    # Ambos devem retornar o mesmo resultado
    assert data1 == data2
    assert data1["diagnosis"] in ["M", "B"]
