"""Testes do pipeline e endpoints de síndrome dos ovários policísticos."""
from src.pcos.dataset import get_pcos_info, get_pcos_samples, load_pcos_dataframe
from src.pcos.model import create_pcos_models
from src.pcos.predict import PCOSPredictor
from fastapi.testclient import TestClient
from src.api.main import app


def test_pcos_dataset_has_expected_schema():
    dataframe = load_pcos_dataframe()
    assert len(dataframe) == 541
    assert "PCOS (Y/N)" in dataframe
    assert dataframe["PCOS (Y/N)"].isin([0, 1]).all()


def test_pcos_models_are_created():
    assert len(create_pcos_models()) == 6


def test_pcos_predictor_uses_persisted_models():
    samples = get_pcos_samples()
    predictor = PCOSPredictor()
    result = predictor.predict(samples["samples"]["positive"]["features"])
    assert result["prediction"] in {"PCOS", "No PCOS"}
    assert 0 <= result["probability_positive"] <= 1


def test_pcos_api_endpoints():
    samples = get_pcos_samples()
    with TestClient(app) as client:
        response = client.get("/predict/pcos/info")
        assert response.status_code == 200
        assert response.json()["features"] == 41

        response = client.get("/predict/pcos/models")
        assert response.status_code == 200
        assert len(response.json()["available_models"]) == 6

        response = client.get("/predict/pcos/metrics")
        assert response.status_code == 200
        assert len(response.json()["tables"]["metricas"]) == 6

        response = client.post("/predict/pcos/", json={
            "features": samples["samples"]["positive"]["features"],
            "model_name": "random_forest",
        })
        assert response.status_code == 200
        assert response.json()["prediction"] in {"PCOS", "No PCOS"}


def test_pcos_eda_uses_correct_class_labels():
    """O EDA de SOP deve exibir 'Sem SOP'/'Com SOP' (não rótulos do câncer)."""
    with TestClient(app) as client:
        response = client.get("/predict/pcos/eda")
        assert response.status_code == 200
        classes = response.json()["tables"]["classes"]
        assert [entry["classe"] for entry in classes] == ["Sem SOP", "Com SOP"]
