"""Testes do módulo CNN (mamografias) e endpoints de imagem.

Os testes de endpoint usam a fixture ``client`` com lifespan ativo (modelos
carregados uma única vez). Quando o dataset/checkpoint não existir (ex.: CI
sem os dados baixados), os testes relevantes são pulados com ``pytest.skip``.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

CBIS_DIR = Path("data/images/cbis-ddsm")
MODEL_PATH = Path("models/cnn/mobilenet_mammo.pth")


@pytest.fixture(scope="module")
def client():
    """Cliente com lifespan ativo (modelos carregados uma única vez)."""
    with TestClient(app) as test_client:
        yield test_client


class TestIndexCbisImages:
    def test_index_entries_have_expected_fields(self):
        from src.cnn.dataset import index_cbis_images

        index = index_cbis_images()
        if not index:
            pytest.skip("Dataset CBIS-DDSM não preparado")
        for image_id, item in index.items():
            assert set(item) == {"path", "label", "split"}
            assert item["label"] in {"benign", "malignant"}
            assert item["split"] in {"train", "val", "test"}
            assert item["path"].exists()

    def test_index_ids_are_unique(self):
        from src.cnn.dataset import index_cbis_images

        index = index_cbis_images()
        if not index:
            pytest.skip("Dataset CBIS-DDSM não preparado")
        assert len(set(index.keys())) == len(index)

    def test_index_splits_are_valid(self):
        from src.cnn.dataset import index_cbis_images

        index = index_cbis_images()
        if not index:
            pytest.skip("Dataset CBIS-DDSM não preparado")
        splits = {item["split"] for item in index.values()}
        assert splits.issubset({"train", "val", "test"})


class TestCheckpointMeta:
    def test_checkpoint_meta_shape(self):
        if not MODEL_PATH.exists():
            pytest.skip("Checkpoint CNN não encontrado")
        from src.cnn.predict import MammoPredictor

        predictor = MammoPredictor()
        meta = predictor.get_checkpoint_meta()
        assert "architecture" in meta
        assert "img_size" in meta
        assert isinstance(meta.get("test_metrics", {}), dict)


class TestImageEndpoints:
    def test_samples_include_split_and_counts(self, client):
        response = client.get("/predict/image/samples")
        assert response.status_code == 200
        data = response.json()
        if not data.get("samples"):
            pytest.skip("Sem imagens preparadas")
        assert data["total"] == len(data["samples"])
        assert set(data["split_counts"]) == {"train", "val", "test"}
        first = data["samples"][0]
        assert first["split"] in {"train", "val", "test"}
        assert first["image_url"].startswith("/predict/image/samples/")

    def test_info_endpoint(self, client):
        response = client.get("/predict/image/info")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "CBIS-DDSM (Mammography)"
        assert set(data["classes"]) == {"benign", "malignant"}
        assert "splits" in data and "total_images" in data

    def test_metrics_endpoint(self, client):
        response = client.get("/predict/image/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "tables" in data and "graphs" in data
        assert "metricas" in data["tables"]

    def test_eda_endpoint(self, client):
        response = client.get("/predict/image/eda")
        assert response.status_code == 200
        data = response.json()
        if not data.get("tables", {}).get("distribuicao"):
            pytest.skip("Sem imagens preparadas")
        for row in data["tables"]["distribuicao"]:
            assert row["split"] in {"train", "val", "test"}
            assert row["classe"] in {"benign", "malignant"}
        assert data["graphs"]["distribuicao"].startswith("data:image/png;base64,")

    def test_sample_image_served_with_media_type(self, client):
        response = client.get("/predict/image/samples")
        samples = response.json().get("samples", [])
        if not samples:
            pytest.skip("Sem imagens preparadas")
        image_response = client.get(samples[0]["image_url"])
        assert image_response.status_code == 200
        assert image_response.headers["content-type"] in {"image/jpeg", "image/png"}

    def test_unknown_sample_returns_404(self, client):
        response = client.get("/predict/image/samples/arquivo_inexistente.jpg")
        assert response.status_code == 404

    def test_predict_local_sample_any_split(self, client):
        """Predição funciona para amostras de qualquer split (não só test)."""
        samples = client.get("/predict/image/samples").json().get("samples", [])
        if not samples or not MODEL_PATH.exists():
            pytest.skip("Sem imagens ou modelo CNN")
        sample = next((s for s in samples if s["split"] == "train"), samples[0])
        response = client.post(f"/predict/image/mammography/sample/{sample['id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in {"benign", "malignant"}
        assert 0 <= data["probability_malignant"] <= 1
