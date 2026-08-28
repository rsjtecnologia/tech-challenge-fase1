"""Servico de predicao por imagem (PyTorch)."""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.cnn.dataset import CBIS_CLASSES, CBIS_SPLITS, index_cbis_images
from src.cnn.predict import MammoPredictor
from src.api.cache import PredictionCache, cached
from src.api.schemas import ImagePredictionResponse, GradCAMResponse
from src.cancer.reporting import _fig_to_base64

logger = logging.getLogger(__name__)

METRIC_NAMES = {
    "test_accuracy": "Acurácia",
    "test_auc": "AUC-ROC",
    "test_recall": "Recall",
    "test_precision": "Precisão",
    "test_f1": "F1-Score",
}
METRIC_COLORS = ["#3b82f6", "#6366f1", "#8b5cf6", "#a855f7", "#d946ef"]


def _distribution_graph(index: dict) -> str:
    """Barras agrupadas da distribuicao de imagens por split e classe."""
    width = 0.35
    x = np.arange(len(CBIS_SPLITS))
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, label in enumerate(CBIS_CLASSES):
        counts = [
            sum(1 for item in index.values()
                if item["split"] == split and item["label"] == label)
            for split in CBIS_SPLITS
        ]
        bars = ax.bar(
            x + i * width, counts, width,
            label=label.capitalize(),
            color="#3b82f6" if label == "benign" else "#ef4444",
        )
        for bar, value in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1, str(value),
                    ha="center", fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([split.capitalize() for split in CBIS_SPLITS])
    ax.set_ylabel("Imagens")
    ax.set_title("Distribuição de Mamografias por Split e Classe",
                 fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    return _fig_to_base64(fig)


def _metrics_graph(test_metrics: dict) -> str:
    """Barras das metricas de teste do checkpoint da CNN."""
    items = [(METRIC_NAMES[key], value * 100)
             for key, value in test_metrics.items() if key in METRIC_NAMES]
    if not items:
        return ""
    labels = [name for name, _ in items]
    values = [value for _, value in items]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=METRIC_COLORS[:len(items)], width=0.55)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Métricas de Teste — MobileNetV2 (CBIS-DDSM)",
                 fontsize=14, fontweight="bold")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%",
                ha="center", fontweight="bold")
    plt.tight_layout()
    return _fig_to_base64(fig)


class ImageService:
    """Gerencia o modelo CNN e faz predicoes em mamografias."""

    def __init__(self):
        self.predictor = None
        self._load_model()
        # Cache menor para imagens (imagens grandes ocupam mais memória)
        self.cache = PredictionCache(maxsize=64, ttl=600)

    def _load_model(self):
        try:
            self.predictor = MammoPredictor()
            logger.info("Modelo CNN (PyTorch) carregado")
        except Exception as e:
            logger.warning("Falha ao carregar CNN: %s", e)

    @cached(lambda self, image_bytes: self.cache.make_bytes_key(image_bytes, "mammo"))
    def predict_from_bytes(self, image_bytes: bytes) -> ImagePredictionResponse:
        """Recebe bytes da imagem e retorna predicao (com cache por hash MD5)."""
        if self.predictor is None:
            raise RuntimeError("Modelo CNN nao esta carregado")

        result = self.predictor.predict_from_bytes(image_bytes)
        return ImagePredictionResponse(**result)

    @cached(lambda self, image_bytes: self.cache.make_bytes_key(image_bytes, "gradcam"))
    def compute_gradcam_from_bytes(self, image_bytes: bytes) -> GradCAMResponse:
        """Recebe bytes da imagem e retorna predicao + Grad-CAM (com cache)."""
        if self.predictor is None:
            raise RuntimeError("Modelo CNN nao esta carregado")

        result = self.predictor.gradcam_from_bytes(image_bytes)
        return GradCAMResponse(**result)

    def get_info(self) -> dict:
        """Informacoes do dataset CBIS-DDSM preparado + modelo treinado."""
        index = index_cbis_images()
        splits = {split: {label: 0 for label in CBIS_CLASSES} for split in CBIS_SPLITS}
        for item in index.values():
            splits[item["split"]][item["label"]] += 1
        meta = self.predictor.get_checkpoint_meta() if self.predictor is not None else {}
        return {
            "name": "CBIS-DDSM (Mammography)",
            "source": "data/images/cbis-ddsm",
            "classes": list(CBIS_CLASSES),
            "class_names": {"benign": "Benigna", "malignant": "Maligna"},
            "total_images": len(index),
            "splits": splits,
            "model": meta,
        }

    def get_metrics(self) -> dict:
        """Tabela + grafico das metricas de teste salvas no checkpoint da CNN."""
        if self.predictor is None:
            return {"tables": {"metricas": []}, "graphs": {"comparativo": ""}}
        test_metrics = self.predictor.get_checkpoint_meta().get("test_metrics", {})
        rows = [
            {"metrica": METRIC_NAMES.get(key, key), "valor": round(value * 100, 2)}
            for key, value in test_metrics.items() if key in METRIC_NAMES
        ]
        return {
            "tables": {"metricas": rows},
            "graphs": {"comparativo": _metrics_graph(test_metrics)},
        }

    def get_eda(self) -> dict:
        """Tabelas + graficos exploratorios do conjunto de mamografias."""
        index = index_cbis_images()
        distribuicao = []
        for split in CBIS_SPLITS:
            for label in CBIS_CLASSES:
                count = sum(1 for item in index.values()
                            if item["split"] == split and item["label"] == label)
                distribuicao.append({"split": split, "classe": label, "contagem": count})

        # Dimensoes (amostra) das imagens por classe, semelhante ao describe() do cancer
        dimensoes = []
        for label in CBIS_CLASSES:
            widths, heights = [], []
            for item in index.values():
                if item["label"] != label:
                    continue
                try:
                    with Image.open(item["path"]) as img:
                        widths.append(img.width)
                        heights.append(img.height)
                except Exception:
                    continue
                if len(widths) >= 12:
                    break
            if widths:
                dimensoes.append({
                    "classe": label,
                    "largura_min": min(widths), "largura_max": max(widths),
                    "altura_min": min(heights), "altura_max": max(heights),
                    "amostra": len(widths),
                })

        resumo = [
            {
                "split": split,
                "total": sum(row["contagem"] for row in distribuicao if row["split"] == split),
            }
            for split in CBIS_SPLITS
        ]
        return {
            "tables": {
                "distribuicao": distribuicao,
                "dimensoes": dimensoes,
                "resumo": resumo,
            },
            "graphs": {"distribuicao": _distribution_graph(index)},
        }

    @property
    def is_loaded(self) -> bool:
        return self.predictor is not None
