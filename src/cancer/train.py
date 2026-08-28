"""Pipeline de treino dos modelos de câncer de mama."""
import logging
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from sklearn.metrics import precision_score, f1_score, confusion_matrix

from src.cancer.dataset import load_cancer_data
from src.cancer.model import create_cancer_models
from src.tracking.mlflow_logger import ExperimentLogger

logger = logging.getLogger(__name__)


def train_cancer_models(
    experiment_name: str = "cancer_prediction",
    test_size: float = 0.2,
    random_state: int = 42,
):
    logger.info("Carregando dados...")
    X_train, X_test, y_train, y_test, scaler, feature_names = load_cancer_data(
        test_size=test_size, random_state=random_state
    )
    logger.info("Train: %d | Test: %d", X_train.shape[0], X_test.shape[0])
    logger.info("Features: %d", len(feature_names))

    modelos = create_cancer_models(random_state=random_state)
    resultados = {}
    modelos_treinados = {}

    os.makedirs("models/cancer", exist_ok=True)
    os.makedirs("outputs/cancer", exist_ok=True)

    for nome, modelo in modelos.items():
        logger.info("Treinando %s...", nome)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_proba),
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
        resultados[nome] = metrics
        modelos_treinados[nome] = modelo
        logger.info("  Metrics: %s", metrics)
        filename = nome.lower().replace(" ", "_")
        joblib.dump(modelo, f"models/cancer/{filename}.joblib")
        logger.info("  Modelo salvo: models/cancer/%s.joblib", filename)

    if scaler is not None:
        joblib.dump(scaler, "models/cancer/scaler.joblib")
    joblib.dump(feature_names, "models/cancer/feature_names.joblib")

    _plot_results(resultados, "outputs/cancer/comparativo.png")
    _plot_confusion_matrices(modelos_treinados, X_test, y_test, "outputs/cancer/confusion_matrices.png")

    try:
        logger_mlflow = ExperimentLogger(experiment_name)
        for nome, modelo in modelos_treinados.items():
            metrics = resultados[nome]
            params = {k: v for k, v in modelo.get_params().items()
                      if isinstance(v, (int, float, str, bool))}
            logger_mlflow.log_sklearn_model(
                model=modelo,
                metrics=metrics,
                params={**params, "dataset": "breast_cancer_wisconsin", "test_size": test_size},
                artifacts=[
                    "outputs/cancer/comparativo.png",
                    "outputs/cancer/confusion_matrices.png",
                ],
                model_name=nome.lower().replace(" ", "_"),
            )
    except Exception as e:
        logger.warning(
            "MLflow indisponível, seguindo sem tracking (%s). "
            "Inicie o servidor ou defina MLFLOW_TRACKING_URI=sqlite:///mlflow.db.", e
        )

    logger.info("Resultados salvos em models/cancer/")
    return modelos_treinados, resultados, scaler, feature_names


def _plot_results(resultados: dict, path: str):
    df = pd.DataFrame(resultados).T * 100
    plt.figure(figsize=(10, 6))
    df.plot(kind="bar")
    plt.title("Comparativo de Metricas - Cancer de Mama", fontsize=14, fontweight="bold")
    plt.ylabel("Score (%)")
    plt.xlabel("Modelo")
    plt.ylim(50, 100)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_confusion_matrices(modelos: dict, X_test, y_test, path: str):
    n = len(modelos)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (nome, modelo) in zip(axes, modelos.items()):
        y_pred = modelo.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(nome.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
        ax.set_xticks([0, 1], ["B (0)", "M (1)"])
        ax.set_yticks([0, 1], ["B (0)", "M (1)"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
    plt.suptitle("Matrizes de Confusao - Cancer de Mama", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
