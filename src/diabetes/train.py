"""Pipeline de treino dos modelos de diabetes."""
import logging
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from sklearn.metrics import precision_score, f1_score

from src.diabetes.dataset import load_diabetes_data
from src.diabetes.model import create_diabetes_models
from src.tracking.mlflow_logger import ExperimentLogger

logger = logging.getLogger(__name__)


def train_diabetes_models(
    experiment_name: str = "diabetes_prediction",
    test_size: float = 0.2,
    random_state: int = 42,
):
    logger.info("Carregando dados...")
    X_train, X_test, y_train, y_test, scaler, feature_names = load_diabetes_data(
        test_size=test_size, random_state=random_state
    )
    logger.info("Train: %d | Test: %d", X_train.shape[0], X_test.shape[0])
    logger.info("Features: %s", feature_names)

    modelos = create_diabetes_models(random_state=random_state)
    resultados = {}
    modelos_treinados = {}

    os.makedirs("models/diabetes", exist_ok=True)
    os.makedirs("outputs/diabetes", exist_ok=True)

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
        joblib.dump(modelo, f"models/diabetes/{filename}.joblib")
        logger.info("  Modelo salvo: models/diabetes/%s.joblib", filename)

    if scaler is not None:
        joblib.dump(scaler, "models/diabetes/scaler.joblib")
    joblib.dump(feature_names, "models/diabetes/feature_names.joblib")

    _plot_results(resultados, "outputs/diabetes/comparativo.png")

    logger_mlflow = ExperimentLogger(experiment_name)
    for nome, modelo in modelos_treinados.items():
        metrics = resultados[nome]
        params = {k: v for k, v in modelo.get_params().items()
                  if isinstance(v, (int, float, str, bool))}
        logger_mlflow.log_sklearn_model(
            model=modelo,
            metrics=metrics,
            params={**params, "dataset": "pima_indians", "test_size": test_size},
            artifacts=["outputs/diabetes/comparativo.png"],
            model_name=nome.lower().replace(" ", "_"),
        )

    logger.info("Resultados salvos em models/diabetes/")
    return modelos_treinados, resultados, scaler, feature_names


def _plot_results(resultados: dict, path: str):
    df = pd.DataFrame(resultados).T * 100
    plt.figure(figsize=(10, 6))
    df.plot(kind="bar")
    plt.title("Comparativo de Metricas - Diabetes", fontsize=14, fontweight="bold")
    plt.ylabel("Score (\%)")
    plt.xlabel("Modelo")
    plt.ylim(50, 100)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
