"""Treinamento e persistência dos modelos de SOP."""
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pcos.dataset import load_pcos_data
from src.pcos.model import create_pcos_models

MODEL_DIR = Path("models/pcos")
OUTPUT_DIR = Path("outputs/pcos")


def train_pcos_models(test_size: float = 0.2, random_state: int = 42):
    """Treina, avalia e salva os modelos de classificação de SOP."""
    X_train, X_test, y_train, y_test, feature_names = load_pcos_data(test_size, random_state)
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    models = create_pcos_models(random_state)
    trained_models, results = {}, {}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        prediction = model.predict(X_test_processed)
        probability = model.predict_proba(X_test_processed)[:, 1]
        results[name] = {
            "accuracy": accuracy_score(y_test, prediction),
            "auc": roc_auc_score(y_test, probability),
            "recall": recall_score(y_test, prediction, zero_division=0),
            "precision": precision_score(y_test, prediction, zero_division=0),
            "f1": f1_score(y_test, prediction, zero_division=0),
        }
        trained_models[name] = model
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")

    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
    joblib.dump(feature_names, MODEL_DIR / "feature_names.joblib")
    _plot_results(results, OUTPUT_DIR / "comparativo.png")
    return trained_models, results, preprocessor, feature_names


def _plot_results(results: dict, path: Path) -> None:
    dataframe = pd.DataFrame(results).T.mul(100)
    axis = dataframe.plot(kind="bar", figsize=(10, 6))
    axis.set_title("Comparativo de Métricas - SOP")
    axis.set_ylabel("Score (%)")
    axis.set_xlabel("Modelo")
    axis.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
