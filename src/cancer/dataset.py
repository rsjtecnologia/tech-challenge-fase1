"""Dataset loader para Câncer de Mama (Breast Cancer Wisconsin)."""
import logging
import os
import io
import urllib.request

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer as sklearn_load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Fonte original do dataset (UCI Machine Learning Repository)
CANCER_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/wdbc.data"
)

# As 30 features (sem a coluna 'diagnosis')
FEATURE_NAMES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension",
]

# Mapeamento solicitado: M (maligno) -> 1, B (benigno) -> 0
LABEL_MAP = {"M": 1, "B": 0}
LABEL_NAMES = ["B (benigno)", "M (maligno)"]

FEATURE_DESCRIPTIONS = {
    "mean radius": "Raio médio do núcleo",
    "mean texture": "Textura média do núcleo",
    "mean perimeter": "Perímetro médio do núcleo",
    "mean area": "Área média do núcleo",
    "mean smoothness": "Suavidade média",
    "mean compactness": "Compacidade média",
    "mean concavity": "Concavidade média",
    "mean concave points": "Pontos côncavos médios",
    "mean symmetry": "Simetria média",
    "mean fractal dimension": "Dimensão fractal média",
    "radius error": "Erro padrão do raio",
    "texture error": "Erro padrão da textura",
    "perimeter error": "Erro padrão do perímetro",
    "area error": "Erro padrão da área",
    "smoothness error": "Erro padrão da suavidade",
    "compactness error": "Erro padrão da compacidade",
    "concavity error": "Erro padrão da concavidade",
    "concave points error": "Erro padrão dos pontos côncavos",
    "symmetry error": "Erro padrão da simetria",
    "fractal dimension error": "Erro padrão da dimensão fractal",
    "worst radius": "Pior raio (maior valor)",
    "worst texture": "Pior textura (maior valor)",
    "worst perimeter": "Pior perímetro (maior valor)",
    "worst area": "Pior área (maior valor)",
    "worst smoothness": "Pior suavidade (maior valor)",
    "worst compactness": "Pior compacidade (maior valor)",
    "worst concavity": "Pior concavidade (maior valor)",
    "worst concave points": "Piores pontos côncavos (maior valor)",
    "worst symmetry": "Pior simetria (maior valor)",
    "worst fractal dimension": "Pior dimensão fractal (maior valor)",
}


def download_cancer_data(data_path: str = "data/breast_cancer_wisconsin.csv") -> bool:
    """
    Baixa o dataset Breast Cancer Wisconsin da UCI e salva como CSV.

    Returns:
        True se o download foi bem-sucedido; False caso contrário.
    """
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    try:
        logger.info("Baixando dataset de %s ...", CANCER_URL)
        with urllib.request.urlopen(CANCER_URL, timeout=30) as resp:
            raw = resp.read().decode("utf-8")

        # O arquivo da UCI tem 32 colunas: id + diagnosis + 30 features (sem header)
        df = pd.read_csv(io.StringIO(raw), header=None)
        df = df.iloc[:, 1:]  # remove a coluna 'id'
        df.columns = ["diagnosis"] + FEATURE_NAMES
        df.to_csv(data_path, index=False)
        logger.info("Dataset salvo em %s (%d amostras)", data_path, len(df))
        return True
    except Exception as e:
        logger.warning("Falha ao baixar dataset: %s", e)
        return False


def _ensure_dataset_file(data_path: str = "data/breast_cancer_wisconsin.csv") -> str:
    """
    Garante que o CSV do dataset existe em disco.

    Ordem: arquivo local -> scikit-learn (bundled, offline) -> download UCI.
    Retorna o caminho do arquivo disponível.
    """
    if os.path.exists(data_path):
        return data_path

    # Fallback offline/deterministico: o dataset do sklearn é o mesmo
    # Breast Cancer Wisconsin (569 amostras, 30 features, M/B).
    data = sklearn_load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df.insert(0, "diagnosis", np.where(data.target == 0, "M", "B"))
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path, index=False)
    logger.info("Dataset do sklearn salvo em %s", data_path)
    return data_path


def load_cancer_dataframe(
    data_path: str = "data/breast_cancer_wisconsin.csv",
    download: bool = True,
) -> pd.DataFrame:
    """
    Carrega o dataset completo (com target numérico M=1/B=0) para EDA.

    Se o arquivo local não existir, tenta baixar da UCI (quando download=True);
    caso contrário usa o dataset bundled do scikit-learn.

    Returns:
        DataFrame com coluna 'diagnosis' (original), 'target' (M=1, B=0)
        e as 30 features.
    """
    if not os.path.exists(data_path):
        if download:
            download_cancer_data(data_path)
        if not os.path.exists(data_path):
            _ensure_dataset_file(data_path)
    df = pd.read_csv(data_path)
    df["target"] = df["diagnosis"].map(LABEL_MAP)
    return df


def load_cancer_data(
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
    download: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler], list]:
    """
    Carrega o dataset Breast Cancer Wisconsin.

    A coluna 'diagnosis' é transformada em: M -> 1 (maligno), B -> 0 (benigno).

    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    df = load_cancer_dataframe(download=download)

    feature_names = FEATURE_NAMES if all(f in df.columns for f in FEATURE_NAMES) \
        else [c for c in df.columns if c not in ("diagnosis", "target")]

    X = df[feature_names].values
    y = df["target"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_names


def get_cancer_info() -> dict:
    """Retorna informações sobre o dataset de câncer de mama."""
    return {
        "name": "Breast Cancer Wisconsin (Diagnostic)",
        "source": "UCI Machine Learning Repository",
        "url": CANCER_URL,
        "samples": 569,
        "features": 30,
        "classes": LABEL_NAMES,
        "encoding": {"M": 1, "B": 0},
        "task": "classification",
        "feature_descriptions": FEATURE_DESCRIPTIONS,
    }
