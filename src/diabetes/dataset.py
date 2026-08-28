"""Dataset loader para Diabetes (Pima Indians)."""
import logging
import os
import urllib.request

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

PIMA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

COLUMNS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age", "outcome"
]

FEATURE_DESCRIPTIONS = {
    "pregnancies": "Numero de gestacoes",
    "glucose": "Nivel de glicose plasmatica (mg/dL)",
    "blood_pressure": "Pressao arterial diastolica (mm Hg)",
    "skin_thickness": "Espessura da dobra cutanea (mm)",
    "insulin": "Insulina serica (mu U/mL)",
    "bmi": "Indice de massa corporal (kg/m2)",
    "diabetes_pedigree": "Funcao de pedigree de diabetes",
    "age": "Idade (anos)",
}

def download_diabetes_data(
    data_path: str = "data/diabetes/pima-indians-diabetes.csv",
    url: str = PIMA_URL,
) -> bool:
    """Baixa o dataset Pima Indians Diabetes e salva como CSV (sem cabecalho).

    Returns:
        True se o download foi bem-sucedido; False caso contrário.
    """
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    try:
        logger.info("Baixando dataset de %s ...", url)
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read()
        with open(data_path, "wb") as f:
            f.write(raw)
        logger.info("Dataset salvo em %s", data_path)
        return True
    except Exception as e:
        logger.warning("Falha ao baixar dataset: %s", e)
        return False


def load_diabetes_dataframe(
    data_path: str = "data/diabetes/pima-indians-diabetes.csv",
    download: bool = True,
) -> pd.DataFrame:
    """Carrega o dataset Pima completo (com target) para EDA.

    Se o arquivo local não existir, tenta baixar da URL oficial; caso
    contrário gera dados sintéticos apenas para demonstração.

    Returns:
        DataFrame com as 8 features + coluna 'outcome' (1 = diabetes).
    """
    if not os.path.exists(data_path):
        if download:
            download_diabetes_data(data_path)
        if not os.path.exists(data_path):
            logger.warning("Dataset não encontrado em %s", data_path)
            logger.warning("Faça o download de: %s", PIMA_URL)
            logger.warning("Salve como: %s", data_path)
            # Cria dados sinteticos para demonstracao
            logger.info("Usando dados sintéticos para demonstração...")
            np.random.seed(42)
            n_samples = 768
            X = np.random.randn(n_samples, 8)
            # Cria target com alguma correlacao
            y = (X[:, 1] + X[:, 5] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
            df = pd.DataFrame(X, columns=COLUMNS[:-1])
            df["outcome"] = y
            return df
    return pd.read_csv(data_path, names=COLUMNS)


def load_diabetes_data(
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
    download: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler], list]:
    """
    Carrega dataset Pima Indians Diabetes.

    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    df = load_diabetes_dataframe(download=download)

    X = df.drop(columns=["outcome"]).values
    y = df["outcome"].values.astype(int)
    feature_names = [c for c in df.columns if c != "outcome"]

    # Garante que nao haja NaN (o dataset original nao possui; apenas defesa)
    X = np.nan_to_num(X, nan=0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_names


def get_diabetes_info() -> dict:
    """Retorna informacoes sobre o dataset de diabetes."""
    return {
        "name": "Pima Indians Diabetes Database",
        "samples": 768,
        "features": 8,
        "classes": ["No diabetes", "Diabetes"],
        "task": "classification",
        "feature_descriptions": FEATURE_DESCRIPTIONS,
    }
