"""Carregamento e preparação do dataset clínico de SOP."""
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = Path("data/polycystic/polycystic_ovary_syndrome.zip")
EXTRACTED_WORKBOOK_PATH = Path(
    "data/polycystic/polycystic_ovary_syndrome/PCOS_data_without_infertility.xlsx"
)
WORKBOOK_NAME = "PCOS_data_without_infertility.xlsx"
SHEET_NAME = "Full_new"
TARGET_COLUMN = "PCOS (Y/N)"
IDENTIFIER_COLUMNS = {"Sl. No", "Patient File No."}


@lru_cache(maxsize=4)
def load_pcos_dataframe(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    """Lê o workbook clínico de SOP e retorna colunas numéricas utilizáveis.

    Cacheada em memória (lru_cache) porque a leitura do xlsx é relativamente
    cara e o dataset não muda durante a execução da API.
    """
    if EXTRACTED_WORKBOOK_PATH.exists():
        dataframe = pd.read_excel(EXTRACTED_WORKBOOK_PATH, sheet_name=SHEET_NAME)
    else:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset de SOP não encontrado em {dataset_path}.")
        with ZipFile(dataset_path) as archive:
            workbook = BytesIO(archive.read(WORKBOOK_NAME))
        dataframe = pd.read_excel(workbook, sheet_name=SHEET_NAME)
    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    dataframe = dataframe.loc[:, ~dataframe.columns.str.startswith("Unnamed")]
    dataframe = dataframe.drop(columns=IDENTIFIER_COLUMNS, errors="ignore")
    dataframe = dataframe.dropna(subset=[TARGET_COLUMN]).copy()

    for column in dataframe.columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    dataframe[TARGET_COLUMN] = dataframe[TARGET_COLUMN].astype(int)
    return dataframe


def load_pcos_data(test_size: float = 0.2, random_state: int = 42):
    """Retorna treino, teste, alvo e nomes das features de SOP."""
    dataframe = load_pcos_dataframe()
    feature_names = [column for column in dataframe.columns if column != TARGET_COLUMN]
    features = dataframe[feature_names]
    target = dataframe[TARGET_COLUMN]
    return (*train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    ), feature_names)


def get_pcos_info() -> dict:
    """Retorna metadados do dataset clínico de SOP."""
    dataframe = load_pcos_dataframe()
    return {
        "name": "Polycystic Ovary Syndrome Dataset",
        "samples": len(dataframe),
        "features": len(dataframe.columns) - 1,
        "classes": ["Sem SOP", "Com SOP"],
        "target": TARGET_COLUMN,
        "feature_names": [column for column in dataframe.columns if column != TARGET_COLUMN],
    }


def get_pcos_samples() -> dict:
    """Retorna uma amostra real de cada classe, sem valores ausentes."""
    dataframe = load_pcos_dataframe()
    feature_names = [column for column in dataframe.columns if column != TARGET_COLUMN]
    features = dataframe[feature_names].fillna(dataframe[feature_names].median())
    return {
        "source": str(EXTRACTED_WORKBOOK_PATH),
        "feature_names": feature_names,
        "default_sample": "negative",
        "samples": {
            "negative": {
                "features": features[dataframe[TARGET_COLUMN] == 0].iloc[0].tolist(),
                "label": "No PCOS",
            },
            "positive": {
                "features": features[dataframe[TARGET_COLUMN] == 1].iloc[0].tolist(),
                "label": "PCOS",
            },
        },
    }
