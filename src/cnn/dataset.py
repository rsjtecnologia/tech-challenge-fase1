"""Dataset loader para mamografias CBIS-DDSM usando PyTorch."""
import logging
import os
import shutil
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from PIL import Image
from typing import Callable, Optional, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class MammoDataset(Dataset):
    """
    Dataset PyTorch para mamografias CBIS-DDSM estruturado em:

        data/images/cbis-ddsm/
            train/benign/, train/malignant/
            val/benign/, val/malignant/
            test/benign/, test/malignant/
    """

    def __init__(
        self,
        base_dir: str = "data/images/cbis-ddsm",
        split: str = "train",
        img_size: Tuple[int, int] = (224, 224),
        transform: Optional[Callable] = None,
        augment: bool = False,
    ):
        self.base_dir = base_dir
        self.split = split
        self.img_size = img_size
        self.classes = ["benign", "malignant"]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples: List[Tuple[str, int]] = []
        self._load_samples()

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform(augment)

    def _load_samples(self):
        split_dir = os.path.join(self.base_dir, self.split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"Diretorio {split_dir} nao encontrado. "
                "Baixe o dataset CBIS-DDSM do Kaggle."
            )

        for cls_name in self.classes:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((
                        os.path.join(cls_dir, fname),
                        self.class_to_idx[cls_name]
                    ))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Nenhuma imagem encontrada em {split_dir}."
            )

    def _get_default_transform(self, augment: bool) -> Callable:
        if augment:
            return transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def class_counts(self) -> dict:
        counts = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            counts[self.classes[label]] += 1
        return counts

    @property
    def class_weights(self) -> torch.Tensor:
        counts = self.class_counts
        n_samples = len(self.samples)
        weights = [n_samples / (len(counts) * counts[cls]) for cls in self.classes]
        return torch.tensor(weights, dtype=torch.float)


def create_dataloaders(
    base_dir: str = "data/images/cbis-ddsm",
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = MammoDataset(
        base_dir=base_dir, split="train",
        img_size=img_size, augment=True,
    )
    val_dataset = MammoDataset(
        base_dir=base_dir, split="val",
        img_size=img_size, augment=False,
    )
    test_dataset = MammoDataset(
        base_dir=base_dir, split="test",
        img_size=img_size, augment=False,
    )

    train_targets = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(train_targets)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_targets]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=sampler, num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


CBIS_ARCHIVE_PATH = Path("data/images/Breast_cancer_Image.zip")
CBIS_EXTRACTED_PATH = Path("data/images/Breast_cancer_Image")


def prepare_cbis_ddsm_dataset(
    archive_path: Path = CBIS_ARCHIVE_PATH,
    output_dir: Path = Path("data/images/cbis-ddsm"),
    max_samples_per_class: Optional[int] = None,
    random_state: int = 42,
) -> dict:
    """Extrai JPGs rotulados do CBIS-DDSM ZIP para train/val/test.

    Os CSVs de casos fornecem a patologia e ``dicom_info.csv`` conecta a pasta
    DICOM do caso ao JPG já convertido presente no mesmo arquivo compactado.
    """
    if not archive_path.exists():
        raise FileNotFoundError(f"Arquivo CBIS-DDSM não encontrado: {archive_path}")

    with ZipFile(archive_path) as archive:
        dicom_info = pd.read_csv(archive.open("csv/dicom_info.csv"))
        image_map = (
            dicom_info[dicom_info["SeriesDescription"].eq("cropped images")]
            .dropna(subset=["PatientID", "image_path"])
            .drop_duplicates("PatientID")
            .set_index("PatientID")["image_path"]
            .str.replace("CBIS-DDSM/", "", regex=False)
            .to_dict()
        )
        entries = set(archive.namelist())
        records = []
        for split, files in {
            "train": ["csv/mass_case_description_train_set.csv", "csv/calc_case_description_train_set.csv"],
            "test": ["csv/mass_case_description_test_set.csv", "csv/calc_case_description_test_set.csv"],
        }.items():
            for file_name in files:
                cases = pd.read_csv(archive.open(file_name))
                for _, case in cases.iterrows():
                    pathology = str(case["pathology"]).upper()
                    if pathology not in {"BENIGN", "BENIGN_WITHOUT_CALLBACK", "MALIGNANT"}:
                        continue
                    patient_key = str(case["cropped image file path"]).split("/")[0]
                    image_path = image_map.get(patient_key)
                    if image_path in entries:
                        records.append((split, "malignant" if pathology == "MALIGNANT" else "benign", image_path))

        dataframe = pd.DataFrame(records, columns=["split", "label", "image_path"]).drop_duplicates("image_path")
        if dataframe.empty:
            raise RuntimeError("Não foi possível mapear casos rotulados para JPGs no CBIS-DDSM.")

        train_records = dataframe[dataframe["split"] == "train"]
        train, validation = train_test_split(
            train_records, test_size=0.2, random_state=random_state, stratify=train_records["label"]
        )
        splits = {"train": train, "val": validation, "test": dataframe[dataframe["split"] == "test"]}
        summary = {}
        for split, split_records in splits.items():
            for label in ("benign", "malignant"):
                candidates = split_records[split_records["label"] == label]["image_path"]
                if max_samples_per_class is not None:
                    candidates = candidates.iloc[:max_samples_per_class]
                target_dir = output_dir / split / label
                target_dir.mkdir(parents=True, exist_ok=True)
                for image_path in candidates:
                    image_id = sha256(image_path.encode("utf-8")).hexdigest()[:16]
                    target_path = target_dir / f"{image_id}_{Path(image_path).name}"
                    if not target_path.exists():
                        with archive.open(image_path) as source, target_path.open("wb") as destination:
                            shutil.copyfileobj(source, destination)
                summary[f"{split}_{label}"] = len(candidates)
    logger.info("Dataset CBIS-DDSM preparado em %s: %s", output_dir, summary)
    return summary


def prepare_extracted_cbis_ddsm_dataset(
    source_dir: Path = CBIS_EXTRACTED_PATH,
    output_dir: Path = Path("data/images/cbis-ddsm"),
    max_samples_per_class: Optional[int] = 500,
    random_state: int = 42,
) -> dict:
    """Cria splits da extração local usando os CSVs e JPGs do CBIS-DDSM.

    Vincula cada imagem existente ao caminho de caso correspondente no CSV.
    O arquivo ``dicom_info.csv`` contém tanto mamografias completas quanto
    recortes da lesão; ambos são imagens diagnósticas e têm rótulo de
    patologia nos CSVs de casos. Máscaras de ROI são deliberadamente
    excluídas, pois não são entradas válidas para o classificador.
    """
    csv_dir = source_dir / "csv"
    jpeg_dir = source_dir / "jpeg"
    if not csv_dir.exists() or not jpeg_dir.exists():
        raise FileNotFoundError("Extração CBIS-DDSM incompleta: esperados diretórios csv e jpeg.")

    dicom_info = pd.read_csv(csv_dir / "dicom_info.csv")
    image_records = (
        dicom_info[
            dicom_info["SeriesDescription"].isin(
                ["full mammogram images", "cropped images"]
            )
        ]
        .dropna(subset=["PatientID", "image_path"])
        .assign(
            image_path=lambda df: df["image_path"].str.replace(
                "CBIS-DDSM/jpeg/", "", regex=False
            )
        )
    )
    image_records["source_path"] = image_records["image_path"].map(jpeg_dir.__truediv__)
    image_records = image_records[image_records["source_path"].map(Path.exists)]

    records = []
    for split, filenames in {
        "train": ["mass_case_description_train_set.csv", "calc_case_description_train_set.csv"],
        "test": ["mass_case_description_test_set.csv", "calc_case_description_test_set.csv"],
    }.items():
        for filename in filenames:
            cases = pd.read_csv(csv_dir / filename)
            eligible = cases[
                cases["pathology"].astype(str).str.upper().isin(
                    {"BENIGN", "BENIGN_WITHOUT_CALLBACK", "MALIGNANT"}
                )
            ].copy()
            eligible["label"] = eligible["pathology"].astype(str).str.upper().eq("MALIGNANT").map(
                {True: "malignant", False: "benign"}
            )

            for case_path_column, series_description in (
                ("image file path", "full mammogram images"),
                ("cropped image file path", "cropped images"),
            ):
                case_images = eligible[["label", case_path_column]].copy()
                case_images["PatientID"] = case_images[case_path_column].str.split("/").str[0]
                matched = case_images.merge(
                    image_records[image_records["SeriesDescription"].eq(series_description)][
                        ["PatientID", "source_path"]
                    ],
                    on="PatientID",
                    how="inner",
                )
                records.extend(
                    (split, row.label, row.source_path)
                    for row in matched.itertuples(index=False)
                )

    dataframe = pd.DataFrame(
        records, columns=["split", "label", "image_path"]
    ).drop_duplicates("image_path")
    if dataframe.empty:
        raise RuntimeError("Os CSVs não puderam ser relacionados aos JPGs extraídos.")
    train_records = dataframe[dataframe["split"] == "train"]
    train, validation = train_test_split(
        train_records, test_size=0.2, random_state=random_state, stratify=train_records["label"]
    )
    split_frames = {"train": train, "val": validation, "test": dataframe[dataframe["split"] == "test"]}
    summary = {}
    for split, split_records in split_frames.items():
        for label in ("benign", "malignant"):
            candidates = split_records[split_records["label"] == label]["image_path"]
            if max_samples_per_class is not None:
                candidates = candidates.iloc[:max_samples_per_class]
            target_dir = output_dir / split / label
            target_dir.mkdir(parents=True, exist_ok=True)
            expected_paths = {
                target_dir
                / f"{sha256(str(image_path).encode()).hexdigest()[:16]}_{image_path.name}"
                for image_path in candidates
            }
            for existing_path in target_dir.glob("*"):
                if existing_path.is_file() and existing_path not in expected_paths:
                    existing_path.unlink()
            for image_path in candidates:
                target_path = target_dir / f"{sha256(str(image_path).encode()).hexdigest()[:16]}_{image_path.name}"
                if not target_path.exists():
                    shutil.copy2(image_path, target_path)
            summary[f"{split}_{label}"] = len(candidates)
    logger.info("Splits CBIS-DDSM extraídos preparados em %s: %s", output_dir, summary)
    return summary


def prepare_sample_from_kaggle():
    """Compatibilidade retroativa: prepara os dados fornecidos no projeto."""
    return prepare_cbis_ddsm_dataset()


CBIS_BASE_DIR = Path("data/images/cbis-ddsm")
CBIS_SPLITS = ("test", "val", "train")
CBIS_CLASSES = ("benign", "malignant")


@lru_cache(maxsize=2)
def index_cbis_images(base_dir: Path = CBIS_BASE_DIR) -> dict:
    """Indexa as mamografias preparadas (train/val/test) para a API.

    Retorna um dicionario id -> {path, label, split} com id unico
    (split_classe_arquivo), usado para servir miniaturas, predicoes,
    EDA e metricas no frontend.
    """
    index = {}
    for split in CBIS_SPLITS:
        split_dir = base_dir / split
        if not split_dir.is_dir():
            continue
        for label in CBIS_CLASSES:
            label_dir = split_dir / label
            if not label_dir.is_dir():
                continue
            image_paths = sorted(label_dir.glob("*.jpg")) + sorted(label_dir.glob("*.png"))
            for image_path in image_paths:
                image_id = f"{split}_{label}_{image_path.name}"
                index[image_id] = {
                    "path": image_path,
                    "label": label,
                    "split": split,
                }
    return index
