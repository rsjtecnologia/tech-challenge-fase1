"""Valida o pipeline de treino da CNN (train_cnn) com um dataset reduzido.

IMPORTANTE: o train_cnn grava em ``models/cnn/mobilenet_mammo.pth`` (caminho fixo).
Este script faz backup do checkpoint real antes e o restaura depois (mesmo em
caso de erro), para nunca sobrescrever o modelo de produção.

Uso:
    python scripts/validate_cnn_training.py

Recuperação manual (caso o processo seja morto DURANTE o treino):
    cp models/cnn/mobilenet_mammo.pth.bak_val models/cnn/mobilenet_mammo.pth
"""
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, ".")

CHECKPOINT = Path("models/cnn/mobilenet_mammo.pth")
BACKUP = Path("models/cnn/mobilenet_mammo.pth.bak_val")
DATASET = Path("data/images/cbis-ddsm")
PER_CLASS = 12


def build_tiny_dataset(dest: Path, per_class: int = PER_CLASS) -> None:
    """Copia um subconjunto pequeno do CBIS-DDSM para treinar rápido."""
    dest.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        for label in ("benign", "malignant"):
            source_dir = DATASET / split / label
            target_dir = dest / split / label
            target_dir.mkdir(parents=True, exist_ok=True)
            for image_path in sorted(source_dir.glob("*.jpg"))[:per_class]:
                shutil.copy2(image_path, target_dir / image_path.name)


if __name__ == "__main__":
    if not DATASET.exists():
        sys.exit("Dataset CBIS-DDSM não encontrado em data/images/cbis-ddsm.")

    had_checkpoint = False
    if CHECKPOINT.exists():
        shutil.copy2(CHECKPOINT, BACKUP)
        had_checkpoint = True
        print(f"Backup do checkpoint real -> {BACKUP}")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tiny = Path(tmp) / "cbis-tiny"
            build_tiny_dataset(tiny)
            print(f"Dataset reduzido criado: {tiny}")
            print(f"  train={len(list((tiny/'train').rglob('*.jpg')))} "
                  f"val={len(list((tiny/'val').rglob('*.jpg')))} "
                  f"test={len(list((tiny/'test').rglob('*.jpg')))}")

            from src.cnn.train import train_cnn

            t0 = time.time()
            model, metrics = train_cnn(
                epochs=2,
                finetune_epochs=1,
                batch_size=4,
                patience=3,
                pretrained=False,
                data_dir=str(tiny),
                experiment_name="mammo_cnn_validation",
            )
            print(f"\nPipeline CNN validado em {round(time.time() - t0, 1)}s")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
    finally:
        if BACKUP.exists():
            shutil.copy2(BACKUP, CHECKPOINT)
            BACKUP.unlink()
            print("Checkpoint real restaurado.")
        elif had_checkpoint:
            print("AVISO: backup não encontrado no finally.")
