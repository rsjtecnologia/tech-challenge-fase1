"""Pipeline de treino da CNN com PyTorch e MLflow."""
import logging
import os
from copy import deepcopy
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from contextlib import nullcontext
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score

from src.cnn.dataset import (
    CBIS_EXTRACTED_PATH,
    MammoDataset,
    create_dataloaders,
    prepare_cbis_ddsm_dataset,
    prepare_extracted_cbis_ddsm_dataset,
)
from src.cnn.model import build_mammography_cnn, unfreeze_for_finetune, count_parameters
from src.cnn.loss import FocalLoss
from src.tracking.mlflow_logger import ExperimentLogger

logger = logging.getLogger(__name__)


def train_cnn(
    epochs: int = 25,
    batch_size: int = 16,
    img_size: tuple = (224, 224),
    data_dir: str = "data/images/cbis-ddsm",
    experiment_name: str = "mammo_cnn",
    lr_initial: float = 1e-3,
    lr_finetune: float = 1e-5,
    patience: int = 7,
    device: str = None,
    pretrained: bool = False,
    finetune_epochs: int = 1,
):
    """
    Treina a CNN com PyTorch e registra tudo no MLflow.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    logger.info("Dispositivo: %s", device)

    if not os.path.exists(data_dir):
        if CBIS_EXTRACTED_PATH.exists():
            prepare_extracted_cbis_ddsm_dataset(output_dir=Path(data_dir))
        else:
            prepare_cbis_ddsm_dataset(output_dir=Path(data_dir))

    # 1. DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        base_dir=data_dir, img_size=img_size,
        batch_size=batch_size, num_workers=0,
    )

    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    test_samples = len(test_loader.dataset)
    logger.info("Train: %d | Val: %d | Test: %d", train_samples, val_samples, test_samples)

    # 2. Modelo
    model = build_mammography_cnn(
        n_classes=1, pretrained=pretrained, freeze_backbone=pretrained
    )
    model = model.to(device)
    param_info = count_parameters(model)
    logger.info("Parametros: %s", param_info)
    writer = SummaryWriter(log_dir='outputs/cnn/tensorboard')
    # Log modelo graph com batch dummy
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1]).to(device)
    try:
        writer.add_graph(model, dummy_input)
    except Exception as e:
        logger.warning("TensorBoard graph logging skipped: %s", e)
    writer.add_text('params', str(param_info), 0)

    # 3. Loss, Optimizer, Scheduler
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_initial,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None

    # 4. Early Stopping
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None

    # 5. Historico
    history = {
        'train_loss': [], 'train_auc': [],
        'val_loss': [], 'val_auc': [],
        'lr': [],
    }

    logger.info("Iniciando treino (Feature Extraction)...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda') if torch.cuda.is_available() else nullcontext():
                outputs = model(images)
                loss = criterion(outputs, labels)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            train_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            train_preds.extend(probs.flatten())
            train_targets.extend(labels.cpu().numpy().flatten())

        # Validacao
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                with autocast(device_type='cuda') if torch.cuda.is_available() else nullcontext():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(probs.flatten())
                val_targets.extend(labels.cpu().numpy().flatten())

        # Metrica
        train_auc = roc_auc_score(train_targets, train_preds)
        val_auc = roc_auc_score(val_targets, val_preds)
        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss / train_samples)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss / val_samples)
        history['val_auc'].append(val_auc)
        history['lr'].append(current_lr)

        # TensorBoard logging
        writer.add_scalar('Loss/train', history['train_loss'][-1], epoch)
        writer.add_scalar('Loss/val', history['val_loss'][-1], epoch)
        writer.add_scalar('AUC/train', train_auc, epoch)
        writer.add_scalar('AUC/val', val_auc, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)

        logger.info(
            "Epoch [%d/%d] Train Loss: %.4f AUC: %.4f | Val Loss: %.4f AUC: %.4f | LR: %.2e",
            epoch + 1, epochs,
            history['train_loss'][-1], train_auc,
            history['val_loss'][-1], val_auc,
            current_lr,
        )

        scheduler.step()

        # Early Stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping na época %d", epoch + 1)
                break

    # Restaura melhor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 6. Fine-tuning
    logger.info("Iniciando Fine-tuning...")
    unfreeze_for_finetune(model, num_unfrozen_layers=20)

    optimizer_ft = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_finetune,
    )
    scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_ft, T_max=max(finetune_epochs, 1)
    )

    ft_history = {
        'train_loss': [], 'train_auc': [],
        'val_loss': [], 'val_auc': [],
    }

    for epoch in range(finetune_epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer_ft.zero_grad(set_to_none=True)

            with autocast(device_type='cuda') if torch.cuda.is_available() else nullcontext():
                outputs = model(images)
                loss = criterion(outputs, labels)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer_ft)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler:
                scaler.step(optimizer_ft)
                scaler.update()
            else:
                optimizer_ft.step()

            train_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            train_preds.extend(probs.flatten())
            train_targets.extend(labels.cpu().numpy().flatten())

        # Validacao
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                with autocast(device_type='cuda') if torch.cuda.is_available() else nullcontext():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(probs.flatten())
                val_targets.extend(labels.cpu().numpy().flatten())

        train_auc = roc_auc_score(train_targets, train_preds)
        val_auc = roc_auc_score(val_targets, val_preds)

        ft_history['train_loss'].append(train_loss / train_samples)
        ft_history['train_auc'].append(train_auc)
        ft_history['val_loss'].append(val_loss / val_samples)
        ft_history['val_auc'].append(val_auc)

        # TensorBoard logging (fine-tune)
        ft_epoch = epochs + epoch
        writer.add_scalar('Loss/train_ft', ft_history['train_loss'][-1], ft_epoch)
        writer.add_scalar('Loss/val_ft', ft_history['val_loss'][-1], ft_epoch)
        writer.add_scalar('AUC/train_ft', train_auc, ft_epoch)
        writer.add_scalar('AUC/val_ft', val_auc, ft_epoch)

        logger.info(
            "FT Epoch [%d/%d] Train Loss: %.4f AUC: %.4f | Val Loss: %.4f AUC: %.4f",
            epoch + 1, finetune_epochs,
            ft_history['train_loss'][-1], train_auc,
            ft_history['val_loss'][-1], val_auc,
        )

        scheduler_ft.step()

    # 7. Avaliacao final no test set
    model.eval()
    test_preds = []
    test_targets = []

    with torch.no_grad(
):
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            test_preds.extend(probs.flatten())
            test_targets.extend(labels.cpu().numpy().flatten())

    test_metrics = {
        'test_accuracy': accuracy_score(test_targets, np.array(test_preds) > 0.5),
        'test_auc': roc_auc_score(test_targets, test_preds),
        'test_recall': recall_score(test_targets, np.array(test_preds) > 0.5),
        'test_precision': precision_score(test_targets, np.array(test_preds) > 0.5),
        'test_f1': f1_score(test_targets, np.array(test_preds) > 0.5),
    }

    logger.info("Test Metrics: %s", test_metrics)

    # Log test metrics to TensorBoard
    for k, v in test_metrics.items():
        writer.add_scalar(f'Test/{k}', v, 0)
    writer.flush()

    # 8. Salva o checkpoint de inferência antes do tracking opcional.
    os.makedirs("outputs/cnn", exist_ok=True)
    _plot_history(history, ft_history, "outputs/cnn/training_history.png")
    os.makedirs("models/cnn", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': 'MobileNetV2',
        'img_size': img_size,
        'test_metrics': test_metrics,
    }, "models/cnn/mobilenet_mammo.pth")

    # 9. Log no MLflow

    experiment_logger = ExperimentLogger(experiment_name)
    params = {
        "architecture": "MobileNetV2",
        "framework": "PyTorch",
        "img_size": str(img_size),
        "batch_size": batch_size,
        "epochs_initial": epochs,
        "epochs_finetune": finetune_epochs,
        "optimizer": "Adam",
        "lr_initial": lr_initial,
        "lr_finetune": lr_finetune,
        "loss": "FocalLoss(alpha=0.75, gamma=2.0)",
        "scheduler": "CosineAnnealingLR",
        "gradient_clipping": 1.0,
        "mixed_precision": True,
        "sampler": "WeightedRandomSampler",
        "params_total": param_info["total"],
        "params_trainable": param_info["trainable"],
    }

    final_metrics = {}
    for k, v in test_metrics.items():
        final_metrics[k] = v
    for k, v in history.items():
        if k in ['train_loss', 'val_loss', 'train_auc', 'val_auc']:
            final_metrics[f'initial_{k}'] = v[-1] if v else 0
    for k, v in ft_history.items():
        if k in ['train_loss', 'val_loss', 'train_auc', 'val_auc']:
            final_metrics[f'ft_{k}'] = v[-1] if v else 0

    experiment_logger.log_pytorch_model(
        model=model,
        metrics=final_metrics,
        params=params,
        artifacts=["outputs/cnn/training_history.png"],
        model_name="mammo_cnn",
    )

    logger.info("Modelo salvo em models/cnn/mobilenet_mammo.pth")

    writer.close()
    logger.info("TensorBoard logs salvos em outputs/cnn/tensorboard")

    return model, final_metrics


def _plot_history(h1, h2, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(h1['train_loss'], label='Train (initial)')
    axes[0].plot(h1['val_loss'], label='Val (initial)')
    offset = len(h1['train_loss'])
    axes[0].plot(range(offset, offset + len(h2['train_loss'])),
                 h2['train_loss'], label='Train (fine-tune)')
    axes[0].plot(range(offset, offset + len(h2['val_loss'])),
                 h2['val_loss'], label='Val (fine-tune)')
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # AUC
    axes[1].plot(h1['train_auc'], label='Train AUC (initial)')
    axes[1].plot(h1['val_auc'], label='Val AUC (initial)')
    axes[1].plot(range(offset, offset + len(h2['train_auc'])),
                 h2['train_auc'], label='Train AUC (fine-tune)')
    axes[1].plot(range(offset, offset + len(h2['val_auc'])),
                 h2['val_auc'], label='Val AUC (fine-tune)')
    axes[1].set_title("AUC-ROC")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
