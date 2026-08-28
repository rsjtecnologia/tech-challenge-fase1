"""Inferencia da CNN para mamografias com PyTorch."""
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.cnn.model import build_mammography_cnn


class MammoPredictor:
    """Classe para predicao em mamografias usando PyTorch."""

    def __init__(
        self,
        model_path: str = "models/cnn/mobilenet_mammo.pth",
        img_size: tuple = (224, 224),
        device: str = None,
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.img_size = img_size
        self.classes = {0: "benign", 1: "malignant"}
        self._model_path = model_path

        # Carrega modelo
        self.model = build_mammography_cnn(n_classes=1, pretrained=False, freeze_backbone=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        self._checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Transform para inferencia
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def preprocess(self, img_array: np.ndarray):
        """Pre-processa imagem numpy para tensor PyTorch."""
        # Converte numpy array para PIL Image
        if img_array.ndim == 2:  # grayscale
            img = Image.fromarray(img_array).convert('RGB')
        elif img_array.shape[-1] == 1:
            img = Image.fromarray(img_array[:, :, 0]).convert('RGB')
        else:
            img = Image.fromarray(img_array).convert('RGB')

        # Aplica transform
        img_tensor = self.transform(img)
        img_batch = img_tensor.unsqueeze(0).to(self.device)
        return img_batch

    def predict(self, img_array: np.ndarray) -> dict:
        """Retorna predicao com probabilidade e classe."""
        img_tensor = self.preprocess(img_array)

        with torch.no_grad():
            output = self.model(img_tensor)
            prob = torch.sigmoid(output).item()

        class_idx = 1 if prob > 0.5 else 0
        return {
            "prediction": self.classes[class_idx],
            "probability_malignant": float(prob),
            "probability_benign": float(1 - prob),
            "confidence": float(max(prob, 1 - prob)),
            "model_used": "mobilenet_mammo_pytorch",
        }

    def compute_gradcam(self, img_array: np.ndarray) -> np.ndarray:
        """Retorna mapa de calor Grad-CAM como numpy array (H, W)."""
        img_tensor = self.preprocess(img_array)
        img_tensor.requires_grad = True

        target_layer = self.model.backbone.features[-1]

        gradients = []
        activations = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        f_handle = target_layer.register_forward_hook(forward_hook)
        b_handle = target_layer.register_full_backward_hook(backward_hook)

        output = self.model(img_tensor)
        self.model.zero_grad()
        output.backward()

        f_handle.remove()
        b_handle.remove()

        grads = gradients[0].mean(dim=[2, 3], keepdim=True)
        cam = (activations[0] * grads).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)

        cam = torch.nn.functional.interpolate(
            cam, size=self.img_size, mode='bilinear', align_corners=False
        )

        return cam.squeeze().detach().cpu().numpy()

    def gradcam_from_bytes(self, image_bytes: bytes) -> dict:
        """Recebe bytes da imagem e retorna Grad-CAM + predicao."""
        import io
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(img)
        heatmap = self.compute_gradcam(img_array)
        prediction = self.predict(img_array)
        return {
            **prediction,
            'heatmap': heatmap.tolist(),
            'heatmap_shape': list(heatmap.shape),
        }

    def get_checkpoint_meta(self) -> dict:
        """Retorna os metadados salvos junto ao checkpoint (arquitetura, img_size, metricas de teste)."""
        checkpoint = self._checkpoint
        return {
            "architecture": checkpoint.get("model_architecture", "MobileNetV2"),
            "img_size": list(checkpoint.get("img_size", self.img_size)),
            "test_metrics": checkpoint.get("test_metrics", {}),
        }

    @classmethod
    def from_file(cls, file_path: str, model_path: str = "models/cnn/mobilenet_mammo.pth"):
        """Carrega imagem do disco e prediz."""
        predictor = cls(model_path=model_path)
        img = Image.open(file_path).convert('RGB')
        img_array = np.array(img)
        return predictor.predict(img_array)

    def predict_from_bytes(self, image_bytes: bytes) -> dict:
        """Recebe bytes da imagem e retorna predicao."""
        import io
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(img)
        return self.predict(img_array)
