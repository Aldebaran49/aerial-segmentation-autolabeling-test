import torch
import numpy as np
from torchvision import transforms as T
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_path = os.path.join(current_dir, 'deeplab_repo')

if repo_path not in sys.path:
    sys.path.append(repo_path)

try:
    from network.modeling import deeplabv3plus_resnet101
except ImportError:
    raise ImportError(f"Не удалось найти файлы модели в {repo_path}")

class DeepLabModel:
    def __init__(self, weights_path="weights/best_deeplabv3plus_resnet101_cityscapes_os16.pth"):
        """
        DeepLabV3+ (ResNet101), локальная версия.
        """
        print(f"[DeepLab] Initializing local model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = deeplabv3plus_resnet101(num_classes=19, output_stride=16)
        
        if not os.path.exists(weights_path):
             raise FileNotFoundError(f"Нет файла весов: {weights_path}")
             
        print(f"[DeepLab] Loading weights...")
        try:
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(weights_path, map_location=self.device)
        
        if 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        print(f"[DeepLab] Ready on {self.device}")
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image):
        """
        Обычный инференс (для Aeroscapes).
        """
        input_tensor = self.transform(image.copy()).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        pred_mask = output.max(1)[1].cpu().numpy()[0]
        return pred_mask.astype(np.uint8)

    def predict_tiled(self, image, tile_size=1024, stride=1024):
        """
        Инференс с нарезкой (для UAVid).
        """
        h, w = image.shape[:2]
        full_pred = np.zeros((h, w), dtype=np.uint8)
        
        y_steps = range(0, h, stride)
        x_steps = range(0, w, stride)
        
        for y in y_steps:
            for x in x_steps:
                y1 = y
                x1 = x
                y2 = y1 + tile_size
                x2 = x1 + tile_size
                
                if y2 > h: y2 = h; y1 = h - tile_size
                if x2 > w: x2 = w; x1 = w - tile_size
                
                if y1 < 0: y1 = 0
                if x1 < 0: x1 = 0
                
                crop = image[y1:y2, x1:x2]
                pred_crop = self.predict(crop)
                full_pred[y1:y2, x1:x2] = pred_crop
                
        return full_pred
