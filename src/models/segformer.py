import torch
import torch.nn.functional as F
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

class SegFormerModel:
    def __init__(self, model_name="nvidia/segformer-b5-finetuned-cityscapes-1024-1024"):
        """
        Обертка для SegFormer (HuggingFace).
        Скачивает веса автоматически при первом запуске.
        """
        print(f"[SegFormer] Loading model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.processor = SegformerImageProcessor.from_pretrained(model_name)
            
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"[SegFormer] Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"[Error] Failed to load SegFormer: {e}")
            raise e

    def predict(self, image):
        """
        Инференс одной картинки.
        
        Args:
            image (np.ndarray): Входное изображение (H, W, 3) RGB.
            
        Returns:
            np.ndarray: Маска (H, W) с ID классов Cityscapes (0..18).
        """
        original_h, original_w = image.shape[:2]
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        
        upsampled_logits = F.interpolate(
            logits, 
            size=(original_h, original_w), 
            mode="bilinear", 
            align_corners=False
        )

        pred_mask = upsampled_logits.argmax(dim=1)
        
        return pred_mask.squeeze().cpu().numpy().astype(np.uint8)

    def predict_tiled(self, image, tile_size=1024, stride=1024):
        """
        Метод для больших изображений (4K).
        Режет картинку на куски, гарантируя, что каждый кусок строго 1024x1024.
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
                
                if y2 > h:
                    y2 = h
                    y1 = h - tile_size
                    
                if x2 > w:
                    x2 = w
                    x1 = w - tile_size
                
                if y1 < 0: y1 = 0
                if x1 < 0: x1 = 0
                
                crop = image[y1:y2, x1:x2]
                
                pred_crop = self.predict(crop)
                
                full_pred[y1:y2, x1:x2] = pred_crop
                
        return full_pred