import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection
from transformers import SamModel, SamProcessor

class GroundedSAMModel:
    def __init__(self, config, box_threshold=0.35, text_threshold=0.25):
        """
        Grounded-SAM (HuggingFace implementation).
        
        Args:
            config (dict): Словарь с 'prompts' и 'z_order' (из utils.py).
            box_threshold (float): Порог уверенности для детекции объектов (DINO).
            text_threshold (float): Порог уверенности для текста.
        """
        self.config = config
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[Grounded-SAM] Loading models on {self.device}...")
        
        # 1. Загружаем GroundingDINO (Детекция)
        self.dino_id = "IDEA-Research/grounding-dino-base"
        self.dino_processor = AutoProcessor.from_pretrained(self.dino_id)
        self.dino_model = GroundingDinoForObjectDetection.from_pretrained(self.dino_id).to(self.device)
        
        self.sam_id = "facebook/sam-vit-base"
        self.sam_processor = SamProcessor.from_pretrained(self.sam_id)
        self.sam_model = SamModel.from_pretrained(self.sam_id).to(self.device)
        
        print("[Grounded-SAM] Ready! (Warning: Inference is slow)")

    def predict(self, image):
        """
        Args:
            image: np.ndarray (H, W, 3) RGB
        Returns:
            mask: np.ndarray (H, W) с ID классов
        """
        pil_image = Image.fromarray(image)
        h, w = image.shape[:2]
        final_mask = np.zeros((h, w), dtype=np.uint8)
        
        z_order = self.config.get("z_order", sorted(self.config["prompts"].keys()))
        
        for class_id in z_order:
            text_prompt = self.config["prompts"].get(class_id)
            if not text_prompt: continue
            if not text_prompt.endswith("."): text_prompt += "."
                
            inputs = self.dino_processor(images=pil_image, text=text_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.dino_model(**inputs)
            
            try:
                results = self.dino_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    target_sizes=[(h, w)]
                )[0]
            except TypeError:
                results = self.dino_processor.image_processor.post_process_object_detection(
                    outputs,
                    threshold=self.box_threshold,
                    target_sizes=[(h, w)]
                )[0]

            boxes = results["boxes"]
            
            if boxes.shape[0] == 0:
                continue
                
            sam_inputs = self.sam_processor(
                pil_image, 
                input_boxes=[boxes.tolist()], 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                sam_outputs = self.sam_model(**sam_inputs)
            
            masks = self.sam_processor.image_processor.post_process_masks(
                sam_outputs.pred_masks,
                sam_inputs["original_sizes"],
                sam_inputs["reshaped_input_sizes"]
            )[0]
            
            masks = masks.squeeze(1)
            
            masks_np = masks.cpu().numpy()
            
            if masks_np.ndim == 4:
                masks_np = masks_np[0]
            
            if masks_np.ndim == 3:
                class_mask_binary = np.any(masks_np, axis=0)
            else:
                class_mask_binary = masks_np
            
            if class_mask_binary.shape != final_mask.shape:
                class_mask_binary = cv2.resize(
                    class_mask_binary.astype(np.uint8), 
                    (w, h), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            final_mask[class_mask_binary] = class_id
            
        return final_mask

    def predict_tiled(self, image, tile_size=1024, stride=1024):
        """
        Версия для 4K картинок. 
        """
        h, w = image.shape[:2]
        full_pred = np.zeros((h, w), dtype=np.uint8)
        y_steps = range(0, h, stride)
        x_steps = range(0, w, stride)
        
        for y in y_steps:
            for x in x_steps:
                y1 = y
                x1 = x
                y2 = min(y + tile_size, h)
                x2 = min(x + tile_size, w)
                
                if h >= tile_size and (y2 - y1) < tile_size: y1 = h - tile_size; y2 = h
                if w >= tile_size and (x2 - x1) < tile_size: x1 = w - tile_size; x2 = w
                
                crop = image[y1:y2, x1:x2]
                pred_crop = self.predict(crop)
                full_pred[y1:y2, x1:x2] = pred_crop
                
        return full_pred