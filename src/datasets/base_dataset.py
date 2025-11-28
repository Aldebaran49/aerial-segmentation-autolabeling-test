import os
import json
import cv2
import numpy as np
import base64
import zlib
from tqdm import tqdm
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """
    Базовый класс для датасетов формата DatasetNinja/Supervisely.
    Поддерживает и Polygons, и Bitmaps.
    """
    CLASSES = {}

    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        self.img_dir = os.path.join(root_dir, 'img')
        self.ann_dir = os.path.join(root_dir, 'ann')
        self.mask_dir = os.path.join(root_dir, 'masks')
        
        self.images = []
        self.masks = []

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image dir not found: {self.img_dir}")
        if not os.path.exists(self.ann_dir):
            raise FileNotFoundError(f"Annotation dir not found: {self.ann_dir}")

        # 1. Генерация масок
        self._check_and_generate_masks()

        # 2. Сопоставление файлов
        self._match_images_and_masks()

        print(f"[{self.__class__.__name__}] Initialized. Found {len(self.images)} valid pairs.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, 0)
        
        return image, mask

    def _match_images_and_masks(self):
        if not os.path.exists(self.mask_dir):
             return

        valid_masks = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.png')])
        
        for mask_filename in valid_masks:
            file_id = os.path.splitext(mask_filename)[0]
            
            found_img_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
                possible_path = os.path.join(self.img_dir, file_id + ext)
                if os.path.exists(possible_path):
                    found_img_path = possible_path
                    break
            
            if found_img_path:
                self.images.append(found_img_path)
                self.masks.append(os.path.join(self.mask_dir, mask_filename))

    def _check_and_generate_masks(self):
        if not os.path.exists(self.mask_dir):
            os.makedirs(self.mask_dir)
            
        if len(os.listdir(self.mask_dir)) > 0:
            return

        print(f"Generating masks for {self.__class__.__name__}...")
        
        img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        for img_file in tqdm(img_files, desc="Converting JSON"):
            json_path = os.path.join(self.ann_dir, img_file + ".json")
            if not os.path.exists(json_path):
                alt_name = os.path.splitext(img_file)[0] + ".json"
                json_path = os.path.join(self.ann_dir, alt_name)
                
            if not os.path.exists(json_path):
                continue
            
            img = cv2.imread(os.path.join(self.img_dir, img_file))
            if img is None: continue
            h, w = img.shape[:2]
            
            try:
                mask = self._json_to_mask(json_path, (h, w))
                save_name = os.path.splitext(img_file)[0] + ".png"
                cv2.imwrite(os.path.join(self.mask_dir, save_name), mask)
            except Exception as e:
                print(f"Error converting {img_file}: {e}")

    def _json_to_mask(self, json_path, shape):
        """
        Парсит JSON (Polygon или Bitmap).
        """
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        objects = data.get('objects', [])
        for obj in objects:
            class_name = obj['classTitle']
            if class_name not in self.CLASSES:
                continue
            
            class_id = self.CLASSES[class_name]
            geom_type = obj.get('geometryType', 'polygon')
            
            if geom_type == 'polygon':
                points = obj['points']['exterior']
                pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], color=int(class_id))
                
            elif geom_type == 'bitmap':
                bitmap_data = obj['bitmap'].get('data')
                origin = obj['bitmap'].get('origin') # [x, y]
                
                if bitmap_data and origin:
                    zlib_bytes = base64.b64decode(bitmap_data)
                    png_bytes = zlib.decompress(zlib_bytes)
                    obj_mask = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                    
                    x, y = origin
                    obj_h, obj_w = obj_mask.shape
                    
                    y_end = min(y + obj_h, h)
                    x_end = min(x + obj_w, w)
                    
                    paste_h = y_end - y
                    paste_w = x_end - x
                    
                    if paste_h > 0 and paste_w > 0:
                        source = obj_mask[:paste_h, :paste_w]
                        target = mask[y:y_end, x:x_end]
                        target[source > 0] = int(class_id)

        return mask