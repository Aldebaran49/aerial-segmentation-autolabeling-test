import os
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.getcwd())

# Импорты датасетов
from src.datasets.aeroscapes import AeroscapesDataset
from src.datasets.uavid import UAVidDataset

# Импорты моделей
from src.models.segformer import SegFormerModel
from src.models.deeplab import DeepLabModel
from src.models.grounded_sam import GroundedSAMModel

# Импорты утилит
from src.evaluator import Evaluator
from src.utils import (
    map_mask, 
    CITYSCAPES_TO_AEROSCAPES, 
    CITYSCAPES_TO_UAVID,
    AEROSCAPES_SAM_CONFIG,
    UAVID_SAM_CONFIG
)

def main():
    # ==========================================
    # КОНФИГУРАЦИЯ ЗАПУСКА
    # ==========================================
    DATASET_NAME = "uavidmin"   # "aeroscapes" / "uavid" / "uavidmin"
    MODEL_NAME = "sam"            # "segformer" / "deeplab" / "sam"
    
    SAVE_VIS_EVERY = 50
    VIS_OUTPUT_DIR = f"output_vis_{MODEL_NAME}_{DATASET_NAME}" 
    LIMIT_IMAGES = None
    # ==========================================

    print(f"ЗАПУСК ИНФЕРЕНСА: {MODEL_NAME.upper()} на {DATASET_NAME.upper()}")

    # 1. Настройка датасета и конфигов
    if DATASET_NAME == "aeroscapes":
        root_path = os.path.join("data", "aeroscapes")
        ds = AeroscapesDataset(root_dir=root_path)
        mapping_dict = CITYSCAPES_TO_AEROSCAPES
        sam_config = AEROSCAPES_SAM_CONFIG
        num_classes = 12
        
    elif DATASET_NAME == "uavid":
        root_path = os.path.join("data", "uavid")
        ds = UAVidDataset(root_dir=root_path)
        mapping_dict = CITYSCAPES_TO_UAVID
        sam_config = UAVID_SAM_CONFIG
        num_classes = 8
    elif DATASET_NAME == "uavidmin":
        root_path = os.path.join("data", "uavid_minimum")
        ds = UAVidDataset(root_dir=root_path)
        mapping_dict = CITYSCAPES_TO_UAVID
        sam_config = UAVID_SAM_CONFIG
        num_classes = 8
        DATASET_NAME = "uavid"
    else:
        print(f"Неизвестный датасет: {DATASET_NAME}")
        return

    if len(ds) == 0:
        print("Датасет пуст!")
        return

    # 2. Инициализация модели
    try:
        if MODEL_NAME == "segformer":
            model = SegFormerModel()
        elif MODEL_NAME == "deeplab":
            model = DeepLabModel()
        elif MODEL_NAME == "sam":
            model = GroundedSAMModel(sam_config, box_threshold=0.25)
        else:
            print(f"Неизвестная модель: {MODEL_NAME}")
            return
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    # 3. Подготовка
    evaluator = Evaluator(num_classes=num_classes, dataset_name=DATASET_NAME)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    
    if not os.path.exists(VIS_OUTPUT_DIR):
        os.makedirs(VIS_OUTPUT_DIR)

    total_items = len(ds)
    if LIMIT_IMAGES is not None:
        total_items = min(total_items, LIMIT_IMAGES)

    print(f"Начата обработка {total_items} изображений")

    for i, (image_tensor, mask_tensor) in tqdm(enumerate(loader), total=total_items):
        
        if LIMIT_IMAGES is not None and i >= LIMIT_IMAGES:
            break

        image = image_tensor[0].numpy()
        gt_mask = mask_tensor[0].numpy()

        # ИНФЕРЕНС
        if MODEL_NAME == "sam":
            if DATASET_NAME == "uavid":
                pred_final = model.predict_tiled(image, tile_size=1024, stride=1024)
            else:
                pred_final = model.predict(image)
        
        else:
            if DATASET_NAME == "uavid":
                pred_city = model.predict_tiled(image, tile_size=1024, stride=1024)
            else:
                pred_city = model.predict(image)
            pred_final = map_mask(pred_city, mapping_dict)

        # МЕТРИКИ
        evaluator.update(gt_mask, pred_final)

        # ВИЗУАЛИЗАЦИЯ
        if i % SAVE_VIS_EVERY == 0:
            file_name = f"{DATASET_NAME}_{i:04d}.png"
            save_path = os.path.join(VIS_OUTPUT_DIR, file_name)
            save_visualization(image, gt_mask, pred_final, save_path, scale=30)

    # ИТОГИ
    metrics = evaluator.get_results()

    print("\n" + "="*40)
    print(f" РЕЗУЛЬТАТЫ: {MODEL_NAME} @ {DATASET_NAME}")
    print("="*40)
    print(f" Mean IoU (mIoU):       {metrics['mIoU']:.4f}")
    print(f" Mean Pixel Acc (mPA):  {metrics['mPA']:.4f}")
    print("-" * 40)
    print(" Детализация по классам (IoU):")
    
    class_map = ds.CLASSES
    sorted_classes = sorted(class_map.items(), key=lambda item: item[1])
    
    for cls_name, cls_id in sorted_classes:
        if cls_id < len(metrics['IoU_per_class']):
            iou_val = metrics['IoU_per_class'][cls_id]
            if np.isnan(iou_val):
                print(f"  {cls_name:<15} ID {cls_id}:  ----")
            else:
                print(f"  {cls_name:<15} ID {cls_id}:  {iou_val:.4f}")
    print("="*40)


def save_visualization(image, gt, pred, path, scale=20):
    """
    Сохраняет изображение
    """
    vis_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    gt_scaled = (gt * scale).astype(np.uint8)
    vis_gt = cv2.applyColorMap(gt_scaled, cv2.COLORMAP_JET)
    
    pred_scaled = (pred * scale).astype(np.uint8)
    vis_pred = cv2.applyColorMap(pred_scaled, cv2.COLORMAP_JET)
    
    cv2.putText(vis_gt, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(vis_pred, f"Prediction ({os.path.basename(path).split('_')[0]})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    combined = np.hstack([vis_img, vis_gt, vis_pred])
    
    if combined.shape[1] > 3000:
        combined = cv2.resize(combined, (0, 0), fx=0.5, fy=0.5)
        
    cv2.imwrite(path, combined)

if __name__ == "__main__":
    main()