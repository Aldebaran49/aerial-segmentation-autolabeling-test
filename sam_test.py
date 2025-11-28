import os
import sys
import cv2
import numpy as np
import torch

sys.path.append(os.getcwd())

from src.datasets.aeroscapes import AeroscapesDataset
from src.datasets.uavid import UAVidDataset
from src.models.grounded_sam import GroundedSAMModel
from src.utils import UAVID_SAM_CONFIG, AEROSCAPES_SAM_CONFIG
from src.evaluator import Evaluator

def main():
    # ==========================================
    # НАСТРОЙКИ
    # ==========================================
    DATASET_NAME = "aeroscapes"  
    IMAGE_INDEX = 490             
    USE_TILING_FOR_UAVID = True   
    # ==========================================
    
    print(f"=== ЗАПУСК ТЕСТА GROUNDED-SAM (1 КАРТИНКА) ===")

    if DATASET_NAME == "aeroscapes":
        ds = AeroscapesDataset(root_dir=os.path.join("data", "aeroscapes"))
        config = AEROSCAPES_SAM_CONFIG
        num_classes = 12
    elif DATASET_NAME == "uavid":
        ds = UAVidDataset(root_dir=os.path.join("data", "uavid"))
        config = UAVID_SAM_CONFIG
        num_classes = 8
    else:
        print("Неверный датасет")
        return

    print(f"Датасет {DATASET_NAME} загружен.")

    if IMAGE_INDEX >= len(ds):
        print(f"Ошибка: Индекс {IMAGE_INDEX} выходит за границы")
        return
        
    image, gt_mask = ds[IMAGE_INDEX]
    print(f"Картинка #{IMAGE_INDEX}, Размер: {image.shape}")

    try:
        model = GroundedSAMModel(config, box_threshold=0.25)
    except Exception as e:
        print(f"Ошибка загрузки SAM: {e}")
        return

    print("Запуск SAM...")
    if DATASET_NAME == "uavid" and USE_TILING_FOR_UAVID:
        pred_mask = model.predict_tiled(image, tile_size=1024, stride=1024)
    else:
        pred_mask = model.predict(image)
    print("Инференс завершен.")

    print("\nРасчет метрик для текущего изображения...")
    evaluator = Evaluator(num_classes=num_classes, dataset_name=DATASET_NAME)
    evaluator.update(gt_mask, pred_mask)
    res = evaluator.get_results()

    print("-" * 30)
    print(f"mIoU: {res['mIoU']:.4f}")
    print(f"mPA:  {res['mPA']:.4f}")
    print(f"Global Acc: {res['Global_Acc']:.4f}")
    print("-" * 30)
    print("IoU по классам:")
    
    sorted_classes = sorted(ds.CLASSES.items(), key=lambda item: item[1])
    for cls_name, cls_id in sorted_classes:
        if cls_id < len(res['IoU_per_class']):
            val = res['IoU_per_class'][cls_id]
            if np.isnan(val):
                print(f"  {cls_name:<15}: ---- (нет в GT)")
            else:
                print(f"  {cls_name:<15}: {val:.4f}")
    print("-" * 30)

    save_visualization(image, gt_mask, pred_mask, f"sam_test_{DATASET_NAME}_{IMAGE_INDEX}.png")
    print(f"Картинка сохранена: sam_test_{DATASET_NAME}_{IMAGE_INDEX}.png")


def save_visualization(image, gt, pred, path):
    """Сохраняет цветную карту (Heatmap)"""
    vis_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    gt_scaled = (gt * 30).astype(np.uint8)
    vis_gt = cv2.applyColorMap(gt_scaled, cv2.COLORMAP_JET)
    
    pred_scaled = (pred * 30).astype(np.uint8)
    vis_pred = cv2.applyColorMap(pred_scaled, cv2.COLORMAP_JET)
    
    def add_label(img, text):
        cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4)
        cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        
    add_label(vis_gt, "Ground Truth")
    add_label(vis_pred, f"Grounded-SAM")

    combined = np.hstack([vis_img, vis_gt, vis_pred])
    
    if combined.shape[1] > 3000:
        scale = 3000 / combined.shape[1]
        combined = cv2.resize(combined, (0, 0), fx=scale, fy=scale)
        
    cv2.imwrite(path, combined)

if __name__ == "__main__":
    main()