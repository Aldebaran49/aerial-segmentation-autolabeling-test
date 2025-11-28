import numpy as np

class Evaluator:
    def __init__(self, num_classes, dataset_name=None):
        """
        Класс для расчета метрик семантической сегментации (mIoU, mPA).
        
        Args:
            num_classes (int): Количество классов (8 для UAVid, 12 для Aeroscapes).
            dataset_name (str): Имя датасета ('uavid' или 'aeroscapes'). 
        """
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def _preprocess_mask(self, mask):
        """
        Внутренний метод для объединения классов перед подсчетом.
        """
        mask = mask.copy()
        
        # СПЕЦИФИКА UAVID
        # В UAVid есть "Static Car" (ID 3) и "Moving Car" (ID 7).
        # Модели обычно не различают состояние движения.
        # Поэтому они объединены в один класс (ID 7).
        if self.dataset_name == 'uavid':
            mask[mask == 3] = 7
            
        return mask

    def update(self, gt_mask, pred_mask):
        """
        Добавляет одну пару картинок в статистику.
        
        Args:
            gt_mask (np.ndarray): Истинная маска (H, W).
            pred_mask (np.ndarray): Предсказанная маска (H, W).
        """
        assert gt_mask.shape == pred_mask.shape, \
            f"Размеры не совпадают: GT {gt_mask.shape} vs Pred {pred_mask.shape}"
            
        gt = self._preprocess_mask(gt_mask)
        pred = self._preprocess_mask(pred_mask)
        
        gt = gt.flatten().astype(np.int32)
        pred = pred.flatten().astype(np.int32)
        
        valid_pixels = (gt >= 0) & (gt < self.num_classes)
        gt = gt[valid_pixels]
        pred = pred[valid_pixels]
        
        if len(gt) == 0:
            return

        label = self.num_classes * gt + pred
        count = np.bincount(label, minlength=self.num_classes**2)
        
        confusion = count.reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += confusion

    def get_results(self):
        """
        Возвращает итоговые метрики.
        """
        tp = np.diag(self.confusion_matrix)
        
        pred_sum = self.confusion_matrix.sum(axis=0)
        
        gt_sum = self.confusion_matrix.sum(axis=1)
        
        union = gt_sum + pred_sum - tp
        iou_per_class = np.divide(tp, union, out=np.full_like(tp, np.nan, dtype=float), where=union!=0)
        pa_per_class = np.divide(tp, gt_sum, out=np.full_like(tp, np.nan, dtype=float), where=gt_sum!=0)
  
        mIoU = np.nanmean(iou_per_class)
        mPA = np.nanmean(pa_per_class)
        
        total_pixels = self.confusion_matrix.sum()
        global_acc = tp.sum() / total_pixels if total_pixels > 0 else 0
        
        return {
            "mIoU": mIoU,
            "mPA": mPA,
            "Global_Acc": global_acc,
            "IoU_per_class": iou_per_class,
            "PA_per_class": pa_per_class
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))