from .base_dataset import BaseDataset

class UAVidDataset(BaseDataset):
    CLASSES = {
        'clutter': 0,
        'building': 1,
        'road': 2,
        'static car': 3,
        'tree': 4,
        'low vegetation': 5,
        'human': 6,
        'moving car': 7
    }