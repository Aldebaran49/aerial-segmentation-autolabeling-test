from .base_dataset import BaseDataset

class AeroscapesDataset(BaseDataset):
    CLASSES = {
        'background': 0,
        'person': 1,
        'bike': 2,
        'car': 3,
        'drone': 4,
        'boat': 5,
        'animal': 6,
        'obstacle': 7,
        'construction': 8,
        'vegetation': 9,
        'road': 10,
        'sky': 11
    }