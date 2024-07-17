import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.OneOf([
            A.GaussNoise(p=0.1),
            A.GaussianBlur(p=0.1),
        ], p=0.2),
        A.CLAHE(p=0.2),
        A.Resize(640, 640),
        A.Normalize(),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))