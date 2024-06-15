from torch.utils.data import DataLoader
from custom_yolo_dataset import CustomYoloDataset  # Assume this is saved in custom_yolo_dataset.py

def main():
    train_dataset = CustomYoloDataset(img_dir='./dataset/train')
    val_dataset = CustomYoloDataset(img_dir='./dataset/val')
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Now use train_loader and val_loader in the training loop
    # Replace the default DataLoader in YOLOv8 with these loaders
    
    # Your training loop here
    # ...

if __name__ == "__main__":
    main()
