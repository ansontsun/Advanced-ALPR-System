
import torch
from ultralytics import YOLOv10
import os

model_path = r'C:\Users\ADMIN\Desktop\Advanced-ALPR-System\models\yolov10\yolov10n.pt'


if not os.path.exists(model_path):
    raise Exception(f"Model file not found at {model_path}")

model = YOLOv10() 
model.load_state_dict(torch.load(model_path)) 

# Setup and train the model
model.train(data='data/data.yaml', epochs=50, batch=256, imgsz=640)
model.val(data='data/data.yaml', batch=256)
