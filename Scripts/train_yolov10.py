
import torch
from ultralytics import YOLOv10

# Initialize the model first
model = YOLOv10()

# Provide the complete path to the model weights file
model_path = r'C:\Users\ADMIN\Desktop\Advanced-ALPR-System\yolov10n.pt'

# Load the model weights
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Now you can continue with training or validation
model.train(data='data/data.yaml', epochs=50, batch=256, imgsz=640)
model.val(data='data/data.yaml', batch=256)
