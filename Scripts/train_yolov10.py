
import torch
from ultralytics import YOLOv10

#model = YOLOv10()
#model_path = r'C:\Users\ADMIN\Desktop\Advanced-ALPR-System\yolov10n.pt'
#model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model = YOLOv10.from_pretrained('jameslahm/yolov10n')

#model.push_to_hub("your-hf-username/yolov10-finetuned")

model.train(data='data/data.yaml', epochs=50, batch=64, imgsz=640)
model.val(data='data/data.yaml', batch=256)
