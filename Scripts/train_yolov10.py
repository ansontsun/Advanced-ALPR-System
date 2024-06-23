
from ultralytics import YOLOv10
import torch

#model = YOLOv10()
#model_path = r'C:\Users\ADMIN\Desktop\Advanced-ALPR-System\yolov10n.pt'
#model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#model.push_to_hub("your-hf-username/yolov10-finetuned")
def main():
    model = YOLOv10.from_pretrained('jameslahm/yolov10n')
    model.train(data='data/data.yaml', epochs=50, batch=10, imgsz=640)
    model.val(data='data/data.yaml', batch=10)

###model = YOLOv10.from_pretrained('jameslahm/yolov10n')

###if __name__ == '__main__':
    ###model.train(data='data/data.yaml', epochs=10, batch=10, imgsz=640)
    #model.val(data='data/data.yaml', batch=16)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()