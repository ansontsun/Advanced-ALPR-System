

from ultralytics import YOLOv10
import torch


#model.push_to_hub("your-hf-username/yolov10-finetuned")
def main():
    model = YOLOv10.from_pretrained('jameslahm/yolov10n')
    model.train(data='data/data.yaml', epochs=50, batch=10, imgsz=640)
    model.val(data='data/data.yaml', batch=10)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
