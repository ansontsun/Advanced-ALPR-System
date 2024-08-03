from ultralytics import YOLO

def main():
    model = YOLO("yolov10m.pt")
    #model = YOLOv10.from_pretrained('jameslahm/yolov10n')
    results = model.train(data='data/Roboflow dataset/data.yaml', epochs=100, batch=8, imgsz=640)
    #model.val(data='data/data.yaml', batch=16)
    #results = model.predict(0, save = True, show = True, conf = 0.15)
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()