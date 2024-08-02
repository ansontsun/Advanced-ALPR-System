from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")
    #model = YOLOv10.from_pretrained('jameslahm/yolov10n')
    results = model.train(data='data/data.yaml', epochs=40, batch=16, imgsz=640)
    #model.val(data='data/data.yaml', batch=16)
    #results = model.predict(0, save = True, show = True, conf = 0.15)
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()