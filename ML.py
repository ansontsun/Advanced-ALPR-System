from ultralytics import YOLOv10
# pip install supervision git+https://github.com/THU-MIG/yolov10.git

model = YOLOv10()

if __name__ == '__main__':
    #model.train(data='config.yaml', epochs=10, plots = True)
     model_path = 'C:/Users/mengbo/PycharmProjects/pythonProject/runs/detect/train3/weights/best.pt'
     model = YOLOv10(model_path)
     results = model("E:/UTSG/APS360/License Plate Recognition.v4-resized640_aug3x-accurate.yolov8 (1)/test/images")
     model.export()

