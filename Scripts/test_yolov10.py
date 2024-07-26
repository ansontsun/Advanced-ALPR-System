from ultralytics import YOLOv10

model_path = '/content/runs/detect/train/weights/best.pt'
model = YOLOv10(model_path)
results = model(source='/content/Fire-Detection-1/test/images', conf=0.25,save=True)

import glob
images = glob.glob('/content/runs/detect/predict2/*.jpg')
for image in images:
  display(Image(filename = image,width=400))