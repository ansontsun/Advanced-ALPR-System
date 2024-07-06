from ultralytics import YOLOv10
from IPython.display import Image, display

model_path = 'C:/Users/mengbo/PycharmProjects/Advanced-ALPR-System/Scripts/runs/detect/train5/weights/best.pt' # number of epoch
model = YOLOv10(model_path)
results = model(source='C:/Users/mengbo/PycharmProjects/Advanced-ALPR-System/data/UFPR-ALPR dataset/testing', conf=0.25,save=True, plots = True)
#所有的image E:/UTSG/APS360/License Plate Recognition.v4-resized640_aug3x-accurate.yolov8 (1)/test/images

# import glob
# images = glob.glob('C:/Users/mengbo/PycharmProjects/Advanced-ALPR-System/Scripts/runs/detect/predict')#where the result save
# for image in images:
#    display(Image(filename = image,width=400))