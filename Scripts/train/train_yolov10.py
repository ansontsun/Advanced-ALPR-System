import os
import cv2
from ultralytics import YOLO
#from IPython.display import Image
import torch

def main():
    model = YOLO("yolov8s.pt")
    model.train(data='data/data.yaml', hsv_h=0.1, degrees=45, epochs=10, batch=8, imgsz=640)
    #model.train(data='data/data.yaml', epochs=20, batch=16, imgsz=640)
    #model.val(data='data/data.yaml', batch=16)


if __name__ == '__main__':
    #import multiprocessing
    #multiprocessing.freeze_support()
    main()
