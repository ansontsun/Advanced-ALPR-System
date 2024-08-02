from ultralytics import YOLO
import cv2
import os
import numpy as np
import sys
from sort.sort import *
import util
from util import get_car, read_license_plate, write_csv

sys.path.append('D:/Advanced-ALPR-System/scripts')
#sys.path.append('/Users/ansonsun/Documents/aps360/project/Advanced-ALPR-System/Scripts')
def process_image(frame, car_model, license_plate_model, mot_tracker, frame_nmr, results):
    results[frame_nmr] = {}
    
    # Detect vehicles
    detections = car_model(frame)[0]
    print(detections)
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in [2, 3, 5, 7]:  # Vehicle types
            detections_.append([x1, y1, x2, y2, score])

    # Track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))

    # Detect license plates
    license_plates = license_plate_model(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Assign license plate to car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id != -1:
            # Crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

            # Process license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # Read license plate number through threshold image
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

            # Make a dictionary
            if license_plate_text is not None:
                results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                              'license_plate': {'bbox': [x1, y1, x2, y2], 'text': license_plate_text,
                                                                'bbox_score': score, 'text_score': license_plate_text_score}}

def main():
    mot_tracker = Sort()
    # Load models
    car_model = YOLO("models/yolov10s.pt")
    license_plate_model = YOLO('model_weights/best.pt')

    # Directory containing images
    image_dir = 'images/'
    
    vehicles = [2, 3, 5, 7]  # vehicle types
    results = {}

    # Read images recursively from all subfolders
    frame_nmr = -1
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('.jpg', '.png')):
                frame_nmr += 1
                image_path = os.path.join(root, file)
                frame = cv2.imread(image_path)
                if frame is not None:
                    process_image(frame, car_model, license_plate_model, mot_tracker, frame_nmr, results)

    # Write results
    write_csv(results, 'outputs/final_output.csv')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()