from ultralytics import YOLO
import cv2
import os
import numpy as np
import sys
sys.path.append('D:/Advanced-ALPR-System/scripts')
from sort.sort import *
import util_pic
from util_pic import get_car, read_license_plate, write_csv

def main():
    mot_tracker = Sort()
    # Load models
    car_model = YOLO("yolov10s.pt")
    license_plate_model = YOLO('model_weights/best.pt')
    # Directory containing images
    image_dir = 'data/UFPR-ALPR dataset/testing/'
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    vehicles = [2, 3, 5, 7]  # vehicle types
    results = {}

    # Read images
    frame_nmr = -1
    for image_file in image_files:
        frame_nmr += 1
        frame = cv2.imread(os.path.join(image_dir, image_file))
        if frame is not None:
            results[frame_nmr] = {}
            # Detect vehicles
            detections = car_model(frame)[0]
            print(detections)
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
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

    # Write results
    write_csv(results, 'outputs/picture_output.csv')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()