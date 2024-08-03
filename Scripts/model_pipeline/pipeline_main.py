from ultralytics import YOLO
import cv2
import sys
sys.path.append('D:/Advanced-ALPR-System/scripts')

from sort.sort import *
import pipeline_helper
from pipeline_helper import get_car, read_license_plate, write_csv

def main(video_path, output_path):
    mot_tracker = Sort()
    #load model
    car_model = YOLO("yolov10l.pt")
    license_plate_model = YOLO('model_weights/best_v8s.pt')
    #load video
    cap = cv2.VideoCapture(video_path)

    vehicles = [2, 3, 5, 7] #vehicle types
    results = {}
    frame_nmr = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results[frame_nmr] = {}
        # detect vehicles
        detections = car_model(frame)[0]
        detections_ = []
        vehicle_detections = [d for d in detections.boxes.data.tolist() if int(d[-1]) in vehicles]
        # track vehicles
        track_ids = mot_tracker.update(np.array(vehicle_detections))
        # detect license plates
        license_plates = license_plate_model(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number through threshold image
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                # !!! make a dictionary
                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                'license_plate': {'bbox': [x1, y1, x2, y2],'text': license_plate_text,
                                                                'bbox_score': score,'text_score': license_plate_text_score}}
        frame_nmr += 1

    # write results
    cap.release()
    write_csv(results, output_path)
    print("Processing complete.")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    video_path = 'D:\\Advanced-ALPR-System\\videos\\test_best.mp4'
    output_path = 'D:\\Advanced-ALPR-System\\outputs\\final_output_v8s.csv'
    main(video_path, output_path)