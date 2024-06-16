import os
import glob

def convert_to_yolo_format(corners, img_width, img_height):
    x_min, y_min = min(corners[::2]), min(corners[1::2])
    x_max, y_max = max(corners[::2]), max(corners[1::2])
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return [x_center, y_center, width, height]

def parse_annotations(base_path):
    processed_annotations_path = os.path.join(base_path, '..', 'processed_annotations')

    if not os.path.exists(processed_annotations_path):
        os.makedirs(processed_annotations_path)

    # search recursively within subdirectories for .txt files
    for txt_file in glob.glob(os.path.join(base_path, '**', '*.txt'), recursive=True):
        print(f"Processing file: {txt_file}")
        with open(txt_file, 'r', encoding='utf-8') as file: 
            data = file.readlines()

        details = {'corners': [], 'plate': None}
        for line in data:
            if 'corners:' in line:
                details['corners'] = list(map(int, line.split(':')[1].strip().replace(',', ' ').split()))
            elif 'plate:' in line:
                details['plate'] = line.split(':')[1].strip()

        yolo_bbox = convert_to_yolo_format(details['corners'], 1920, 1080)
        subdir = os.path.basename(os.path.dirname(txt_file))
        final_save_path = os.path.join(processed_annotations_path, subdir)

        # make sure final directory exists
        if not os.path.exists(final_save_path):
            os.makedirs(final_save_path)

        save_path = os.path.join(final_save_path, os.path.basename(txt_file).replace('.txt', '_yolo.txt'))
        print(f"Saving to: {save_path}")  
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
            f.write(f"{details['plate']}\n")

# path to the dataset
dataset_path = './data/UFPR-ALPR dataset'
parse_annotations(dataset_path)