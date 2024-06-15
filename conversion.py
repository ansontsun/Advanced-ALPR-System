import os
import glob

# Define the image dimensions (update these if different)
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

def convert_to_yolo_format(corners):
    x_min = min(c[0] for c in corners)
    x_max = max(c[0] for c in corners)
    y_min = min(c[1] for c in corners)
    y_max = max(c[1] for c in corners)
    
    x_center = (x_min + x_max) / 2.0 / IMG_WIDTH
    y_center = (y_min + y_max) / 2.0 / IMG_HEIGHT
    width = (x_max - x_min) / IMG_WIDTH
    height = (y_max - y_min) / IMG_HEIGHT
    
    return [x_center, y_center, width, height]

# Path to your dataset
dataset_path = 'E:/APS360/Advanced-ALPR-System/dataset/UFPR-ALPR/training'

# Iterate through all label files
for label_file in glob.glob(os.path.join(dataset_path, '**/*.txt'), recursive=True):
    with open(label_file, 'r') as file:
        lines = file.readlines()
        
        # Find the line with corners
        for line in lines:
            if line.startswith('corners:'):
                corners = line.strip().split(' ')[1:]
                corners = [list(map(int, corner.split(','))) for corner in corners]
                yolo_format = convert_to_yolo_format(corners)
                
                # Create a new label file in YOLO format
                new_label_file = label_file.replace('.txt', '_yolo.txt')
                with open(new_label_file, 'w') as yolo_file:
                    yolo_file.write(f"0 {yolo_format[0]} {yolo_format[1]} {yolo_format[2]} {yolo_format[3]}\n")
                break
