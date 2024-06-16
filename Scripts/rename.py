import os
import glob

# Path to your dataset
dataset_path = 'E:/APS360/Advanced-ALPR-System/dataset/UFPR-ALPR/testing'

# Rename original .txt files to _original.txt
for original_file in glob.glob(os.path.join(dataset_path, '**/*.txt'), recursive=True):
    if not original_file.endswith('_yolo.txt'):  # Skip files that are already _yolo.txt
        new_file = original_file.replace('.txt', '_original.txt')
        os.rename(original_file, new_file)
        #print(f"Renamed {original_file} to {new_file}")

# Rename _yolo.txt files back to .txt
for yolo_file in glob.glob(os.path.join(dataset_path, '**/*_yolo.txt'), recursive=True):
    new_file = yolo_file.replace('_yolo.txt', '.txt')
    os.rename(yolo_file, new_file)
    #print(f"Renamed {yolo_file} to {new_file}")

print("Renaming complete.")
