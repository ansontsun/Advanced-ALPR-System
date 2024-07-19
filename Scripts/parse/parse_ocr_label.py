import os
import glob

def extract_ocr_labels(base_path):
    # Base directory for OCR labels, segmented into train, valid, and test
    ocr_labels_base_path = os.path.join(base_path, '..', 'ocr_labels')

    if not os.path.exists(ocr_labels_base_path):
        os.makedirs(ocr_labels_base_path)

    # Iterate over each label file already in YOLO format
    for txt_file in glob.glob(os.path.join(base_path, '**', '*.txt'), recursive=True):
        with open(txt_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        if len(lines) > 1:
            plate_number = lines[1].strip()

            # Infer train, test, or valid folder from the file path
            parts = txt_file.split(os.sep)
            category = [part for part in parts if part in ['train', 'test', 'valid']]

            if category:
                # Build the correct path for saving OCR labels
                final_save_path = os.path.join(ocr_labels_base_path, category[0])

                if not os.path.exists(final_save_path):
                    os.makedirs(final_save_path)

                # Ensure the folder structure within train/test/valid mirrors original
                inner_path = os.path.join(final_save_path, os.path.dirname(txt_file).split(category[0])[-1].lstrip(os.sep))
                if not os.path.exists(inner_path):
                    os.makedirs(inner_path)

                save_path = os.path.join(inner_path, os.path.basename(txt_file))
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(f"{plate_number}\n")
                print(f"Saved OCR label to: {save_path}")

# Specify the path to the directory containing your YOLO-formatted label files
labels_path = './data/labels'  # Adjust this path to where your YOLO labels are stored
extract_ocr_labels(labels_path)

