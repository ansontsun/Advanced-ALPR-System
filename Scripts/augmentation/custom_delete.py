import os

def delete_augmented_files(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if "_augmented" in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted {file_path}")

if __name__ == "__main__":
    root_directory = "../dataset/UFPR-ALPR/training"
    delete_augmented_files(root_directory)
