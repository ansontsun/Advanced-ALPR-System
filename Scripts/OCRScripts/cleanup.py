import os

# Path to the label file and directory containing images
label_file_path = 'updated_rename_final_labels.txt'
directory_path = 'data/cropped_lps/renamed_images'

# Read all valid filenames from the label file
valid_filenames = set()
with open(label_file_path, 'r') as file:
    for line in file:
        # Extract the filename part before the space
        filename = line.split(' ')[0].split('/')[-1]
        valid_filenames.add(filename)

# Get all files in the directory
all_files = os.listdir(directory_path)

# Delete files that are not in the valid filenames list
for file in all_files:
    if file not in valid_filenames:
        os.remove(os.path.join(directory_path, file))
        print(f"Deleted: {file}")

print("Cleanup completed.")
