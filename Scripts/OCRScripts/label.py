import os

# Define the directory containing the images
image_directory = 'data/cropped_lps/renamed_test'

# Define the path for the output label file
output_label_file = 'rename_test_labels.txt'

# List all JPEG files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith('.jpg')]

# Open the output file and write directory and label
with open(output_label_file, 'w') as outfile:
    for image_file in image_files:
        # Construct the full path to the image
        full_path = os.path.join(image_directory, image_file)
        # Write the path and the label (filename) to the file
        outfile.write(f"{full_path} {image_file}\n")

print("Label file created successfully.")
