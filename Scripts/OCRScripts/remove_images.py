import os

# Define the paths
images_directory = 'data/cropped_lps/renamed_images'
input_label_file = 'rename_final_labels_updated.txt'  # This is your original output file
updated_label_file = 'updated_rename_final_labels.txt'  # This will be your new updated output file

# Get a list of all images in the renamed_images directory
existing_images = set(os.listdir(images_directory))

# Open the input file for reading and the updated output file for writing
with open(input_label_file, 'r', encoding='utf-8') as infile, \
     open(updated_label_file, 'w', encoding='utf-8') as outfile:

    # Iterate over each line in the input file
    for line in infile:
        # Extract the image path from the line
        image_path, label = line.strip().split(maxsplit=1)

        # Extract just the image name from the path
        image_name = os.path.basename(image_path)

        # Check if the image exists in the directory
        if image_name in existing_images:
            # Write the line to the updated file if the image exists
            outfile.write(line)
        else:
            print(f"Image not found, removing line: {line.strip()}")

print(f"Updated label file '{updated_label_file}' has been generated.")
