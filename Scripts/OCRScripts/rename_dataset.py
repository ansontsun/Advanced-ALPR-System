import os
import shutil
import pandas as pd

# Define the paths
csv_file_path = 'data/lpr.csv'  # Replace this with your actual CSV file path
images_directory = 'data/cropped_lps/cropped_lps'
renamed_images_directory = 'data/cropped_lps/renamed_images'
output_label_file = 'rename_final_labels.txt'

# Create the directory for renamed images if it doesn't exist
os.makedirs(renamed_images_directory, exist_ok=True)

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Open the output file for writing
with open(output_label_file, 'w', encoding='utf-8') as f:
    skipped_files = 0  # Counter for skipped files

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Construct the original image path
        original_image_path = os.path.join(images_directory, row['images'])
        label = row['labels']

        # Construct the new image path using the label
        _, file_extension = os.path.splitext(row['images'])
        new_image_name = f"{label}{file_extension}"
        new_image_path = os.path.join(renamed_images_directory, new_image_name)

        # Check if the original image exists
        if not os.path.exists(original_image_path):
            skipped_files += 1
            print(f"File not found: {original_image_path}. Skipping.")
            continue

        # Rename (or copy) the image to the new path
        shutil.copy(original_image_path, new_image_path)

        # Write the new image path and label to the label file
        f.write(f"{new_image_path} {label}\n")

# Output results
print(f"Label file '{output_label_file}' has been generated.")
print(f"Renamed images are stored in '{renamed_images_directory}'.")
print(f"Skipped {skipped_files} files due to missing originals.")