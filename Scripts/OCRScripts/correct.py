import os

# Define the paths to your label files
train_label_file = 'train_labels.txt'
valid_label_file = 'valid_labels.txt'

# Function to fix paths in a label file
def fix_label_file(label_file):
    # Temporary file to store corrected lines
    temp_file = label_file + '.tmp'
    
    with open(label_file, 'r', encoding='utf-8') as infile, open(temp_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Split the line into the image path and label
            image_path, label = line.strip().split(' ', 1)
            
            # Normalize path to use forward slashes and remove leading 'data/'
            corrected_path = image_path.replace('\\', '/').replace('data/', '')
            
            # Write the corrected path and label to the temporary file
            outfile.write(f"{corrected_path} {label}\n")
    
    # Replace the original file with the corrected file
    os.replace(temp_file, label_file)
    print(f"Fixed paths in: {label_file}")

# Fix paths in both label files
fix_label_file(train_label_file)
fix_label_file(valid_label_file)
