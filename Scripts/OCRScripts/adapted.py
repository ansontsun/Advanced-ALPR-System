import os
from PIL import Image

# Define the directories
base_dir = 'dataset/UFPR-ALPR/testing'
output_dir = 'testing_cropped'
output_labels_file = 'testing.txt'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to parse the label file and get the corners and plate number
def parse_label_file(label_path):
    corners = None
    plate_number = None
    with open(label_path, 'r') as file:
        for line in file:
            if line.startswith('corners:'):
                corners_str = line.split(':')[1].strip()
                # Split by spaces, then split by commas and convert to integers
                corners = []
                for coord_pair in corners_str.split():
                    x, y = map(int, coord_pair.split(','))
                    corners.append((x, y))
            elif line.startswith('plate:'):
                plate_number = line.split(':')[1].strip().upper().replace(' ', '')
    return corners, plate_number

# Open the output labels file
with open(output_labels_file, 'w', encoding='utf-8') as f:
    # Traverse through all subdirectories and files in the training directory
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.png') and '_augmented' not in file:
                image_path = os.path.join(root, file)
                label_path = image_path.replace('.png', '_original.txt')
                
                if not os.path.exists(label_path):
                    continue
                
                # Parse the corners and plate number from the label file
                corners, plate_number = parse_label_file(label_path)
                if corners is None or plate_number is None:
                    continue
                
                # Open the image
                image = Image.open(image_path)
                
                # Get the bounding box coordinates
                x_min = min(corners[0][0], corners[1][0], corners[2][0], corners[3][0])
                y_min = min(corners[0][1], corners[1][1], corners[2][1], corners[3][1])
                x_max = max(corners[0][0], corners[1][0], corners[2][0], corners[3][0])
                y_max = max(corners[0][1], corners[1][1], corners[2][1], corners[3][1])
                
                # Crop the image
                cropped_image = image.crop((x_min, y_min, x_max, y_max))
                
                # Create a corresponding directory in the output directory
                relative_path = os.path.relpath(root, base_dir)
                output_image_dir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_image_dir):
                    os.makedirs(output_image_dir)
                
                # Save the cropped image to the output directory
                output_image_filename = file.replace('.png', '_cropped.png')
                output_image_path = os.path.join(output_image_dir, output_image_filename)
                cropped_image.save(output_image_path)
                print(f'Cropped image saved: {output_image_path}')
                
                # Write the label information to the output labels file
                f.write(f'{output_image_path} {plate_number}\n')

print('Image cropping and label extraction completed.')
