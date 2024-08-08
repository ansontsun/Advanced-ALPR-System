import os
import shutil
import random

# Define the directories
base_directory = 'data/cropped_lps'
images_directory = os.path.join(base_directory, 'renamed_images')
train_directory = os.path.join(base_directory, 'renamed_train')
valid_directory = os.path.join(base_directory, 'renamed_valid')
test_directory = os.path.join(base_directory, 'renamed_test')

# Create directories if they do not exist
os.makedirs(train_directory, exist_ok=True)
os.makedirs(valid_directory, exist_ok=True)
os.makedirs(test_directory, exist_ok=True)

# Set the random seed for reproducibility
random_seed = 42
random.seed(random_seed)

# List all images in the renamed_images directory
all_images = os.listdir(images_directory)

# Shuffle the images to ensure random splitting
random.shuffle(all_images)

# Calculate split indices
total_images = len(all_images)
train_end = int(total_images * 0.7)
valid_end = train_end + int(total_images * 0.2)

# Split the images
train_images = all_images[:train_end]
valid_images = all_images[train_end:valid_end]
test_images = all_images[valid_end:]

# Function to move images to the target directory
def move_images(images, target_directory):
    for image in images:
        src = os.path.join(images_directory, image)
        dst = os.path.join(target_directory, image)
        shutil.move(src, dst)

# Move images to the respective directories
move_images(train_images, train_directory)
move_images(valid_images, valid_directory)
move_images(test_images, test_directory)

# Output the results
print(f"Total images: {total_images}")
print(f"Training images: {len(train_images)} stored in '{train_directory}'.")
print(f"Validation images: {len(valid_images)} stored in '{valid_directory}'.")
print(f"Test images: {len(test_images)} stored in '{test_directory}'.")
