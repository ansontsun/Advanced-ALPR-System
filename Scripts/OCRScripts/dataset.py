import pandas as pd

# Define the paths
csv_file_path = 'data/lpr.csv'  # Replace this with your actual CSV file path
images_directory = 'cropped_lps/cropped_lps'
output_label_file = 'final_labels.txt'

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Open the output file for writing
with open(output_label_file, 'w', encoding='utf-8') as f:
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Construct the image path and label
        image_path = f"{images_directory}/{row['images']}"
        label = row['labels']
        
        # Write the image path and label to the file
        f.write(f"{image_path} {label}\n")

print(f"Label file '{output_label_file}' has been generated.")
