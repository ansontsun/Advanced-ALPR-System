# Define the path for the input and output files
input_file_path = 'rename_test_labels_updated.txt'
output_file_path = 'test_final.txt'

# Open the input file and read lines
with open(input_file_path, 'r') as file:
    lines = file.readlines()

# Open the output file and write the modified lines
with open(output_file_path, 'w') as file:
    for line in lines:
        # Remove the .jpg extension only at the end of the line
        if line.strip().endswith('.jpg'):
            modified_line = line.strip()[:-4] + '\n'
        else:
            modified_line = line
        file.write(modified_line)

print("File has been processed. '.jpg' extensions at the end of lines have been removed.")
