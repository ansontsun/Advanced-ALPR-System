# Define the path to the input and output label files
input_label_file = 'rename_test_labels.txt'  # This is your existing output file
output_label_file = 'rename_test_labels_updated.txt'  # This will be your new output file

# Open the input file for reading and the output file for writing
with open(input_label_file, 'r', encoding='utf-8') as infile, \
     open(output_label_file, 'w', encoding='utf-8') as outfile:

    # Iterate over each line in the input file
    for line in infile:
        # Replace backslashes with forward slashes in the line
        updated_line = line.replace('\\', '/')
        
        # Write the updated line to the output file
        outfile.write(updated_line)

print(f"Updated label file '{output_label_file}' has been generated.")
