import re

def clean_ocr_result(ocr_result):
    # Remove all non-alphanumeric characters and convert to uppercase
    return re.sub(r'[^a-zA-Z0-9]', '', ocr_result).upper()

def compare_ocr_ground_truth(file_path, stop_label):
    results = []
    total_entries = 0
    correct_matches = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        for line in lines:
            # Assuming each line follows the format shown, split on colon and newline
            parts = line.split('\n')
            for part in parts:
                if 'Ground Truth' in part:
                    ground_truth = part.split(': ')[1].strip().upper()
                elif 'OCR Result' in part:
                    ocr_result = part.split(': ')[1].strip()
                    cleaned_ocr_result = clean_ocr_result(ocr_result)
                    
                    # Compare cleaned OCR result with ground truth
                    is_correct = cleaned_ocr_result == ground_truth
                    results.append((ground_truth, cleaned_ocr_result, is_correct))
                    
                    # Count total entries and correct matches
                    total_entries += 1
                    if is_correct:
                        correct_matches += 1
                elif 'Image' in part:
                    image_path = part.split(': ')[1].strip()
                    if image_path == stop_label:
                        # Stop processing further lines after the specified label
                        break
            else:
                # Continue if the inner loop was not broken
                continue
            # Break the outer loop if the inner loop was broken
            break

    # Calculate accuracy
    accuracy = (correct_matches / total_entries) * 100 if total_entries > 0 else 0
    return results, correct_matches, total_entries, accuracy

# Example usage
file_path = 'new_ocr_results.txt'
stop_label = 'dataset/UFPR-ALPR/testing\\track0114\\track0114[08].png'
comparison_results, correct_matches, total_entries, accuracy = compare_ocr_ground_truth(file_path, stop_label)
print(f"Correct Matches: {correct_matches}")
print(f"Total Entries: {total_entries}")
print(f"Accuracy: {accuracy:.2f}%")
