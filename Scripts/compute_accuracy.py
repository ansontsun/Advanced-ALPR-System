import re

def clean_ocr_result(ocr_result):
    # Remove all non-alphanumeric characters and convert to uppercase
    return re.sub(r'[^a-zA-Z0-9]', '', ocr_result).upper()

def compare_ocr_ground_truth(file_path):
    results = []
    total_entries = 0
    correct_matches = 0

    with open(file_path, 'r',encoding='utf-8') as file:
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

    # Calculate accuracy
    accuracy = (correct_matches / total_entries) * 100 if total_entries > 0 else 0
    return results, correct_matches, total_entries, accuracy

# Example usage
file_path = 'ocr_results.txt'
comparison_results, correct_matches, total_entries, accuracy = compare_ocr_ground_truth(file_path)
print(f"Correct Matches: {correct_matches}")
print(f"Total Entries: {total_entries}")
print(f"Accuracy: {accuracy:.2f}%")
