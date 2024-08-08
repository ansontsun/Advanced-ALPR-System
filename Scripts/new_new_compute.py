import re

dict_char_to_int = {
    'O': '0',
    'I': '1',
    'J': '3',
    'A': '4',
    'G': '6',
    'S': '5',
    '|': '1',
    '\\': '1'
}

dict_int_to_char = {
    '0': 'O',
    '1': 'I',
    '3': 'J',
    '4': 'A',
    '6': 'G',
    '5': 'S',
    '1': '|',
    '1': '\\'
}

def clean_ocr_result(ocr_result):
    # Remove all non-alphanumeric characters except for | and \
    cleaned_result = re.sub(r'[^a-zA-Z0-9|\\]', '', ocr_result).upper()

    # Ensure the cleaned result has exactly 7 characters
    if len(cleaned_result) != 7:
        return cleaned_result

    # Transform the first three characters if needed
    cleaned_result_list = list(cleaned_result)
    for i in range(3):
        if cleaned_result_list[i] in dict_int_to_char:
            cleaned_result_list[i] = dict_int_to_char[cleaned_result_list[i]]

    # Transform the last four characters if needed
    for i in range(3, 7):
        if cleaned_result_list[i] in dict_char_to_int:
            cleaned_result_list[i] = dict_char_to_int[cleaned_result_list[i]]

    return ''.join(cleaned_result_list)

def compare_ocr_ground_truth(file_path):
    results = []
    total_characters = 0
    correct_character_matches = 0

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

                    # Only consider results that are valid license plates
                    if len(cleaned_ocr_result) != 7 or len(ground_truth) != 7:
                        continue

                    # Compare each character of cleaned OCR result with ground truth
                    char_matches = sum(1 for gt_char, ocr_char in zip(ground_truth, cleaned_ocr_result) if gt_char == ocr_char)
                    total_characters += len(ground_truth)  # Should always be 7
                    correct_character_matches += char_matches
                    is_correct = cleaned_ocr_result == ground_truth
                    results.append((ground_truth, cleaned_ocr_result, is_correct))

    # Calculate character-level accuracy
    accuracy = (correct_character_matches / total_characters) * 100 if total_characters > 0 else 0
    return results, correct_character_matches, total_characters, accuracy

# Example usage
file_path = 'custom_ocr_results.txt' # train_ocr_results.txt
comparison_results, correct_character_matches, total_characters, accuracy = compare_ocr_ground_truth(file_path)
print(f"Correct Character Matches: {correct_character_matches}")
print(f"Total Characters: {total_characters}")
print(f"Character-Level Accuracy: {accuracy:.2f}%")
