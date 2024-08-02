import string
import easyocr
import csv

# initialize OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# mappings to handle common OCR misrecognitions
dict_char_to_int = {'O': '0', 'I': '1', 'Z': '2', 'J': '3', 'A': '4', 'S': '5', 'G': '6', 'B': '8'}
dict_int_to_char = {'0': 'O', '1': 'I', '2': 'Z', '3': 'J', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}

#XXX####
#    Check if the license plate text complies with the required format, taking into account common OCR misreadings.
def license_complies_format(text):
    if len(text) != 7:
        return False
    
    # Check if the first two characters are letters (considering possible misrecognitions)
    if not all(c in string.ascii_uppercase or c in dict_int_to_char for c in text[:3]):
        return False
    
    # Check if the middle two characters are digits (considering possible misrecognitions)
    if not all(c.isdigit() or c in dict_char_to_int for c in text[3:]):
        return False
    
    return True

#   Format the license plate text by replacing commonly misidentified characters.
def format_license(text):
    mapping_letters = {'O': '0', 'I': '1', 'Z': '2', 'J': '3', 'A': '4', 'S': '5', 'G': '6', 'B': '8'}
    mapping_digits = {'0': 'O', '1': 'I', '2': 'Z', '3': 'J', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}
    formatted_text = ''
    
    for i, char in enumerate(text):
        if i < 3:  # positions are letters
            formatted_text += mapping_digits.get(char, char)
        else:  # positions are digits
            formatted_text += mapping_letters.get(char, char)
    
    return formatted_text

#    Read the license plate text from the given cropped image and check its format.
def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection
        # remove blank
        text = text.upper().replace(' ', '')

        #only if text is the same form(robust)
        if license_complies_format(text):
            #change the text into proper format
            formatted_text = format_license(text)
            return formatted_text, score

    return None, None

#   Retrieve the vehicle coordinates and ID based on the license plate coordinates.
def get_car(license_plate, vehicle_track_ids):
    lx1, ly1, lx2, ly2, score, class_id = license_plate

    # iterate through all vehicles so we can find one that encloses the license plate
    for vehicle in vehicle_track_ids:
        vx1, vy1, vx2, vy2, car_id = vehicle
        # if license plate is within the vehicle's bounding box
        if lx1 >= vx1 and ly1 >= vy1 and lx2 <= vx2 and ly2 <= vy2:
            return vehicle  # return matching vehicle's information

    return -1, -1, -1, -1, -1  # return -1s if no match

#   Write the results to a CSV file in a more structured and Pythonic way.
def write_csv(results, output_path):
    headers = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 
               'license_plate_bbox_score', 'license_number', 'license_number_score']

    with open(output_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(headers)  # Write the header

        for frame_nmr, cars in results.items():
            for car_id, car_data in cars.items():
                if 'car' in car_data and 'license_plate' in car_data:
                    car_bbox = car_data['car']['bbox']
                    plate_data = car_data['license_plate']
                    plate_bbox = plate_data['bbox']

                    # Prepare row data
                    row = [
                        frame_nmr,
                        car_id,
                        format_bbox(car_bbox),
                        format_bbox(plate_bbox),
                        plate_data.get('bbox_score', ''),  # Using .get() to handle missing keys
                        plate_data.get('text', ''),
                        plate_data.get('text_score', '')
                    ]
                    csv_writer.writerow(row)  # Write the row data

def format_bbox(bbox):
    """Format bounding box coordinates into a string."""
    return '[{} {} {} {}]'.format(*bbox)  # Unpacking the bbox tuple/list