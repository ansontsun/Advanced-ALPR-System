import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from ultralytics import YOLOv10
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import easyocr
import numpy as np

# Define STNet class
class STNet(nn.Module):
    def __init__(self):
        super(STNet, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.MaxPool2d(3, stride=3),
            nn.ReLU(True)
        )
        # Regressor for the 3x2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 14 * 2, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 14 * 2)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

# Function to convert a Tensor to numpy image
def convert_image(inp):
    inp = inp.squeeze(0).cpu()
    inp = inp.detach().numpy().transpose((1, 2, 0))
    inp = 127.5 + inp / 0.0078125
    inp = inp.astype('uint8')
    return inp

# Initialize YOLOv10 model with weights
model_path = 'runs/detect/train122/train12/weights/best.pt'
model = YOLOv10(model_path)

# Path to the input image
#image_path = 'dataset/UFPR-ALPR/testing/track0097/track0097[13].png'  # Replace with the actual image path
image_path='image.png'
# Run the model on the image
results = model(source=image_path, conf=0.25, save=True)

# Get the bounding box coordinates from the results
# Assuming results is a list of detection dictionaries
if results and len(results[0].boxes) > 0:
    # Extract the first bounding box
    bbox = results[0].boxes[0]  # This may need adjustment based on the actual structure

    # Extract coordinates and convert them to integers
    x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0].tolist())

    # Print bounding box coordinates
    print(f"Bounding box coordinates: ({x_min-10}, {y_min-10}, {x_max-10}, {y_max-10})")

    # Open the image
    image = Image.open(image_path)

    # Crop the image using the bounding box coordinates
    cropped_image = image.crop((x_min-10, y_min-10, x_max+10, y_max + 10))

    # Convert PIL image to OpenCV format
    cropped_image_cv = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)

    # Load pre-trained LapSRN model
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "Scripts/unused/LapSRN_x8.pb"  # Path to the LapSRN model
    sr.readModel(path)
    sr.setModel("lapsrn", 8)  # Choose the model and scale factor

    # Enhance the image
    enhanced_image_cv = sr.upsample(cropped_image_cv)

    # Convert the enhanced image back to PIL format
    enhanced_image = Image.fromarray(cv2.cvtColor(enhanced_image_cv, cv2.COLOR_BGR2RGB))

    # Initialize the STN model
    STN = STNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    STN.to(device)
    STN.load_state_dict(torch.load('Final_STN_model.pth', map_location=lambda storage, loc: storage))
    STN.eval()

    # Preprocess the image for STN
    im = cv2.resize(np.array(enhanced_image), (94, 24), interpolation=cv2.INTER_CUBIC)
    im = np.transpose(np.float32(im), (2, 0, 1))
    im = (im - 127.5) * 0.0078125
    data = torch.from_numpy(im).float().unsqueeze(0).to(device)  # torch.Size([1, 3, 24, 94]) 

    # Apply STN
    transfer = STN(data)
    print(transfer.shape)
    transformed_img = convert_image(transfer)

    # Convert the transformed image to PIL format and invert colors
    transformed_img_pil = Image.fromarray(transformed_img)
    inverted_image = ImageOps.invert(transformed_img_pil)

    # Save the inverted image to a temporary file
    enhanced_image_path = 'enhanced_image_inverted.png'
    inverted_image.save(enhanced_image_path)

    # Display the inverted image
    plt.figure(figsize=(10, 10))
    plt.imshow(inverted_image)
    plt.axis('off')
    plt.show()

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])  # Use 'en' for English

    # Use EasyOCR to read text from the inverted image
    ocr_result = reader.readtext(enhanced_image_path, detail=1, decoder='beamsearch', beamWidth=5, contrast_ths=0.1, adjust_contrast=0.5, text_threshold=0.7, link_threshold=0.4, canvas_size=2560, mag_ratio=1.5)

    # Print the OCR results
    print("OCR Results:")
    for (bbox, text, prob) in ocr_result:
        print(f"Detected text: {text} (Probability: {prob})")
else:
    print("No detections were made.")
