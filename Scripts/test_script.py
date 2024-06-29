import torch
from ultralytics import YOLOv10
import os
import glob

def test_model():
    # Initialize the model architecture
    model = YOLOv10.from_pretrained('jameslahm/yolov10n')
    
    # Load the trained weights
    model_path = 'runs/detect/train12/weights/best.pt'
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Extract only the model state dictionary
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    
    # Iterate over each subdirectory in the testing directory
    base_test_dir = 'dataset/UFPR-ALPR/testing'
    subdirs = [d for d in os.listdir(base_test_dir) if os.path.isdir(os.path.join(base_test_dir, d))]
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    for subdir in subdirs:
        test_path = os.path.join(base_test_dir, subdir)
        
        # Get all image files in the subdirectory
        image_files = glob.glob(os.path.join(test_path, '*.png'))  # Adjust the extension if needed
        
        # Predict for each image
        for image_file in image_files:
            results = model.predict(source=image_file, save=True, conf=0.25)
            for result in results:
                print(result)
                
                # Save results with unique names to avoid overwriting
                result.save(f"results/{subdir}_{os.path.basename(image_file)}")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    test_model()
