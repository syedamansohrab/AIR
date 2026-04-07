import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pickle
import numpy as np
import warnings

# Suppress annoying warnings
warnings.filterwarnings("ignore")

IMAGE_DIR = "extracted_images"
INDEX_FILE = "visual_index.pkl"

def build_index():
    print("🤖 Downloading & Loading ResNet50 AI Model...")
    
    # 1. Load pre-trained ResNet50 (It will download the model weights the first time)
    resnet = models.resnet50(pretrained=True)
    
    # 2. Chop off the final layer so it outputs features instead of classifications
    feature_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    feature_extractor.eval() # Set to evaluation mode (no training)

    # 3. Standard image processing (ResNet requires 224x224 sizing)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    visual_index = {}
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total = len(image_files)

    print(f"🧠 Processing {total} images. Converting pixels to math...")

    # 4. Loop through every image and extract its mathematical fingerprint
    with torch.no_grad():
        for i, filename in enumerate(image_files):
            if i % 50 == 0 and i != 0:
                print(f"   ...processed {i}/{total} images...")
            
            try:
                img_path = os.path.join(IMAGE_DIR, filename)
                # Convert to RGB (patents are often black and white/grayscale)
                img = Image.open(img_path).convert('RGB')
                img_t = preprocess(img)
                batch_t = torch.unsqueeze(img_t, 0)
                
                # Push through ResNet50
                features = feature_extractor(batch_t)
                
                # Flatten the tensor into a simple numpy array of 2,048 numbers
                feature_vector = features.numpy().flatten()
                
                # Save it in our dictionary
                visual_index[filename] = feature_vector
            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")

    # 5. Save the entire dictionary to a file
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump(visual_index, f)
        
    print("✅ Visual Index successfully created and saved as 'visual_index.pkl'!")

if __name__ == "__main__":
    build_index()