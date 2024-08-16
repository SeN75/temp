import torch
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Normalize
from transformers import SegformerForSemanticSegmentation
import numpy as np
import os

# Load your models (ensure these paths are correct)
part_seg_model = SegformerForSemanticSegmentation.from_pretrained("Mohaddz/huggingCars")
damage_seg_model = SegformerForSemanticSegmentation.from_pretrained("Mohaddz/DamageSeg")

def process_image(image_path):
    try:
        # Print information about the image being processed
        print(f"Processing image: {image_path}")
        print(f"File format: {os.path.splitext(image_path)[1]}")

        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        print(f"Image size: {image.size}")
        
        transform = Resize((512, 512))
        image = transform(image)
        image_tensor = ToTensor()(image)
        image_tensor = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        # Process with part segmentation model
        with torch.no_grad():
            part_output = part_seg_model(image_tensor).logits
        part_mask = torch.argmax(part_output, dim=1).squeeze().numpy()

        # Process with damage segmentation model
        with torch.no_grad():
            damage_output = damage_seg_model(image_tensor).logits
        damage_mask = torch.argmax(damage_output, dim=1).squeeze().numpy()

        # Create one-hot encoded vector
        num_part_classes = part_output.shape[1]
        num_damage_classes = damage_output.shape[1]
        one_hot_vector = np.zeros(num_part_classes * num_damage_classes)

        for part_id in np.unique(part_mask):
            for damage_id in np.unique(damage_mask):
                if part_id != 0 and damage_id != 0:  # Assuming 0 is background
                    vector_index = part_id * num_damage_classes + damage_id
                    one_hot_vector[vector_index] = 1

        print("One-hot vector created successfully")
        print(f"Vector shape: {one_hot_vector.shape}")
        print(f"Non-zero elements: {np.count_nonzero(one_hot_vector)}")

        return one_hot_vector.tolist()

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None