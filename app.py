from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import json
import torch
from PIL import Image
import numpy as np
import tensorflow as tf
from transformers import SegformerForSemanticSegmentation, AutoFeatureExtractor
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import uuid
import base64

app = Flask(__name__)

# Load models
part_seg_model = SegformerForSemanticSegmentation.from_pretrained("Mohaddz/huggingCars")
damage_seg_model = SegformerForSemanticSegmentation.from_pretrained("Mohaddz/DamageSeg")
feature_extractor = AutoFeatureExtractor.from_pretrained("Mohaddz/huggingCars")
dl_model = tf.keras.models.load_model('improved_car_damage_prediction_model.h5')

# Load parts list
with open('cars117.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
all_parts = sorted(list(set(part for entry in data.values() for part in entry.get('replaced_parts', []))))

# In-memory storage for assessment history
assessment_history = []


def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        part_features = part_seg_model(**inputs).logits
        damage_features = damage_seg_model(**inputs).logits
    
    # Convert to numpy and remove batch dimension
    part_features = part_features.squeeze().numpy()
    damage_features = damage_features.squeeze().numpy()
    
    # Reduce feature map size
    part_features_mean = part_features.mean(axis=(1, 2))
    damage_features_mean = damage_features.mean(axis=(1, 2))
    
    # Concatenate features
    combined_features = np.concatenate([part_features_mean, damage_features_mean])
    
    return combined_features, part_features, damage_features
def convert_img_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_string}"
def create_heatmap(features):
    # Sum across channels and normalize
    heatmap = np.sum(features, axis=0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = np.uint8(255 * heatmap)
    return heatmap

def overlay_heatmap(image, heatmap, alpha=0.4):
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

def predict_parts_to_replace(input_vector, threshold=0.1):
    prediction = dl_model.predict(np.array([input_vector]))
    predicted_parts_with_probs = [
        (all_parts[i], float(prob))
        for i, prob in enumerate(prediction[0])
        if prob > threshold
    ]
    predicted_parts_with_probs.sort(key=lambda x: x[1], reverse=True)
    
    predicted_parts = [part for part, _ in predicted_parts_with_probs]
    probabilities = [prob for _, prob in predicted_parts_with_probs]
    
    return predicted_parts, probabilities, prediction[0]
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Generate a unique ID for this assessment
    assessment_id = str(uuid.uuid4())
    
    # Create result directory
    result_path = os.path.join('static', 'results', assessment_id)
    os.makedirs(result_path, exist_ok=True)
    
    # Save the original uploaded file
    original_filename = secure_filename(file.filename)
    original_path = os.path.join(result_path, original_filename)
    file.save(original_path)
    
    # Process the image
    original_image = cv2.imread(original_path)
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    
    # Prepare input for the model
    inputs = feature_extractor(images=image_pil, return_tensors="pt")
    
    # Get damage segmentation
    with torch.no_grad():
        damage_output = damage_seg_model(**inputs).logits
    damage_features = damage_output.squeeze().numpy()
    
    # Create damage segmentation heatmap
    damage_heatmap = create_heatmap(damage_features)
    damage_heatmap_resized = cv2.resize(damage_heatmap, (original_image.shape[1], original_image.shape[0]))
    damage_overlay = overlay_heatmap(original_image, damage_heatmap_resized)
    
    # Create annotated damage image (semi-transparent red overlay)
    damage_mask = np.argmax(damage_features, axis=0)
    damage_mask_resized = cv2.resize(damage_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = np.zeros_like(original_image)
    overlay[damage_mask_resized > 0] = [0, 0, 255]  # Red color for damage
    annotated_image = cv2.addWeighted(original_image, 1, overlay, 0.5, 0)
    
    # Process for part prediction and heatmap
    input_vector, part_features, _ = process_image(original_path)
    part_heatmap = create_heatmap(part_features)
    part_heatmap_resized = cv2.resize(part_heatmap, (original_image.shape[1], original_image.shape[0]))
    part_overlay = overlay_heatmap(original_image, part_heatmap_resized)
    
    # Save the results
    cv2.imwrite(os.path.join(result_path, 'annotated_damage.jpg'), annotated_image)
    cv2.imwrite(os.path.join(result_path, 'damage_heatmap.jpg'), damage_overlay)
    cv2.imwrite(os.path.join(result_path, 'part_heatmap.jpg'), part_overlay)
    # Convert images to Base64
    original_image_base64 = convert_img_to_base64(original_path)
    annotated_damage_base64 = convert_img_to_base64(os.path.join(result_path, 'annotated_damage.jpg'))
    damage_heatmap_base64 = convert_img_to_base64(os.path.join(result_path, 'damage_heatmap.jpg'))
    part_heatmap_base64 = convert_img_to_base64(os.path.join(result_path, 'part_heatmap.jpg'))
    
    
    # Predict parts to replace
    predicted_parts, part_probabilities, all_probabilities = predict_parts_to_replace(input_vector)

    top_5_indices = np.argsort(all_probabilities)[-5:][::-1]
    top_5_parts = [{'part': all_parts[idx], 'probability': float(all_probabilities[idx])} for idx in top_5_indices]
    
    # Create result dictionary
    result = {
        'id': assessment_id,
        'date': datetime.now().isoformat(),
        'original_image': original_image_base64,
        'annotated_damage': annotated_damage_base64,
        'damage_heatmap': damage_heatmap_base64,
        'part_heatmap':  part_heatmap_base64,
        'predicted_parts': [{'part': part, 'probability': prob} for part, prob in zip(predicted_parts, part_probabilities)],
        'top_5_parts': top_5_parts
    }
    
    # Add to history
    assessment_history.append(result)
    os.remove((os.path.join(result_path, 'annotated_damage.jpg')))  # Remove the image file after converting to Base64
    os.remove((os.path.join(result_path, 'damage_heatmap.jpg')))  # Remove the image file after converting to Base64
    os.remove((os.path.join(result_path, 'part_heatmap.jpg')))  # Remove the image file after converting to Base64
    os.remove((original_path))  # Remove the image file after converting to Base64
    os.rmdir(result_path)  # Remove the result directory
    return jsonify(result)

@app.route('/history')
def get_history():
    return jsonify([{
        'id': item['id'],
        'date': item['date'],
        'thumbnail': item['original_image']
    } for item in assessment_history])

@app.route('/history/<string:assessment_id>')
def get_history_item(assessment_id):
    for item in assessment_history:
        if item['id'] == assessment_id:
            return jsonify(item)
    return jsonify({'error': 'Assessment not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)