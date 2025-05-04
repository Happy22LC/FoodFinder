# model.py
# Placeholder for your ML model. Replace with actual model loading and prediction.

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import os
import requests
from pathlib import Path

class FoodRecognitionModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load pre-trained ResNet model
        try:
            self.model = models.resnet50(pretrained=True)
            num_classes = 101  # Food101 dataset classes
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
            
            # Load model weights if they exist, otherwise use pretrained
            weights_path = Path(__file__).parent / 'food_model_weights.pth'
            if weights_path.exists():
                print("Loading saved model weights...")
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            else:
                print("Using pretrained weights...")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
        # Load food classes and descriptions
        try:
            self.classes, self.descriptions = self.load_food_data()
        except Exception as e:
            print(f"Error loading food data: {str(e)}")
            raise
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_food_data(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, 'food_data.json')
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                food_data = json.load(f)
            
            classes = list(food_data.keys())
            return classes, food_data
        except Exception as e:
            print(f"Error reading food data: {str(e)}")
            raise

    def predict(self, image):
        try:
            # Preprocess image
            image_tensor = self.transform(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top prediction
            prob, class_idx = torch.max(probabilities, 0)
            predicted_class = self.classes[class_idx]
            
            # Get food description and cuisine info
            food_info = self.descriptions[predicted_class]
            
            return {
                'food': predicted_class,
                'confidence': float(prob),
                'description': food_info['description'],
                'cuisine': food_info['cuisine'],
                'cuisine_description': food_info['cuisine_description']
            }
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {'error': str(e)}

# Global model instance
model = None

def load_model():
    global model
    try:
        if model is None:
            print("Initializing model...")
            model = FoodRecognitionModel()
            print("Model initialized successfully!")
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        raise

def predict_image(img):
    global model
    try:
        # Ensure model is loaded
        if model is None:
            load_model()
        
        # Make prediction
        result = model.predict(img)
        return result
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return {
            'error': 'Failed to process image',
            'details': str(e)
        } 