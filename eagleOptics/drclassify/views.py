from django.shortcuts import render
from django.http import JsonResponse
import torch
from .model import load_model
from torchvision import transforms
from PIL import Image

# Load the model
model = load_model()

# Define class labels
class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']

def preprocess_image(image_file):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_file).convert('RGB')
    return transform(image).unsqueeze(0)

def predict_dr(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        image_tensor = preprocess_image(image_file)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]

        return render(request, 'drclassify/result.html', {'prediction': predicted_class})

    return render(request, 'drclassify/upload.html')

def home(request):
    return render(request, 'drclassify/upload.html')  # Redirect root to upload page
