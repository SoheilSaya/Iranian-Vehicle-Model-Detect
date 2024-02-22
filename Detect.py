import os
import timm
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models 
from PIL import Image
import numpy as np
import pandas as pd




normal_mapping={'Mazda-2000': 0, 'Nissan-Zamiad': 1, 'Peugeot-206': 2, 'Peugeot-207i': 3, 'Peugeot-405': 4, 'Peugeot-Pars': 5, 'Peykan': 6, 'Pride-111': 7, 'Pride-131': 8, 'Quik': 9, 'Renault-L90': 10, 'Samand': 11, 'Tiba2': 12}
reverse_mapping={0: 'Mazda-2000', 1: 'Nissan-Zamiad', 2: 'Peugeot-206', 3: 'Peugeot-207i', 4: 'Peugeot-405', 5: 'Peugeot-Pars', 6: 'Peykan', 7: 'Pride-111', 8: 'Pride-131', 9: 'Quik', 10: 'Renault-L90', 11: 'Samand', 12: 'Tiba2'}
path_label=[]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class MyModel(nn.Module):

    def __init__(self, model_name='skresnet34', pretrained=True):
        super(MyModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained, in_chans=3)
        self.fc1 = nn.Linear(1000,16)
        self.fc2 = nn.Linear(16,64)        
        self.fc3 = nn.Linear(64,len(class_names))
        
    def forward(self, x):
        #print(x.shape)
        x = self.model(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #print(x.shape)
        return x
    
device=torch.device("cpu")
loaded_model = torch.load('accepted.pth')
# Define transformations

loaded_model.eval()

# Move model to the same device as input tensor
loaded_model.to(device)


# Get the current directory
dir_path = os.getcwd()
# List all the files in the directory
files = os.listdir(dir_path)
# Filter the files with .png or .jpg extension
image_files = [file for file in files if file.endswith(('.png', '.jpg'))]
# Print the list of image files
print(image_files)


for item in image_files:
    image_path = item
    image = Image.open(image_path).convert('RGB')

    # Apply transformations
    input_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move input tensor to the same device as the model
    input_image = input_image.to(device)

    # Set your model to evaluation mode


    # Make prediction
    with torch.no_grad():
        output = loaded_model(input_image)

    # Get predicted class index
    predicted_index = torch.argmax(output).item()

    # Map index to class name
    predicted_class = reverse_mapping[predicted_index]

    print(item, ':',predicted_class)