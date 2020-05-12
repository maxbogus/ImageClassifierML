# Imports here
import argparse
import matplotlib.pyplot as plt

import torch
import json
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, SELECTED_MODEL
from PIL import Image
import numpy as np
from torch.autograd import Variable

# parse args
parser = argparse.ArgumentParser(description='Parse arguments.')
parser.add_argument("--image", type=str, default=1)
parser.add_argument("--checkpoint", type=str, defalut=0.003)
parser.add_argument("--jsonFile", type=str, default=0.01)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--use-gpu", type=bool, default=False)

args = parser.parse_args()

image = None
checkpoint = None
jsonFile = None
USE_GPU = args.useGpu

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    SELECTED_MODEL.densenet121(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 102),
                                 nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    preprocess = transforms.Compose([transforms.Resize(255),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])
                       ])
    image = preprocess(image)

    return image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    img = Image.open(image_path)
    img = process_image(img)
    
    # Convert 2D image to 1D vector
    img = np.expand_dims(img, 0)
    
    
    img = torch.from_numpy(img)
    
    model.eval()
    inputs = Variable(img).to(device)
    logits = model.forward(inputs)
    
    ps = F.softmax(logits,dim=1)
    topk = ps.cpu().topk(topk)
    
    return (e.data.numpy().squeeze().tolist() for e in topk)

probs, classes = predict('flowers/valid/102/image_08006.jpg', model)
class_names = train_data.classes
labels = [cat_to_name[class_names[e]] for e in classes]
y_pos = np.arange(len(probs))

print(labels, y_pos, class_names, probs, classes)

