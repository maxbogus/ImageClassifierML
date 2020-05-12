#!/usr/bin/env python
"""
Module for model load and making predictions
"""
import json
import sys
import argparse

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np

# parse args
PARSER = argparse.ArgumentParser(description='Parse arguments.')
PARSER.add_argument("--image", type=str, default='flowers/valid/102/image_08006.jpg')
PARSER.add_argument("--checkpoint", type=str, default='checkpoint.pth')
PARSER.add_argument("--json", type=str, default='cat_to_name.json')
PARSER.add_argument("--epochs", type=int, default=1)
PARSER.add_argument("--gpu", type=bool, default=False)

ARGS = PARSER.parse_args()

IMAGE = ARGS.image
CHECKPOINT =ARGS.checkpoint
JSON_FILE = ARGS.json
USE_GPU = ARGS.gpu

if USE_GPU:
    if torch.cuda.is_available() is not False:
        print('GPU is not supported')
        sys.exit()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_TRANSFORMS = transforms.Compose(
    [transforms.RandomRotation(30),
     transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

DATA_DIR = 'flowers'
TRAIN_DIR = DATA_DIR + '/train'

TRAIN_DATA = datasets.ImageFolder(TRAIN_DIR, transform=TRAIN_TRANSFORMS)

with open(JSON_FILE, 'r') as f:
    CAT_TO_NAME = json.load(f)

def load_checkpoint(filepath):
    """
    Load saved checkpoint
    """
    checkpoint = torch.load(filepath)
    models.densenet121(pretrained=True)
    MODEL.classifier = nn.Sequential(
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 102),
        nn.LogSoftmax(dim=1))
    #criterion = nn.NLLLoss()
    optimizer = optim.Adam(MODEL.classifier.parameters(), lr=0.003)
    MODEL.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return MODEL

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    preprocess = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image = preprocess(image)

    return image

def imshow(image, ax=None):
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
    img = Image.open(image_path)
    img = process_image(img)

    # Convert 2D image to 1D vector
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img)

    model.eval()
    inputs = Variable(img).to(DEVICE)
    logits = model.forward(inputs)

    applied_softmax = F.softmax(logits, dim=1)
    topk = applied_softmax.cpu().topk(topk)

    return (e.data.numpy().squeeze().tolist() for e in topk)

MODEL = load_checkpoint(CHECKPOINT)

PROBS, PREDICTED_CLASSES = predict(IMAGE, MODEL)
CLASS_NAMES = TRAIN_DATA.classes
LABELS = [CAT_TO_NAME[CLASS_NAMES[e]] for e in PREDICTED_CLASSES]
Y_POS = np.arange(len(PROBS))

print(LABELS, Y_POS, CLASS_NAMES, PROBS, PREDICTED_CLASSES)
