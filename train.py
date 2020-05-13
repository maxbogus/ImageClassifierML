#!/usr/bin/env python
"""
Module for model setup and training
"""
import argparse
import sys

import torch
from torch import nn
from torch import optim

from torchvision import datasets, transforms, models

# parse args
PARSER = argparse.ArgumentParser(description='Parse arguments.')
PARSER.add_argument("--model", type=int, default=1)
PARSER.add_argument("--checkpoint", type=str, default='checkpoint.pth')
PARSER.add_argument("--rate", type=float, default=0.003)
PARSER.add_argument("--units", type=int, default=256)
PARSER.add_argument("--epochs", type=int, default=1)
PARSER.add_argument("--gpu", type=bool, default=False)

ARGS = PARSER.parse_args()

SELECTED_MODEL = ARGS.model
LEARNING_RATE = ARGS.rate
HIDDEN_UNITS = ARGS.units
EPOCHS = ARGS.epochs
USE_GPU = ARGS.gpu
CHECKPOINT_FILE_NAME = ARGS.checkpoint

DATA_DIR = 'flowers'
TRAIN_DIR = DATA_DIR + '/train'
VALID_DIR = DATA_DIR + '/valid'
TEST_DIR = DATA_DIR + '/test'

if USE_GPU:
    if torch.cuda.is_available() is not False:
        print('GPU is not supported')
        sys.exit()

TRAIN_TRANSFORMS = transforms.Compose(
    [transforms.RandomRotation(30),
     transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

VALID_TRANSFORMS = transforms.Compose(
    [transforms.Resize(255),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

TEST_TRANSFORMS = transforms.Compose(
    [transforms.Resize(255),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

TRAIN_DATA = datasets.ImageFolder(TRAIN_DIR, transform=TRAIN_TRANSFORMS)
VALID_DATA = datasets.ImageFolder(VALID_DIR, transform=VALID_TRANSFORMS)
TEST_DATA = datasets.ImageFolder(TEST_DIR, transform=TEST_TRANSFORMS)

TRAIN_LOADER = torch.utils.data.DataLoader(TRAIN_DATA, batch_size=64, shuffle=True)
VALID_LOADER = torch.utils.data.DataLoader(VALID_DATA, batch_size=64)
TEST_LOADER = torch.utils.data.DataLoader(TEST_DATA, batch_size=64)
FEATURES = 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if SELECTED_MODEL == 1:
    PRETRAINED_MODEL = models.densenet121(pretrained=True)
else:
    FEATURES = 2208
    PRETRAINED_MODEL = models.densenet161(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in PRETRAINED_MODEL.parameters():
    param.requires_grad = False

PRETRAINED_MODEL.classifier = nn.Sequential(
    nn.Linear(FEATURES, HIDDEN_UNITS),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(HIDDEN_UNITS, 102),
    nn.LogSoftmax(dim=1)
    )

CRITERION = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
OPTIMIZER = optim.Adam(PRETRAINED_MODEL.classifier.parameters(), lr=LEARNING_RATE)

PRETRAINED_MODEL.to(DEVICE)

STEPS = 0
RUNNING_LOSS = 0
PRINT_EVERY = 5
for epoch in range(EPOCHS):
    for inputs, labels in TRAIN_LOADER:
        STEPS += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        OPTIMIZER.zero_grad()

        logps = PRETRAINED_MODEL.forward(inputs)
        loss = CRITERION(logps, labels)
        loss.backward()
        OPTIMIZER.step()

        RUNNING_LOSS += loss.item()

        if STEPS % PRINT_EVERY == 0:
            test_loss = 0
            accuracy = 0
            PRETRAINED_MODEL.eval()
            with torch.no_grad():
                for inputs, labels in TEST_LOADER:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    logps = PRETRAINED_MODEL.forward(inputs)
                    batch_loss = CRITERION(logps, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {EPOCHS+1}/{EPOCHS}.. "
                  f"Train loss: {RUNNING_LOSS/PRINT_EVERY:.3f}.. "
                  f"Test loss: {test_loss/len(TEST_LOADER):.3f}.. "
                  f"Test accuracy: {accuracy/len(TEST_LOADER):.3f}")
            RUNNING_LOSS = 0
            PRETRAINED_MODEL.train()

CHECKPOINT = {
    'model_state_dict': PRETRAINED_MODEL.state_dict(),
    'optimizer_state_dict': OPTIMIZER.state_dict()
    }

torch.save(CHECKPOINT, CHECKPOINT_FILE_NAME)
