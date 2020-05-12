# Imports here
import argparse

import torch
from sys import exit
from torch import nn
from torch import optim
from torch.autograd import Variable

from torchvision import datasets, transforms, models

# parse args
PARSER = argparse.ArgumentParser(description='Parse arguments.')
PARSER.add_argument("--model", type=int, default=1)
PARSER.add_argument("--learning-rate", type=float, defalut=0.003)
PARSER.add_argument("--hidden-units")
PARSER.add_argument("--epochs", type=int, default=1)
PARSER.add_argument("--use-gpu", type=bool, default=False)

ARGS = PARSER.parse_args()

SELECTED_MODEL = ARGS.model
LEARNING_RATE = ARGS.learningRate
HIDDEN_UNITS = ARGS.hiddenUnits
EPOCHS = ARGS.epochs
USE_GPU = ARGS.useGpu

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

if USE_GPU:
    if torch.cuda.is_available() is not False:
        print('GPU is not supported')
        exit()

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED_MODEL = models.densenet121(pretrained=True)
PRETRAINED_MODEL()

# Freeze parameters so we don't backprop through them
for param in PRETRAINED_MODEL.parameters():
    param.requires_grad = False
    
PRETRAINED_MODEL.classifier = nn.Sequential(nn.Linear(1024, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 102),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(PRETRAINED_MODEL.classifier.parameters(), lr=0.003)

PRETRAINED_MODEL.to(device)

# TODO: Do validation on the test set
EPOCHS = 1
steps = 0
running_loss = 0
print_every = 5
for epoch in range(EPOCHS):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = PRETRAINED_MODEL.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            PRETRAINED_MODEL.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = PRETRAINED_MODEL.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{EPOCHS}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            PRETRAINED_MODEL.train()

checkpoint = {'model_state_dict': PRETRAINED_MODEL.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')