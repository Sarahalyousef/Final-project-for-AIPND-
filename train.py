import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
import json
from collections import OrderedDict
cl = argparse.ArgumentParser(description='Train.py')

def parse():
    cl = argparse.ArgumentParser(description='Train.py')
    cl.add_argument('data_directory')
    cl.add_argument('--save_dir', default="./checkpoint.pth")
    cl.add_argument('--arch',help='densenet121 or vgg16',  default="vgg16")
    cl.add_argument('--learning_rate', default=0.001,type=float)
    cl.add_argument('--hidden_units', default=6272,type=int)
    cl.add_argument('--epochs', default=2,type=int)
    cl.add_argument('--gpu',action='store_true',default="cpu")
    args = cl.parse_args()
    return args

def new_model():
    model = create_model()
    
 
def create_model():
    train_dir = args.data_directory + '/train'
    test_dir = args.data_directory + '/test'
    valid_dir = args.data_directory + '/valid'
    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    testing_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_data = datasets.ImageFolder(test_dir ,transform = testing_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size =64,shuffle = True)
    testing_loader = torch.utils.data.DataLoader(testing_data, batch_size = 64, shuffle = True)
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    model_c = 25088 
    if(args.arch == 'densenet121'):
        model = models.densenet121(pretrained=True)
        model_c = 1024
    else:
        model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  
    classifier = nn.Sequential(OrderedDict([
                          ('dropout', nn.Dropout(p=0.2)),
                          ('fc1', nn.Linear(model_c, args.hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(args.hidden_units,                                                  len(train_data.class_to_idx))),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    if (args.gpu and torch.cuda.is_available()):
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)
    train_losses, test_losses = [], []
    for e in range(args.epochs):
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss  += loss.item()
#4            
        else:
            test_loss = 0
            accuracy = 0   
            with torch.no_grad():
                for inputs,labels in validation_loader:
                    inputs, labels = inputs.to(device) , labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs,labels)
                    test_loss += loss.item()
                    ps = torch.exp(outputs)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                    
        model.train()
        train_losses.append(train_loss/len(train_loader))
        test_losses.append(test_loss/len(validation_loader))

        print("Epoch: {}/{}.. ".format(e+1, args.epochs),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),
              "validation Loss: {:.3f}.. ".format(test_losses[-1]),
              "validation Accuracy: {:.3f}".format(accuracy/len(validation_loader)))
        
        model.class_to_idx = train_data.class_to_idx
        torch.save({'pre-model': args.arch,
                    'state_dict':model.state_dict(),
                    'class_to_idx':model.class_to_idx,
                    'optimizer': optimizer.state_dict(),
                    'hidden_layer':args.hidden_units},
            args.save_dir)
    
    
    
      


def main():
    global args
    args = parse()
    new_model()
    
main()
