import argparse 
import time
import torch 
import numpy as np
import json
import sys
from torch import nn, optim
from torchvision import datasets, models, transforms
from PIL import Image 
from collections import OrderedDict
        
def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                          ('dropout', nn.Dropout(p=0.2)),
                          ('fc1', nn.Linear(25088, checkpoint['hidden_layer'])),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(checkpoint['hidden_layer'], 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.classifier = classifier
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        return model


def parse():
    cl = argparse.ArgumentParser(description='predict.py')
    cl.add_argument('input_img', default='paind-project/flowers/test/1/image_06752.jpg')
    cl.add_argument('checkpoint', default='/home/workspace/paind-project/checkpoint.pth')
    cl.add_argument('--top_k', default=5)
    cl.add_argument('--category_names',default='cat_to_name.json')
    cl.add_argument('--gpu',action='store_true',default="cpu")
    args = cl.parse_args()
    return args
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    trans = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ])
    pil_image = trans(pil_image)
    return pil_image

def main():
    args = parse()
    image = process_image(args.input_img)
    model = load_checkpoint(args.checkpoint)
    if (args.gpu):
           image = image.cuda()
           model = model.cuda()
    model.eval()
    with torch.no_grad():
        image.unsqueeze_(0)
        image = image.float()
        outputs = model(image)
        probs, classes = torch.exp(outputs).topk(args.top_k)
        probs, classes = probs[0].tolist(), classes[0].add(1).tolist()
        prediction = zip(probs,classes)
    cat_file = jfile = json.loads(open(args.category_names).read())
    i = 0
    for p, c in prediction:
        i = i + 1
        p = str(round(p,4) * 100.) + '%'
        if (cat_file):
            c = cat_file.get(str(c),'None')
        else:
            c = 'class {}'.format(str(c))
        print("{}.{} ({})".format(i, c,p))

main()
