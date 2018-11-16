# Imports here
import json
import os, sys
import argparse

from PIL import Image
# import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser(description= 'predict.py help')
parser.add_argument('--input', type= str, help= 'path/to/image')
parser.add_argument('--checkpoint', type= str, help= 'path/to/checkpoint')
parser.add_argument('--top_k', type=int, default= 3, help= 'number of top classes')
parser.add_argument('--category_names', type= str, help= 'filename.json')

args = parser.parse_args()

def determine_model(arch):
    if 'vgg' in arch:
        model = models.vgg16(pretrained= True)
    elif 'alexnet' in arch:
        model = models.alexnet(pretrained= True)
    elif 'densenet' in arch:
        model = models.densenet161(pretrained= True)
    else:
        print('Model architecture unknown.  Using vgg16...')
        model = models.vgg16(pretrained= True)

    for param in model.parameters():
        param.requires_grad = False
    return model

def model_builder(filepath):
    c = torch.load(filepath)

    model = determine_model(c['model_arch'])

    classifier = create_classifier(c['input_size'], c['output_size'], c['hidden_layers'], c['drop_p'])

    model.classifier = classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr= c['learning_rate'])
    model.load_state_dict(c['state_dict'])
    optimizer.load_state_dict(c['optimizer'])
    model.class_to_idx = c['class_to_idx']

    for param in model.parameters():
        param.requires_grad = False

    return model


def process_image(path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    new_size = 256
    crop_size = 224
    img = Image.open(path)
    w, h = img.size
    long_side = max(w, h)
    short_side = min(w, h)
    scale = new_size / short_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = img.resize((new_w, new_h), Image.ANTIALIAS)
    img.size
    if short_side == h:
        x1 = int((long_side * scale - crop_size) / 2)
        x2 = x1 + crop_size
        y1 = (short_side * scale - crop_size) / 2
        y2 = y1 + crop_size
    else:
        y1 = int((long_side * scale - crop_size) / 2)
        y2 = x1 + crop_size
        x1 = (short_side * scale - crop_size) / 2
        x2 = y1 + crop_size

    box = (x1, y1, x2, y2)
    img = img.crop(box)

    image = TF.to_tensor(img)
    image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return image


def predict(image_path, model, top_k=3):

    model.eval()
    model.to('cpu')

    image = image_path.unsqueeze(0)

    output = model.forward(image)
    y_hats = torch.exp(output).data.topk(top_k)
    probabilities = y_hats[0].cpu().numpy()
    categories = y_hats[1].cpu().numpy()[0]

    cat_num = model.class_to_idx
    cat_num_r = {model.class_to_idx[x]: x for x in model.class_to_idx}
    class_list = []
    cat_list = []
    for cat in categories:
        cat_list.append(cat_num_r[cat])
    for item in cat_list:
        class_list.append(cat_to_name[item])

    return probabilities, cat_list, class_list


class create_classifier (nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p= 0.5):
        super().__init__()
        self.nn_hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.nn_hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        for linear in self.nn_hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)

        x = self.output(x)
        F.log_softmax(x, dim=1)
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':

    model = model_builder(args.checkpoint)
    print(model)

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    cw_dir = os.path.abspath(os.path.curdir)
    path = os.path.join(cw_dir, args.input)
    image = process_image(args.input)
    ps, cats, classes  = predict(image, model, args.top_k)

    print('Probabilities: ',ps)
    print('Numerical Categories: ', cats)
    print('Flower Classes: ', classes)
