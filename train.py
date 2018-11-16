# Imports here
import os, sys
import argparse

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# define command line inputs
parser = argparse.ArgumentParser(description= 'Train.py')
parser.add_argument('--data_dir', type= str, help='Enter data folder name (somefolder)' )
parser.add_argument('--save_dir', type= str, help = 'Enter checkpoint name (something.pth)')
parser.add_argument('--arch', dest= 'arch', type= str, default= 'vgg16', choices= ['densenet', 'alexnet', 'vgg'], help= 'choose model architecture')
parser.add_argument('--drop_p', dest= 'drop_p', type=float, default= 0.5, help= 'drop percentage (0.5)')
parser.add_argument('--gpu', dest= 'gpu', action= 'store_true', help= 'enable GPU')
parser.add_argument('--epochs', dest= 'epochs', default= 3, type= int)
parser.add_argument('--learning_rate', dest= 'learning_rate', type= float, default= 0.001, help= 'model learning rate')
parser.add_argument('--hidden_units', dest= 'hidden_units', type= int, nargs= '+')

args = parser.parse_args()

def model_builder(arch):
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

def print_model(arch, hidden_units, model):
    print('Model Architecture: ', arch.title())
    print('Hidden Layers: ', hidden_units)
    print('Model: \n', model)


def validation(model, validloader, criterion):
    test_loss = 0
    accuracy = 0
    model.eval()
    model.to(device)

    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        probs = torch.exp(output) # classifier output is LogSoftmax
        check = (labels.data == probs.max(dim= 1)[1])
        accuracy += check.type(torch.FloatTensor).mean()

    return test_loss, accuracy


def train_model(model, trainloader, validloader, criterion, optimizer, device, epochs=3):
# define hyperparameters

    steps = 0
    running_loss = 0
    print_every = 40

    for e in range(epochs):
        model.classifier.train()
        model.to(device)

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            steps += 1
            optimizer.zero_grad()

            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion)

                print('Epoch: {}/{}...'.format(e+1, epochs),
                      'Training Loss: {:.3f} '.format(running_loss / print_every),
                      'Validation Loss: {:.3f} '.format(test_loss / len(validloader)),
                      'Validation Accuracy: {:.2f}%'.format(accuracy / len(validloader) * 100))

                running_loss = 0
                model.classifier.train()


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
    model = model_builder(args.arch)
    features = list(model.classifier.children())[:-1]

    if 'alexnet' in args.arch:
        input_size = features[1].in_features
    elif 'vgg' in args.arch:
        input_size = features[0].in_features
    else:
        input_size = model.classifier.in_features

    output_size = 102
    drop_p = args.drop_p
    learning_rate = args.learning_rate

    if '--hidden_layers':
        hidden_layers = list(args.hidden_units)
    else:
        hidden_layers = [4096, 2048, 512]

    classifier = create_classifier(input_size, output_size, hidden_layers, drop_p)
    model.classifier = classifier
    print_model(args.arch, hidden_layers, model)

    cw_dir = os.path.abspath(os.path.curdir)

    if args.data_dir == None:
        data_folder = os.path.join(cw_dir, 'flowers')
    else:
        data_folder = os.path.join(cw_dir, args.data_dir)


    train_dir = os.path.join(data_folder, 'train')
    valid_dir = os.path.join(data_folder, 'valid')
    test_dir = os.path.join(data_folder, 'test')

    # define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # define datasets
    train_data = datasets.ImageFolder(train_dir, transform= train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform= test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform= test_transforms)

    # define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle= True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle= True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle= True)

    # set device to gpu or cpu
    if args.gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    # define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr= learning_rate) # optimizer set for classifier

    train_model(model, trainloader, validloader, criterion, optimizer, device, args.epochs)

    if args.save_dir:
        chkpt_dir = os.path.join(cw_dir, args.save_dir)
    else:
        chkpt_dir = os.path.join(cw_dir, 'model_chkpt.pth')

    model.class_to_idx = train_data.class_to_idx
    model_classifier = {'epoch': args.epochs,
            'model_arch': args.arch,
            'learning_rate': learning_rate,
            'input_size': input_size,
            'output_size': output_size,
            'hidden_layers': hidden_layers,
            'drop_p': args.drop_p,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'class_to_idx': model.class_to_idx}
    torch.save(model_classifier, chkpt_dir)

    print('Training complete.')
    print('Model checkpoint saved location: ', chkpt_dir)
