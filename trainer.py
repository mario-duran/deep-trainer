# PROGRAMMER: Mario A. Duran
# DATE CREATED: 09/02/2018
# SAMPLE USAGE: trainer.py "./path_to_data" --save_dir "./path_to_save/checkpoint.pth" --arch vgg16 --epochs 5 --learning_rate 0.00075 --hidden_units 512  --gpu --batch_size 32

import time
import datetime
import argparse
import os
import numpy
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

#PROCESS PARAMETERS
def init():
    #PARAMETERS
    arguments = get_args()

    #SET THE DATA PATHS
    data_dir = arguments.data_dir

    data_labels = ['train', 'valid', 'test']

    #SET TRANSFORMS
    data_transforms = {
        'train': transforms.Compose([
                    transforms.RandomRotation(30), 
                    transforms.RandomResizedCrop(224), 
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ]),
        'valid': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])
    }

    #LOAD DATASETS
    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x), 
            transform=data_transforms[x]
        )
        for x in data_labels
    }

    #DATA LOADERS
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=arguments.batch_size,
            shuffle=True
        )
        for x in data_labels
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in data_labels}

    for x in data_labels:
        print("Loaded {} images under {}".format(dataset_sizes[x], x))
    print("-"*10)

    #BUILD THE MODEL
    trainig_model = create_model(arguments.arch, arguments.hidden_units)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainig_model.classifier.parameters(), lr=arguments.learning_rate)

    #VALIDATE GPU USAGE
    gpu_avail = torch.cuda.is_available()
    use_GPU = False
    if gpu_avail and arguments.gpu:
        use_GPU = True
        print("GPU set to true")
    else:
        print("GPU not available")

    #TRAIN THE MODEL
    trained_model = train_model(trainig_model, arguments.epochs, use_GPU, dataloaders['train'], optimizer, criterion)

    #VALIDATE ON TEST DATA
    validate_test_data(trained_model, dataloaders['test'], use_GPU)

    #SAVE THE CHECKPOINT
    if arguments.save_dir:
        checkpoint = {'model': trained_model,
                      'architecture': arguments.arch,
                      'current_epochs': arguments.epochs,
                      'optim_state': optimizer.state_dict,
                      'train_mapping': image_datasets['train'].class_to_idx,
                      'gpu': use_GPU,
                      'state_dict': trained_model.state_dict()}
        save_model(arguments.save_dir, checkpoint)

    return None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Data directory")
    parser.add_argument("--save_dir", type=str, default="", help="Folder and filename to save the checkpoint")
    parser.add_argument("--arch", type=str, default="vgg16", help="The model architecture to use (vgg16, alexnet, densenet)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Sets the learning rate")
    parser.add_argument("--hidden_units", type=int, nargs="*", default=[1024], help="How many hidden layers to add to the classifier")
    parser.add_argument("--epochs", type=int, default=2, help="Sets the Epochs to be used (2 by default)")
    parser.add_argument("--gpu", action='store_true', help="Use GPU")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size to load the data")

    return parser.parse_args()

def get_model_arch(arch):
    #MODELS
    pre_model = None
    print("Loading {} architecture".format(arch))

    if arch == 'alexnet':
        pre_model = models.alexnet(pretrained=True)
        input_size = 9216
    elif arch == 'vgg16':
        pre_model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'densenet':
        pre_model = models.densenet161(pretrained=True)
        input_size = 2208
    
    return pre_model, input_size

def create_model(arch, hidden_units):
    #load pretrained model
    model, input_size = get_model_arch(arch)
    
    #freeze
    for param in model.features.parameters():
        param.require_grad = False

    #add hidden units / set output to 102
    features_list = []
    x = 1
    #BUILD NEW CLASSIFIER BASED ON HIDDEN UNITS
    for hu in hidden_units:
        features_list.extend([('fc' + str(x), nn.Linear(input_size, hu)), 
                            ('relu' + str(x), nn.ReLU()), 
                            ('drop' + str(x), nn.Dropout(p=0.5))])
        input_size = hu
        x+=1
    #ADD OUTPUT LAYER
    features_list.extend([('Output', nn.Linear(input_size, 102))])
    new_classifier = nn.Sequential(OrderedDict(features_list))

    #REPLACE CLASSIFIER
    model.classifier = new_classifier

    print("Classifier to use:")
    print(model.classifier)
    print("-"*10)
    return model

def train_model(model, epochs, use_gpu, train_loader, optimizer, criterion):

    started = time.time()
    print("Training started at {}".format(datetime.datetime.now()))
    
    model.train(True)
    
    if use_gpu:
        model.to('cuda')
    
    batch_data = train_loader.batch_size
    
    for e in range(epochs):
        train_loss = 0
        train_acc = 0
        batch_size = 0
        for i, (inputs, labels) in enumerate(train_loader):
            
            if use_gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            #FORWARD
            outputs = model.forward(inputs)
            _, pred = torch.max(outputs.data, 1)
            
            #BACKWARDS
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #LOG
            train_loss += loss.item()
            train_acc += (pred == labels).sum().item()
            batch_size += labels.size(0)

            if i > 0:
                if i % batch_data == 0:
                    print("Epoch: {}/{} - {}... ".format(e+1, epochs, i),
                          "Loss: {:.4f}".format(train_loss/batch_data),
                          ", Accuracy: {} - {} {:.2f}".format(train_acc, batch_size, (train_acc * 100) / batch_size))

                    train_loss = 0
                    train_acc = 0
                    batch_size = 0

    elapsed_time = time.time() - started
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    
    return model

def validate_test_data(model, test_data, use_gpu):    
    correct = 0
    total = 0
    started = time.time()
    with torch.no_grad():
        for data in test_data:
            inputs, labels = data
            
            if use_gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            outputs = model(inputs)
            _, pred = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    print("Accuracy on {} test Images: {:.2f}%".format(total, (100 * correct) / total))
    elapsed_time = time.time() - started
    print("Test completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))

def save_model(path, chekpoint):
    torch.save(chekpoint, path)

    print("Checkpoint saved at: " + path)
    return None

if __name__ == "__main__":
    init()