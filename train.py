# Import packages
import argparse
from collections import OrderedDict
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def get_classifier(model_parameters):
    classifier = nn.Sequential(OrderedDict([
        ('dropout', nn.Dropout(model_parameters['dropout'])),
        ('fc1', nn.Linear(model_parameters['input_size'], model_parameters['hidden_units'])),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(model_parameters['hidden_units'], model_parameters['output_size'])),
        ('output', nn.LogSoftmax(dim=1))
        ]))

    return classifier

def get_model(arch, epochs, learning_rate):
    model_parameters = {"arch": arch,
                        "epochs": epochs,
                        "learning_rate": learning_rate,
                        "dropout": -1,
                        "input_size": -1,
                        "hidden_units": -1,
                        "output_size": -1
                        }

    if arch == 'densenet121':
        model_parameters["dropout"] = 0.2
        model_parameters["input_size"] = 1024
        model_parameters["hidden_units"] = 256
        model_parameters["output_size"] = 102
        model = models.densenet121(pretrained=True)
    elif arch == 'vgg16':
        model_parameters["dropout"] = 0.5
        model_parameters["input_size"] = 25088
        model_parameters["hidden_units"] = 512
        model_parameters["output_size"] = 102
        model = models.vgg16(pretrained=True)

    return model_parameters, model

def load_data(data_dir):
    my_data_dir = {"train" : None,
                   "valid" : None,
                   "test" : None
                   }

    my_data_dir["train"] = data_dir + '/train'
    my_data_dir["valid"] = data_dir + '/valid'
    my_data_dir["test"] = data_dir + '/test'

    my_transforms = {"train" : None,
                     "valid" : None,
                     "test" : None
                     }

    # TODO: Define your transforms for the training, validation, and testing sets
    my_transforms["train"] = transforms.Compose([transforms.RandomRotation(30),
                                                 transforms.RandomResizedCrop(224),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])])

    my_transforms["valid"] = transforms.Compose([transforms.Resize(255),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])])

    my_transforms["test"] = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    my_datasets = {"train" : None,
                   "valid" : None,
                   "test" : None
                   }

    my_datasets["train"] = datasets.ImageFolder(my_data_dir["train"], transform=my_transforms["train"])
    my_datasets["valid"] = datasets.ImageFolder(my_data_dir["valid"], transform=my_transforms["valid"])
    my_datasets["test"]= datasets.ImageFolder(my_data_dir["test"], transform=my_transforms["test"])

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    my_dataloader = {"train" : None,
                     "valid" : None,
                     "test" : None
                     }

    my_dataloader["train"] = torch.utils.data.DataLoader(my_datasets["train"], batch_size=64, shuffle=True)
    my_dataloader["valid"] = torch.utils.data.DataLoader(my_datasets["valid"], batch_size=64)
    my_dataloader["test"] = torch.utils.data.DataLoader(my_datasets["test"], batch_size=64)

    return my_datasets, my_dataloader

def parse_arguments():
    # Command line test cases:
    # 1.) python train.py flowers
    # 2.) python train.py flowers --save_dir checkpoints --epochs 5 --learning_rate 0.001 --gpu
    # 3.) python train.py flowers --arch "vgg16"
    # 4.) python train.py flowers --learning_rate 0.01 --hidden_units 512 --epochs 20
    # 5.) python train.py flowers --gpu
    parser = argparse.ArgumentParser(description="Prints out training loss, validation loss, and validation accuracy as the network trains.")
    parser.add_argument("data_dir")
    parser.add_argument("--save_dir", default="./", help="The directory where the model checkpoint should be saved.")
    parser.add_argument("--arch", default="vgg16", choices=["vgg16","densenet121"], help="Specify the architecure, or pretrain model, to use.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="The learning rate to use during training.")
    parser.add_argument("--hidden_units", type=int, default=512, help="The number of units in the hidden layer.")
    parser.add_argument("--epochs", type=int, default=10, help="The number of epochs to train the model.")
    parser.add_argument("--gpu", action="store_true", help="Use the GPU for training.")

    return parser.parse_args()

def print_argument_info(args):
    print("Using the following data directory: {}".format(args.data_dir))

    print("Saving the checkpoint to the following directory: {}".format(args.save_dir))

    print("The architecture is {}.".format(args.arch))

    print("The hyperparameters are the following:")

    print("learning_rate: {}".format(args.learning_rate))

    print("hidden_units: {}".format(args.hidden_units))

    print("epochs: {}".format(args.epochs))

    if args.gpu:
        print("GPU will be used to train.")
    else:
        print("CPU will be used to train.")

def print_model_parameters(model_parameters):
    print("model_parameters['arch']: {}".format(model_parameters['arch']))
    print("model_parameters['epochs']: {}".format(model_parameters['epochs']))
    print("model_parameters['learning_rate']: {}".format(model_parameters['learning_rate']))
    print("model_parameters['dropout']: {}".format(model_parameters['dropout']))
    print("model_parameters['input_size']: {}".format(model_parameters['input_size']))
    print("model_parameters['hidden_units']: {}".format(model_parameters['hidden_units']))
    print("model_parameters['output_size']: {}".format(model_parameters['output_size']))

def train(model_parameters, model, my_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training with device: {device}")

    print_model_parameters(model_parameters)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = get_classifier(model_parameters)

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=model_parameters['learning_rate'])

    # Moving model to device for training.
    model.to(device);

    # Set the model mode to training.
    model.train()

    steps = 0
    running_loss = 0
    print_every = 10

    for epoch in range(model_parameters['epochs']):
        for inputs, labels in my_dataloader["train"]:
            steps += 1

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in my_dataloader["valid"]:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{model_parameters['epochs']}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(my_dataloader['valid']):.3f}.. "
                      f"Test accuracy: {accuracy/len(my_dataloader['valid']):.3f}")
                running_loss = 0
                model.train()

def validation(model, dataloader, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    model.train()
    avg_loss = test_loss/len(dataloader)
    avg_accuracy = accuracy/len(dataloader)

    return avg_loss, avg_accuracy

def save_model(model_parameters, model, save_dir):
    checkpoint = {'arch': model_parameters['arch'],
                  'epochs': model_parameters['epochs'],
                  'learning_rate': model_parameters['learning_rate'],
                  'dropout': model_parameters['dropout'],
                  'input_size': model_parameters['input_size'],
                  'hidden_units': model_parameters['hidden_units'],
                  'output_size': model_parameters['output_size'],
                  'class_to_idx': model.class_to_idx,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  }

    checkpoint_name = save_dir + "/" + \
                      model_parameters['arch'] + "_" + \
                      str(model_parameters['epochs']) + "_" + \
                      str(model_parameters['learning_rate']) + "_" + \
                      str(model_parameters['hidden_units']) + "_" + \
                      "checkpoint.pth"

    torch.save(checkpoint,
               checkpoint_name
               )

def main():
    args = parse_arguments()
    print_argument_info(args)
    model_parameters, model = get_model(args.arch, args.epochs, args.learning_rate)
    my_datasets, my_dataloader = load_data(args.data_dir)
    start_time = time.time()
    train(model_parameters, model, my_dataloader)
    end_time = time.time()
    my_seconds = end_time - start_time
    my_time_delta = str(datetime.timedelta(seconds=my_seconds))
    print("Training time: {}".format(my_time_delta))
    criterion = nn.NLLLoss()
    avg_log, avg_accuracy = validation(model, my_dataloader["test"], criterion)
    print("Accuracy of the network on the test images: {:.2f}%".format(100*avg_accuracy))
    model.class_to_idx = my_datasets["train"].class_to_idx
    save_model(model_parameters, model, args.save_dir)

if __name__ == "__main__":
    main()
