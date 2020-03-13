# Import packages
import argparse
from collections import OrderedDict
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def load_model(path):
    checkpoint = torch.load(path)

    reconstructed_model_parameters = {"arch" : "",
                                      "epochs": -1,
                                      "learning_rate": -1.0,
                                      "dropout": -1,
                                      "input_size": -1,
                                      "hidden_layer1": -1,
                                      "output_size": -1
                                      }

    reconstructed_model = None

    reconstructed_model_parameters['arch'] = checkpoint['arch']
    reconstructed_model_parameters['epochs'] = checkpoint['epochs']
    reconstructed_model_parameters['learning_rate'] = checkpoint['learning_rate']
    reconstructed_model_parameters['dropout'] = checkpoint['dropout']
    reconstructed_model_parameters['input_size'] = checkpoint['input_size']
    reconstructed_model_parameters['hidden_units'] = checkpoint['hidden_units']
    reconstructed_model_parameters['output_size'] = checkpoint['output_size']


    if reconstructed_model_parameters['arch'] == 'densenet121':
        reconstruced_model = models.densenet121(pretrained=True)
    elif reconstructed_model_parameters['arch'] == 'vgg16':
        reconstructed_model = models.vgg16(pretrained=True)
    else :
        raise Exception("Invalid arch name: {}".format(reconstructed_model_parameters['arch']))

    reconstructed_model.classifier = checkpoint['classifier']
    reconstructed_model.class_to_idx = checkpoint['class_to_idx']
    reconstructed_model.load_state_dict(checkpoint['state_dict'])

    return reconstructed_model_parameters, reconstructed_model

def parse_arguments():
    # Command line test cases:
    # 1.) python predict.py flowers/train/1/image_06734.jpg vgg16_checkpoint.pth
    # 2.) python predict.py flowers/train/1/image_06734.jpg vgg16_checkpoint.pth --top_k 3
    # 3.) python predict.py flowers/train/1/image_06734.jpg --category_names cat_to_name.json
    # 4.) python predict.py flowers/train/1/image_06734.jpg vgg16_checkpoint.pth --gpu
    parser = argparse.ArgumentParser(description="Predict flower name from an image along with the probability of that name.")
    parser.add_argument("image_path")
    parser.add_argument("checkpoint_path")
    parser.add_argument("--top_k", type=int, default=5, choices=[1, 2, 3, 4, 5], help="The number of top classes to return.")
    parser.add_argument("--category_names", default="cat_to_name.json", help="The mapping of categories to real names.")
    parser.add_argument("--gpu", action="store_true", help="Use the GPU for training.")

    return parser.parse_args()

def predict(image_path, model, top_k=5, device="cuda"):
    ''' Predict the class (or classes) of an image using a trained deep learning model.

    image_path: string. Path to image, directly to image and not to folder.
    model: pytorch neural network.
    top_k: integer. The top K classes to be calculated

    returns top_probabilities(k), top_labels
    '''

    # TODO: Implement the code to predict the class from an image file
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load and preprocess the image.
    numpy_image = process_image(image_path)

    # Swith model to desire device.
    model.to(device)

    # Set model to evaluate
    model.eval();

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(numpy_image)
    torch_image = torch_image.unsqueeze(0)
    torch_image = torch_image.type(torch.FloatTensor)

    # Move input tensors to the GPU/CPU
    torch_image = torch_image.to(device)

    # Find probabilities (results) by passing through the function (note the log softmax means that its on a log scale)
    # We also do torch.no_grad() so it doesn't try to keep track of calculating gradients and executes faster.
    with torch.no_grad():
        log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    torch_top_probs, torch_top_labels = linear_probs.topk(top_k)

    # We want to return numpy arrays so it can be compatible with matplotlib.plt libraries.
    numpy_top_probs = torch_top_probs.cpu().numpy()[0]
    numpy_top_labels = torch_top_labels.cpu().numpy()[0]

    # Convert to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    numpy_top_labels = [idx_to_class[lab] for lab in numpy_top_labels]

    return numpy_top_probs, numpy_top_labels

def print_argument_info(args):
    print("Using the following image path: {}".format(args.image_path))

    print("Using the following checkpoint path: {}".format(args.checkpoint_path))

    print("Number of top classes returned is {}.".format(args.top_k))

    print("Using the following file for mapping category names: {}".format(args.category_names))

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

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''

    # Step 1: Use PIL.Image to load the image.
    pil_image = Image.open(image_path)

    # Step 2: Resize the shortest image side to 256 pixels while keeping the aspect ratio.

    # Get original dimensions
    original_width, original_height = pil_image.size

    aspect_ratio = original_width / original_height

    # Find shorter size and create settings to crop shortest side to 256
    if aspect_ratio > 1:
        pil_image = pil_image.resize((round(aspect_ratio * 256), 256))
    else:
        pil_image = pil_image.resize((256, round(256 / aspect_ratio)))

    # Step 3: Crop out the center 224x224 portion of the image
    resized_width, resized_height = pil_image.size
    new_width = 224
    new_height = 224
    left = (resized_width - new_width)/2
    top = (resized_height - new_height)/2
    right = (resized_width + new_width)/2
    bottom = (resized_height + new_height)/2
    pil_image = pil_image.crop((round(left), round(top), round(right), round(bottom)))

    # Step 4: Convert to numpy and normalize value in 0-1 range.
    # Note, we divided by 255 because imshow() expects floats in the range 0-1.
    np_image = np.array(pil_image)/255

    # Step 5: Normalize each color channel
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std

    # Set 6: Set the color to the first channel as expected by PyTorch. In the PIL image,
    # the color is the third dimension. We make color the first dimension and keep the order
    # of width and height.
    np_image = np_image.transpose(2, 0, 1)

    return np_image

def main():
    args = parse_arguments()
    print_argument_info(args)
    loaded_model_parameters, loaded_model = load_model(args.checkpoint_path)
    print_model_parameters(loaded_model_parameters)

    if args.gpu:
        device = "cuda"
    else:
        device = "cpu"

    probs, classes = predict(args.image_path, loaded_model, args.top_k, device)

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    names = [cat_to_name[i] for i in classes]

    for name, prob in zip(names, probs):
        print("{} has probability {:.4f}".format(name, prob))

if __name__ == "__main__":
    main()
