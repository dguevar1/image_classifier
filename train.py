import argparse

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

def parse_arguments():
    # Command line test cases:
    # 1.) python train.py flowers
    # 2.) python train.py flowers --save_dir checkpoints
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

def main():
    args = parse_arguments()
    print_argument_info(args)

if __name__ == "__main__":
    main()
