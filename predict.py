import argparse

def print_argument_info(args):
    print("Using the following image path: {}".format(args.image_path))

    print("Using the following checkpoint path: {}".format(args.checkpoint_path))

    print("Number of top classes returned is {}.".format(args.top_k))

    print("Using the following file for mapping category names: {}".format(args.category_names))

    if args.gpu:
        print("GPU will be used to train.")
    else:
        print("CPU will be used to train.")

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

def main():
    args = parse_arguments()
    print_argument_info(args)

if __name__ == "__main__":
    main()
