import argparse
import os
import yaml

import PIL.Image as Image
import torch
from tqdm import tqdm

from model_factory import ModelFactory


def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 evaluation script")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data_sketches",
        metavar="D",
        help="folder where data is located. test_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--model",
        type=str,
        metavar="M",
        help="the model file to be evaluated. Usually it is of the form model_X.pth",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="basic_cnn",
        metavar="MOD",
        help="Name of the model for model and transform instantiation.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="experiment/kaggle.csv",
        metavar="D",
        help="name of the output csv file",
    )
    parser.add_argument(
        "--k_layers",
        type=int,
        default=0,
        metavar="K",
        help="Number of layers to freeze in the model.",
    )
    parser.add_argument(
        "--train_full_model",
        type=bool,
        default=False,
        help="Whether to train the full model or not"
    )

    args = parser.parse_args()

    # Load parameters from config file if specified
    if args.config is not None:
        with open(args.config, "r") as file:
            config_params = yaml.safe_load(file)
        
        # Update args with values from the config file if they exist
        for key, value in config_params.items():
            if hasattr(args, key):
                setattr(args, key, value)

    return args


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def main() -> None:
    """Main Function."""
    # options
    args = opts()
    test_dir = args.data + "/test_images/mistery_category"

    # cuda
    use_cuda = torch.cuda.is_available()

    # load model and transform
    checkpoint = torch.load(args.model, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    # Load the model state dict from the checkpoint
    state_dict = checkpoint['model_state_dict']

    model, data_transforms, _, _ = ModelFactory(model_name=args.model_name, train_full_model=args.train_full_model, freeze_layers = args.k_layers, use_cuda=use_cuda, augment=args.data_augmentation).get_all()
    model.load_state_dict(state_dict)
    model.eval()
    if use_cuda:
        print("Using GPU")
        model.cuda()    
    else:
        print("Using CPU")

    output_file = open(args.outfile, "w")
    output_file.write("Id,Category\n")
    for f in tqdm(os.listdir(test_dir)):
        if "jpeg" in f:
            data = data_transforms(pil_loader(test_dir + "/" + f))
            data = data.view(1, data.size(0), data.size(1), data.size(2))
            if use_cuda:
                data = data.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            output_file.write("%s,%d\n" % (f[:-5], pred))

    output_file.close()

    print(
        "Succesfully wrote "
        + args.outfile
        + ", you can upload this file to the kaggle competition website"
    )


if __name__ == "__main__":
    main()
