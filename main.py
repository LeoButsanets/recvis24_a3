import argparse
import yaml
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter

from model_factory import ModelFactory
import random
import string

import wandb

patience = 3

def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
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
        help="folder where data is located. train_images/ and val_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="basic_cnn",
        metavar="MOD",
        help="Name of the model for model and transform instantiation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.001,
        metavar="MIN_LR",
        help="mnimum learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment",
        metavar="E",
        help="folder where experiment outputs are located.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        metavar="C",
        help="checkpoint model to resume training",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        metavar="P",
        help="Early stopping patience",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        metavar="NW",
        help="number of workers for data loading",
    )
    parser.add_argument(
        "--train_full_model",
        type=bool,
        default=False,
        help="Whether to train the full model or not"
    )
    parser.add_argument(
        "--k_layers",
        type=int,
        default=3,
        help="Number of layers to freeze"
    )
    parser.add_argument(
        "--data_augmentation",
        type=bool,
        default=True,
        help="Whether to use data augmentation or not"
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

    # Print to check if k_layers is set correctly
    print(f"Parsed k_layers: {args.k_layers}")

    return args

def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
    epoch: int,
    args: argparse.ArgumentParser,
    writer: SummaryWriter,
) -> None:
    """Default Training Loop.

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optimizer): Optimizer to use
        train_loader (torch.utils.data.DataLoader): Training data loader
        use_cuda (bool): Whether to use cuda or not
        epoch (int): Current epoch
        args (argparse.ArgumentParser): Arguments parsed from command line
    """
    model.train()
    correct = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # Log the loss to TensorBoard and Weights & Biases
        writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)
        wandb.log({"Training Loss": loss.item(), "epoch": epoch})

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )
    accuracy = 100.0 * correct / len(train_loader.dataset)
    print(
        "\nTrain set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct,
            len(train_loader.dataset),
            accuracy,
        )
    )
    # Log training accuracy to Weights & Biases
    wandb.log({"Training Accuracy": accuracy, "epoch": epoch})

    return accuracy

def validation(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
    epoch: int,
    writer: SummaryWriter,
) -> float:
    """Default Validation Loop.

    Args:
        model (nn.Module): Model to train
        val_loader (torch.utils.data.DataLoader): Validation data loader
        use_cuda (bool): Whether to use cuda or not

    Returns:
        float: Validation loss
    """
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)

    # Log validation loss and accuracy to TensorBoard and Weights & Biases
    writer.add_scalar('Validation Accuracy', accuracy, epoch)
    wandb.log({"Validation Loss": validation_loss, "Validation Accuracy": accuracy, "epoch": epoch})
    
    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            validation_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )
    return validation_loss, accuracy

# Save model checkpoint with training parameters
def save_checkpoint(model, optimizer, epoch, val_loss, args, filepath):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "momentum": args.momentum,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "model_name": args.model_name,
    }
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)



def main():
    """Default Main Function."""
    # options
    args = opts()

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set the seed (for reproducibility)
    torch.manual_seed(args.seed)
    args.experiment = os.path.join(args.experiment, args.model_name)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # Create logs directory if it doesn't exist
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    model_name_save = f"{args.model_name}/k_layers_{args.k_layers}_batch_size_{args.batch_size}_lr_{args.lr}_{random_str}" + ("_augmented" if args.data_augmentation else "")

    log_dir = os.path.join('logs', model_name_save)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=log_dir)
    checkpoint_path = None
    if args.checkpoint is not None:
        print(f"Loading model from {args.checkpoint}")
        checkpoint_path = args.checkpoint
    else:
        for file in os.listdir(args.experiment):
            if file.startswith("best_model"):
                checkpoint_path = os.path.join(args.experiment, file)
                break

    # Load model and transform
    model, data_transforms, optimizer_state, start_epoch = ModelFactory(args.model_name, args.train_full_model, args.k_layers, checkpoint_path=checkpoint_path, use_cuda=use_cuda, augment=args.data_augmentation).get_all()
    
    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    # Data initialization and loading
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + "/train_images", transform=data_transforms),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + "/val_images", transform=data_transforms),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Load the optimizer state if it exists
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # Learning rate scheduler with minimum learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    # Loop over the epochs
    best_val_loss = 1e8 
    no_improve_epochs = 0
    patience = args.patience
    wandb.init(project="AS3", name=model_name_save)
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        # training loop
        train_accuracy = train(model, optimizer, train_loader, use_cuda, epoch, args, writer)
        # validation loop
        val_loss, val_accuracy = validation(model, val_loader, use_cuda, epoch, writer)
        writer.add_scalar('Validation Loss', val_loss, epoch)
        # Log accuracies on the same plot
        wandb.log({
            "Training Accuracy": train_accuracy,
            "Validation Accuracy": val_accuracy,
            "epoch": epoch
        })
        
        # validation loss to save model checkpoint
        val_loss = val_accuracy  # Use accuracy for deciding on saving the model
        writer.add_scalar('Validation Loss', val_loss, epoch)
        wandb.log({"Validation Loss": val_loss, "epoch": epoch})


        if val_loss < best_val_loss:
            # save the best model for validation
            best_val_loss = val_loss
            best_model_file = os.path.join(args.experiment, f"best_model_{args.model_name}_epoch{model_name_save}_{epoch}_val_loss_{val_loss:.4f}.pth")
            save_checkpoint(model, optimizer, epoch, val_loss, args, best_model_file)
            no_improve_epochs = 0  # Reset the patience counter when an improvement is seen

        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print("Early stopping triggered")
            break

        # Save the model every epoch
        model_file = os.path.join(args.experiment, f"best_model_{args.model_name}_epoch{model_name_save}_{epoch}_val_loss_{val_loss:.4f}.pth")
        save_checkpoint(model, optimizer, epoch, val_loss, args, model_file)

        print(
            "Saved model to "
            + model_file
            + f". You can run `python evaluate.py --model_name {args.model_name} --model "
            + best_model_file
            + "` to generate the Kaggle formatted csv file\n"
        )
        # Step the learning rate scheduler
        scheduler.step()
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
        wandb.log({"Learning Rate": scheduler.get_last_lr()[0], "epoch": epoch})
    
    writer.close()
    wandb.finish()

if __name__ == "__main__":
    main()
