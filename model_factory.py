"""Python file to instantiate the model and the transform that goes with it."""

from data import data_transforms, data_transforms_resnet, data_transforms_sketch
from model import Net
import torchvision.models as models
import torch.nn as nn
import os
import torch
import torch.nn.functional as F

from torchvision.transforms import functional as F
import torchvision.transforms as transforms

nclasses = 500

class ModelFactory:
    def __init__(self, model_name: str, only_last_layers: bool, experiment_path: str):
        self.model_name = model_name
        self.only_last_layers = only_last_layers
        self.experiment_path = experiment_path
        self.model, self.start_epoch = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        model = None
        checkpoint_path = os.path.join(self.experiment_path, f"{self.model_name}_best.pth")
        start_epoch = 0

        if os.path.exists(checkpoint_path):
            print(f"Loading existing model from {checkpoint_path}")
            model = self._create_model_instance()
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
        else:
            print(f"No existing model found. Initializing a new {self.model_name} model.")
            model = self._create_model_instance()

        return model, start_epoch

    def _create_model_instance(self):
        if self.model_name == "basic_cnn":
            return Net()
        
        if self.model_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model.layer4[1].conv2.name = "last_conv"  # Assign a name to the last convolutional layer
            model.fc = nn.Sequential(
                nn.Dropout(0.5),  # Add dropout to prevent overfitting
                nn.Linear(model.fc.in_features, nclasses)
            )
            
            if self.only_last_layers:
                # Freeze all layers except the last three
                for name, param in model.named_parameters():
                    if not any(layer in name for layer in ["layer4.1", "layer4.0", "fc"]):
                        param.requires_grad = False
            return model
        
        if self.model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            model.layer4[2].conv3.name = "last_conv"  # Assign a name to the last convolutional layer
            model.fc = nn.Sequential(
                nn.Dropout(0.5),  # Add dropout to prevent overfitting
                nn.Linear(model.fc.in_features, nclasses)
            )

            if self.only_last_layers:
                # Freeze all layers except the last three
                for name, param in model.named_parameters():
                    if not any(layer in name for layer in ["layer4.2", "layer4.1", "fc"]):
                        param.requires_grad = False
            return model
        
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return data_transforms
        if self.model_name == "resnet18":
            return data_transforms_sketch
        if self.model_name == "resnet50":
            return data_transforms_sketch
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform, self.start_epoch
