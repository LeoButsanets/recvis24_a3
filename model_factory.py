"""Python file to instantite the model and the transform that goes with it."""

from data import data_transforms, data_transforms_resnet
from model import Net
import torchvision.models as models
import torch.nn as nn

nclasses = 500

class ModelFactory:
    def __init__(self, model_name: str, only_last_layers: bool):
        self.model_name = model_name
        self.only_last_layers = only_last_layers
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        
        if self.model_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
             
            if self.only_last_layers:
                # Freeze all layers except the last one
                for param in model.parameters():
                    param.requires_grad = False
            
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, nclasses)
            if self.only_last_layers:
                # Ensure the new fully connected layer is trainable
                for param in model.fc.parameters():
                    param.requires_grad = True
            return model
        
        if self.model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            
            if self.only_last_layers:
                # Freeze all layers except the last one
                for param in model.parameters():
                    param.requires_grad = False
            
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, nclasses)
            
            if self.only_last_layers:
                # Ensure the new fully connected layer is trainable
                for param in model.fc.parameters():
                    param.requires_grad = True

            return model
        
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return data_transforms
        if self.model_name == "resnet18":
            return data_transforms_resnet
        if self.model_name == "resnet50":
            return data_transforms_resnet
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
