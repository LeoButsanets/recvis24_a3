from data import data_transforms, data_transforms_resnet, data_transforms_sketch
from model import Net
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageClassification
import torchvision.transforms as transforms
from torchsummary import summary
import timm  # Library for pretrained models from Hugging Face

nclasses = 500

class ModelFactory:
    def __init__(self, model_name: str, train_full_model: bool = False, freeze_layers: int = 0, checkpoint_path: str = None, use_cuda: bool = False, augment: bool = False):
        self.model_name = model_name
        self.use_cuda = use_cuda
        self.freeze_layers = freeze_layers
        self.augment = augment
        self.train_full_model = train_full_model
        self.checkpoint_path = checkpoint_path
        self.model, self.optimizer_state, self.start_epoch = self.init_model()
        self.transform = self.init_transform()

        # Print the summary of the model using torchsummary
        self.print_summary()

    def init_model(self):
        model = None
        optimizer_state = None
        start_epoch = 0

        if self.checkpoint_path is not None and os.path.exists(self.checkpoint_path):
            print(f"Loading existing model from {self.checkpoint_path}")
            model = self._create_model_instance()
            checkpoint = torch.load(self.checkpoint_path) if self.use_cuda else torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state = checkpoint.get('optimizer_state_dict', None)
            start_epoch = checkpoint.get('epoch', 0)
        else:
            print(f"No existing model found. Initializing a new {self.model_name} model.")
            model = self._create_model_instance()

        return model, optimizer_state, start_epoch

    def _create_model_instance(self):
        if self.model_name == "basic_cnn":
            return Net()

        if self.model_name == "resnet18":
            model = timm.create_model('resnet18', pretrained=True)  # Load model from timm
            # Add dropout to the convolutional layers to prevent overfitting
            model.layer4[1].conv2 = nn.Sequential(
                model.layer4[1].conv2,
                nn.Dropout(0.5)
            )

            # Freeze all layers except the last specified layers
            if self.freeze_layers > 0 and not self.train_full_model:
                self._set_k_conv_trainable_layers(self.freeze_layers, model)

            model.fc = nn.Linear(model.fc.in_features, nclasses)

            # Print the ratio of trainable parameters
            self.print_trainable_ratio(model)

            return model

        if self.model_name == "resnet50":
            model = timm.create_model('resnet50', pretrained=True)  # Load model from timm

            # Adding dropout layers in convolutional blocks
            model.layer4[2].conv3 = nn.Sequential(
                model.layer4[2].conv3,
                nn.Dropout(0.5)
            )
            model.fc = nn.Sequential(
                nn.Dropout(0.5),  # Add dropout before fully connected layer to prevent overfitting
                nn.Linear(model.fc.in_features, nclasses)
            )

            # Freeze all layers except the last specified layers
            if self.freeze_layers > 0 and not self.train_full_model:
                self._set_k_conv_trainable_layers(self.freeze_layers, model)

            # Print the ratio of trainable parameters
            self.print_trainable_ratio(model)

            return model

        # Load a model from Hugging Face Model Hub
        if self.model_name.startswith("huggingface/"):
            print(f"Loading model from Hugging Face: {self.model_name}")
            model = AutoModelForImageClassification.from_pretrained(self.model_name)
            model.classifier = nn.Linear(model.classifier.in_features, nclasses)

            # Freeze all layers except the last specified layers
            if self.freeze_layers > 0 and not self.train_full_model:
                self._set_k_conv_trainable_layers(self.freeze_layers,  model)

            # Print the ratio of trainable parameters
            self.print_trainable_ratio(model)

            return model

        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return data_transforms
        if self.model_name == "resnet18":
            return data_transforms_resnet
        if self.model_name == "resnet50":
            print("Using data_transforms_resnet")
            return data_transforms_resnet
        if self.model_name.startswith("huggingface/"):
            # Use the same transforms as resnet for simplicity
            return data_transforms_resnet
        else:
            raise NotImplementedError("Transform not implemented")

    def _set_k_conv_trainable_layers(self,k:int, model):
        print(f"Freezing all layers except the last {k} convolutional layers")
        if k==0:
            return
        list_param = list(model.named_parameters())
        for name, param in list_param :
            param.requires_grad = False
        c=0
        for name, param in reversed(list_param):
            if "conv" in name:
                print(name)
                c+=1
            param.requires_grad = True
            if c==k:
                break

    def print_trainable_ratio(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ratio = trainable_params / total_params
        print(f"Trainable parameters ratio: {ratio:.4f}")

    def print_summary(self):
        # Print the summary of the model using torchsummary
        if self.use_cuda:
            summary(self.model, input_size=(3, 224, 224), device="cuda")
        else:
            summary(self.model, input_size=(3, 224, 224), device="cpu")

    def get_model(self):
        return self.model

    def get_optimizer_state(self):
        return self.optimizer_state

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform, self.optimizer_state, self.start_epoch
