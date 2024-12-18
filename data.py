from torchvision.transforms import functional as F
from PIL import ImageOps
import torch
import random
import torchvision.transforms as transforms

# Define enhanced data augmentation with more diverse transformations
data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),  # Increase probability for horizontal flip
    transforms.RandomRotation(degrees=5, fill=(255, 255, 255)),  # Increase rotation angle to add more variety
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Apply slight changes in brightness, contrast, etc.
    transforms.RandomAffine(degrees=1, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=(255, 255, 255)),  # Add random affine transformations
])


data_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)




data_transforms_resnet = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

data_transforms_resnet_augmented = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        data_augmentation
    ]
)

### Sketch data augmentation

class InvertColorsIfNeeded:
    def __call__(self, img):
        # Convert the image to grayscale to calculate mean intensity
        grayscale_img = img.convert("L")
        img_np = np.array(grayscale_img)

        # Calculate the mean intensity
        mean_intensity = img_np.mean()
        # If the image is mostly dark, invert the colors
        if mean_intensity > 128:
            return ImageOps.invert(img)
        return img
    
class AdaptiveBinarize:
    def __call__(self, img):
        # Calculate the mean pixel value of the grayscale image
        mean_intensity = img.mean()
        
        # Adjust threshold relative to mean intensity to avoid an all-black image
        threshold = min(max(mean_intensity, 0.1), 0.9)  # Threshold between 0.1 and 0.9 to avoid extreme values
        
        # Apply thresholding
        return (img > threshold).float()


# Replace lambda for repeating channels with a proper class
class RepeatChannels:
    def __init__(self, num_channels):
        self.num_channels = num_channels

    def __call__(self, img):
        return img.repeat(self.num_channels, 1, 1)

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

class GaussianBlur:
    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        # Convert the PIL image to a torch tensor if it's not already
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
        
        # Create Gaussian kernel
        channels, height, width = img.shape
        kernel = self._create_gaussian_kernel(self.kernel_size, self.sigma)

        # Apply the Gaussian filter to each channel separately
        img_filtered = []
        for c in range(channels):
            img_c = img[c].unsqueeze(0).unsqueeze(0)  # Add batch and channel dimension
            img_c_blurred = F.conv2d(img_c, kernel, padding=self.kernel_size // 2)
            img_filtered.append(img_c_blurred.squeeze())

        return torch.stack(img_filtered)

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float):
        """Creates a 2D Gaussian kernel to be used for convolution."""
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        return kernel


# Define data augmentation


# Define data augmentation with less variation
data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomRotation(degrees=5)
])


data_transforms_sketch = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        InvertColorsIfNeeded(),  # Invert colors to make background black and sketch white
        transforms.ToTensor(),
        AdaptiveBinarize(),  # Binarization with a specified threshold
        GaussianBlur(kernel_size=5, sigma=0.5),  # Apply Gaussian blur to smooth out the image
        RepeatChannels(num_channels=3),  # Duplicate grayscale channel to 3 channels
        data_augmentation,  # Apply data augmentation
        transforms.Normalize(mean=[0.485], std=[0.229]),  # Adjust mean and std for grayscale
   
    ]
)   

data_transforms_sketch_augmented = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        InvertColorsIfNeeded(),  # Invert colors to make background black and sketch white
        transforms.ToTensor(),
        AdaptiveBinarize(),  # Binarization with a specified threshold
        GaussianBlur(kernel_size=5, sigma=0.5),  # Apply Gaussian blur to smooth out the image
        RepeatChannels(num_channels=3),  # Duplicate grayscale channel to 3 channels
        transforms.Normalize(mean=[0.485], std=[0.229]),  # Adjust mean and std for grayscale
   
    ]
)   