from torchvision.transforms import functional as F
from PIL import ImageOps

import torchvision.transforms as transforms

# InvertColors class (as previously defined)
class InvertColors:
    def __call__(self, img):
        return ImageOps.invert(img)

class AdaptiveBinarize:
    def __call__(self, img):
        # Calculate the mean pixel value of the grayscale image
        mean_intensity = img.mean()
        
        # Set a threshold relative to the mean intensity (e.g., mean + some constant offset)
        threshold = mean_intensity + 0.1  # You can adjust the constant to fine-tune
        
        # Apply thresholding
        return (img > threshold).float()

# Replace lambda for repeating channels with a proper class
class RepeatChannels:
    def __init__(self, num_channels):
        self.num_channels = num_channels

    def __call__(self, img):
        return img.repeat(self.num_channels, 1, 1)

# Define the updated transformation
data_transforms_sketch = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        InvertColors(),  # Invert colors to make background black and sketch white
        transforms.ToTensor(),
        AdaptiveBinarize(),  # Binarization with a specified threshold
        RepeatChannels(num_channels=3),  # Duplicate grayscale channel to 3 channels
        transforms.Normalize(mean=[0.485], std=[0.229]),  # Adjust mean and std for grayscale
    ]
)