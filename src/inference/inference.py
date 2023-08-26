import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import os

# Hyperparameters
batch_size = 64
epochs = 100
latent_dim = 200
#Image settings:
main_image_dim = 100
main_seg_dim = 45

image_size = main_image_dim * main_image_dim * 3  # Image size with three channels (RGB)
segmentation_size = main_seg_dim * main_seg_dim # Segmentation mask size (binary)
center_size = main_seg_dim

data_path = ''

generator_model_path = ''
discriminator_model_path = ''

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_segmentation_mask(center_size, batch_size):
    mask = np.zeros((batch_size, 1, center_size, center_size), dtype=np.float32)
    mask[:, :, center_size // 2, center_size // 2] = 1.0
    return torch.tensor(mask, device=device)

# Create a custom generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim + segmentation_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
        )
        self.image_generation = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.LeakyReLU(0.2),
            nn.Linear(4096, image_size),
            nn.Tanh()
        )

    def forward(self, x, seg_mask):
        x = torch.cat((x, seg_mask), dim=1)  # Concatenate the latent vector and segmentation mask
        x = self.latent_projection(x)
        return self.image_generation(x)

# Create a custom discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(image_size + segmentation_size, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
        )
        self.discriminator_output = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, seg_mask):
        x = torch.cat((x, seg_mask), dim=1)  # Concatenate the image and segmentation mask
        x = self.input_projection(x)
        return self.discriminator_output(x)


# Load the saved models
generator = Generator()
discriminator = Discriminator()
generator.load_state_dict(torch.load(generator_model_path, map_location=torch.device(device)))
discriminator.load_state_dict(torch.load(discriminator_model_path, map_location=torch.device(device)) )

# Set the models to evaluation mode
generator.eval()
discriminator.eval()

# Create the output directory if it doesn't exist

# Generate and save images
def generate_coin(save_path, num_images_to_generate):
    with torch.no_grad():
        generator.to(device)
        seg_mask = create_segmentation_mask(center_size, num_images_to_generate)
        z = torch.randn(num_images_to_generate, latent_dim).to(device)
        fake_images = generator(z, seg_mask.view(num_images_to_generate, -1))
        fake_images = fake_images.view(fake_images.size(0), 3, 100, 100)
        save_image(fake_images, save_path, normalize=True)