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
#Complexity rating:
latent_dim = 200
#Image dimensions and settings:
main_image_dim = 100
main_seg_dim = 45

image_size = main_image_dim * main_image_dim * 3  # Image size with three channels (RGB)
segmentation_size = main_seg_dim * main_seg_dim # Segmentation mask size (binary)
center_size = main_seg_dim

#Path to the data (one class = mixed)
data_path = ''
save_image_path = ''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Manual seed (diversity)
torch.manual_seed(34535)

# Create a custom generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #Noise mapping:
        self.latent_projection = nn.Sequential(
            #Connected layers:
            nn.Linear(latent_dim + segmentation_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
        )
        #Generation after mapping:
        self.image_generation = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.LeakyReLU(0.2),
            nn.Linear(4096, image_size),
            nn.Tanh()
        )

    def forward(self, x, seg_mask):
        x = torch.cat((x, seg_mask), dim=1)  # Concatenate the latent vector and segmentation mask
        x = self.latent_projection(x)
        #Return generation:
        return self.image_generation(x)

# Create a custom discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.input_projection = nn.Sequential(
            #Connected layers (adding more requires more memory)
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

# Initialize the generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss function and optimizers
#BCELoss is used, but other loss functions may be used as well to avoid certain issues (wGAN Loss)
adversarial_loss = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Load coin dataset
#Transform the image:
transform = transforms.Compose([transforms.Resize((main_image_dim, main_image_dim)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#Load the one 'mixed' class:
dataset = ImageFolder(data_path, transform=transform)
#Load the class labels and the images from the dataset:
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create a square segmentation mask for the center of the image (Capture finer details at the center of each coin):
def create_segmentation_mask(center_size, batch_size):
    mask = np.zeros((batch_size, 1, center_size, center_size), dtype=np.float32)
    mask[:, :, center_size // 2, center_size // 2] = 1.0
    return torch.tensor(mask, device=device)

def save_generated_images(epoch):
    # Generate and save some sample images
    with torch.no_grad():
        seg_mask = create_segmentation_mask(center_size, 16)
        fake_images = generator(torch.randn(16, latent_dim).to(device), seg_mask.view(16, -1)).cpu()
        fake_images = fake_images.view(fake_images.size(0), 3, main_image_dim, main_image_dim)
        save_image(fake_images, os.path.join(save_image_path, f"gan_images_epoch_{epoch+1}.png"), normalize=True)


# Training loop
for epoch in range(epochs):
    #Train the disc and the gen, the disc becomes better at determining what is real and what is fake. Gen gets better at generating better images by fooling the discriminator. As the discriminator gets better, the generator gets better as well.
    for i, (images, _) in enumerate(dataloader):
        real_images = images.to(device)
        batch_size = real_images.size(0)
        #Add label smoothing (To avoid the discriminator being too confident):
        real_labels = torch.full((batch_size, 1), 0.95, device=device)
        fake_labels = torch.full((batch_size, 1), 0.05, device=device)

        # Prepare segmentation mask for conditional GAN
        seg_mask = create_segmentation_mask(center_size, batch_size)

        # Train the discriminator
        discriminator.zero_grad()
        real_outputs = discriminator(real_images.view(batch_size, -1), seg_mask.view(batch_size, -1))
        real_loss = adversarial_loss(real_outputs, real_labels)

        z = torch.randn(batch_size, latent_dim, device=device)
        fake_images = generator(z, seg_mask.view(batch_size, -1))
        fake_outputs = discriminator(fake_images.detach().view(batch_size, -1), seg_mask.view(batch_size, -1))
        fake_loss = adversarial_loss(fake_outputs, fake_labels)

        d_loss = (real_loss + fake_loss) / 2.0
        #Back prop
        d_loss.backward()
        optimizer_d.step()

        # Train the generator
        generator.zero_grad()
        fake_outputs = discriminator(fake_images.view(batch_size, -1), seg_mask.view(batch_size, -1))
        g_loss = adversarial_loss(fake_outputs, real_labels)
        #Back prop
        g_loss.backward()
        optimizer_g.step()
        #Record epoch if divisible by 10 (with respect to the batch size in [Hyperparameters])
        if i % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] Loss_D: {d_loss.item():.4f} Loss_G: {g_loss.item():.4f}")

    # Save generated images during training (optional)
    save_generated_images(epoch)

# Save the final generator model
torch.save(generator.state_dict(), "generator_model.pth")
torch.save(discriminator.state_dict(), "discriminator_model.pth")