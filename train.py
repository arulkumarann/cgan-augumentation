import torch
import torch.optim as optim
from models.generator import Generator
from models.discriminator import Discriminator
from data.prepare_data import load_cifar10
from torchvision.utils import save_image
import os

# Initialize the models
z_dim = 100  # Latent space dimension
num_classes = 10  # For CIFAR-10
generator = Generator(z_dim, num_classes)
discriminator = Discriminator(num_classes)

# Optimizers and Loss
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = torch.nn.BCELoss()

# Load Data
train_loader, labels = load_cifar10()

# Create a folder to save generated images
os.makedirs('./outputs/generated_images', exist_ok=True)

def save_generated_images(epoch, fixed_z, labels_one_hot):
    fake_images = generator(fixed_z, labels_one_hot)
    save_image(fake_images, f'./outputs/generated_images/epoch_{epoch}.png', nrow=8, normalize=True)

def train_gan():
    fixed_z = torch.randn(64, z_dim)  # Fixed noise for visualizing progress
    fixed_labels = torch.eye(num_classes).repeat(8, 1)  # Repeat labels to match batch size of 64

    for epoch in range(100):  # Train for 100 epochs
        for real_images, real_labels in train_loader:
            batch_size = real_images.size(0)
            labels_one_hot = torch.zeros(batch_size, num_classes).scatter_(1, real_labels.view(-1, 1), 1)
            z = torch.randn(batch_size, z_dim)

            # Train Discriminator
            optimizer_d.zero_grad()
            fake_images = generator(z, labels_one_hot)
            real_labels_tensor = torch.ones(batch_size, 1)
            fake_labels_tensor = torch.zeros(batch_size, 1)

            real_loss = criterion(discriminator(real_images, labels_one_hot), real_labels_tensor)
            fake_loss = criterion(discriminator(fake_images.detach(), labels_one_hot), fake_labels_tensor)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            g_loss = criterion(discriminator(fake_images, labels_one_hot), real_labels_tensor)
            g_loss.backward()
            optimizer_g.step()

        print(f'Epoch [{epoch+1}/100], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

        # Save generated images every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_generated_images(epoch + 1, fixed_z, fixed_labels)

if __name__ == "__main__":
    train_gan()
