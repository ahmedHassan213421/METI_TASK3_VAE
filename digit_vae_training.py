import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


# Define the Variational Autoencoder architecture
class DigitVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(DigitVAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        # Calculate the size after flattening
        self.flatten_size = 64 * 7 * 7

        # Latent space
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

        self.latent_dim = latent_dim

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        z = self.decoder_input(z)
        x = self.decoder(z)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# Loss function: Reconstruction loss + KL divergence
def vae_loss_function(recon_x, x, mu, logvar):
    # Binary cross entropy for reconstruction loss
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")

    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


# Training function
def train(model, train_loader, optimizer, device, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        loss = vae_loss_function(recon_batch, data, mu, logvar)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item() / len(data):.6f}"
            )

    print(
        f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}"
    )


# Function to generate digits
def generate_digit(model, digit, device, num_samples=5):
    # Load a subset of the test set containing only the specified digit
    test_dataset = datasets.MNIST(
        "./data", train=False, download=True, transform=transforms.ToTensor()
    )

    # Filter dataset to contain only the specified digit
    digit_indices = [i for i, (_, label) in enumerate(test_dataset) if label == digit]

    # Select a random sample of the digit
    idx = np.random.choice(digit_indices)
    sample, _ = test_dataset[idx]
    sample = sample.unsqueeze(0).to(device)

    # Encode the sample to get mean and variance
    with torch.no_grad():
        mu, logvar = model.encode(sample)

    # Generate samples by sampling from the latent space
    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            # Sample from the latent space with some noise
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) * 0.5
            # Decode the latent vector
            sample = model.decode(z)
            samples.append(sample.cpu().squeeze().numpy())

    return samples


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer
    latent_dim = 32
    model = DigitVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    num_epochs = 20
    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, optimizer, device, epoch)

    # Save the model
    torch.save(model.state_dict(), "digit_vae_model.pth")
    print("Model saved successfully!")

    # Generate and display some sample digits
    for digit in range(10):
        samples = generate_digit(model, digit, device)

        plt.figure(figsize=(12, 3))
        for i, sample in enumerate(samples):
            plt.subplot(1, 5, i + 1)
            plt.imshow(sample, cmap="gray")
            plt.title(f"Generated {digit}")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"generated_digit_{digit}.png")


if __name__ == "__main__":
    main()
