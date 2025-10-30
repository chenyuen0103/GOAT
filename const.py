import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset selection
MODE = "mnist"  # Options: "mnist", "cifar"

# Paths to datasets
PATH_TO_MNIST = "./data/mnist"
PATH_TO_CIFAR = "./data/cifar"

# Training parameters
TRAIN_BS = 64  # Training batch size
TEST_BS = 1000  # Test batch size

# VAE-specific parameters
DIM = 20  # Latent dimension size
