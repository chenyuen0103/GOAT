import torch.optim as optim
from train_model import train, val, test
from dataset import get_loaders
from model import VAE, CVAE
import const

def train_goat_vae(mode, target, save_path, save=False, decode_images=False, encoded_images=None):
    """
    Train a VAE model in the GOAT framework and optionally decode generated intermediate images.

    Args:
        mode (str): Dataset mode ('mnist' or 'cifar').
        target (int): Target transformation (e.g., rotation angle).
        save_path (str): Path to save the trained model.
        save (bool): Whether to save the trained model.
        decode_images (bool): Whether to decode intermediate generated images.
        encoded_images (torch.Tensor): Encoded images to be decoded if decode_images=True.
    """
    # Load dataset
    if mode == "mnist":
        from torchvision import datasets, transforms
        dataset_class = datasets.MNIST
        transform = transforms.ToTensor()
        vae = VAE(x_dim=28*28, z_dim=const.DIM).to(const.DEVICE)
        path = f"models/mnist/vae/vae_{target}_{const.DIM}.pt"
    elif mode == "cifar":
        from torchvision import datasets, transforms
        dataset_class = datasets.CIFAR10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        vae = CVAE(x_dim=32*32, z_dim=const.DIM, res=True).to(const.DEVICE)
        path = f"models/cifar/vae/res_vae_og_{const.DIM}.pt"
    else:
        raise ValueError("Unsupported dataset mode")
    
    # Load dataset and split
    raw_trainset = dataset_class(root=const.PATH_TO_DATA, train=True, download=True, transform=transform)
    raw_testset = dataset_class(root=const.PATH_TO_DATA, train=False, download=True, transform=transform)
    trainloader, valloader, testloader = get_loaders(raw_trainset, raw_testset)
    
    # Optimizer
    optimizer = optim.Adam(vae.parameters(), weight_decay=1e-5, lr=1e-3)
    
    # Train the VAE
    for epoch in range(1, 301):
        train(epoch, trainloader, vae, optimizer, vae=True)
        if epoch % 5 == 0:
            val(valloader, vae, vae=True)
    
    # Final evaluation
    test(testloader, vae, vae=True)
    
    # Save the model if required
    if save:
        torch.save(vae.state_dict(), save_path)
        print(f"Model saved at {save_path}")
    
    # Decode images if requested
    if decode_images and encoded_images is not None:
        with torch.no_grad():
            decoded_images = vae.decoder(encoded_images.to(const.DEVICE)).cpu()
        print("Decoded images generated.")
        return vae, decoded_images
    
    return vae

