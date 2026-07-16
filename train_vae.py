import argparse
import os
import torch
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
import const
import torch.optim as optim
from train_model import train, test, val
from dataset import *
from model import *
from util import *
try:
    from plot import og_dec_plot
except Exception:
    og_dec_plot = None


def train_vae(vae, trainloader, valloader, testloader, path, save=False):
    raw_data = []
    for i in range(10):
        raw_data.append(raw_trainset[i][0])
    raw_data = torch.stack(raw_data)

    lr = 1e-3
    optimizer = optim.Adam(vae.parameters(), weight_decay=1e-5, lr=lr)
    for epoch in range(1, 301):
        # if epoch % 75 == 0:
        #     lr *= 0.1
        #     optimizer = optim.Adam(vae.parameters(), weight_decay=1e-5, lr=lr)
        train(epoch, trainloader, vae, optimizer, vae=True)
        if epoch % 5 == 0:
            val(valloader, vae, vae=True)
        if epoch % 20 == 0 and og_dec_plot is not None:
            with torch.no_grad():
                data, _ = vae.encoder(raw_data.to(const.DEVICE))
                # data = vae.sampling(mu, log_var)
                c = int(raw_data.shape[1])
                h = int(raw_data.shape[2])
                w = int(raw_data.shape[3])
                data = vae.decoder(data.to(const.DEVICE)).reshape(-1, c, h, w).cpu().detach()
            og_dec_plot(raw_data, data, f"og_dec/ce_vae1024_og_dec_{epoch}.png", False)

    test(testloader, vae, vae=True)

    if save:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(vae.state_dict(), path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE/CVAE with RMNIST pipeline.")
    parser.add_argument("--target", type=int, default=30, help="Target rotation angle for MNIST checkpoint naming.")
    parser.add_argument("--dim", type=int, default=None, help="Override latent dimension (default: const.DIM).")
    parser.add_argument("--save", action="store_true", help="Save trained checkpoint.")
    return parser.parse_args()


def main():
    global raw_trainset, raw_testset
    args = parse_args()
    if args.dim is not None:
        const.DIM = int(args.dim)
    target = int(args.target)
    angles = get_angles(target // 5, target)[1:]
    print(angles)

    if const.MODE == "mnist":
        # Train/evaluate on a union of source (0°) and target-rotated domains.
        src_train = get_single_rotate(True, 0)
        tgt_train = get_single_rotate(True, target)
        src_test = get_single_rotate(False, 0)
        tgt_test = get_single_rotate(False, target)
        raw_trainset = ConcatDataset([src_train, tgt_train])
        raw_testset = ConcatDataset([src_test, tgt_test])
        print(
            f"[VAE] MNIST train domains: source(0°) + target({target}°), "
            f"train={len(raw_trainset)}, test={len(raw_testset)}"
        )
        vae = VAE(x_dim=28 * 28, z_dim=const.DIM).to(const.DEVICE)
        ckpt_path = f"models/mnist/vae/vae_{target}_{const.DIM}.pt"
        if os.path.exists(ckpt_path):
            vae.load_state_dict(torch.load(ckpt_path, map_location=const.DEVICE))
            print(f"[VAE] Loaded checkpoint: {ckpt_path}")
        else:
            print(f"[VAE] No checkpoint found at {ckpt_path}; training from scratch.")
    elif const.MODE == "cifar":
        raw_trainset = datasets.CIFAR10(
            root=const.PATH_TO_MNIST, train=True, download=True, transform=transforms.ToTensor()
        )
        raw_testset = datasets.CIFAR10(
            root=const.PATH_TO_MNIST, train=False, download=True, transform=transforms.ToTensor()
        )
        vae = CVAE(x_dim=32 * 32, z_dim=const.DIM, res=True).to(const.DEVICE)
        ckpt_path = f"models/cifar/vae/res_vae_og_{const.DIM}.pt"
    else:
        raise ValueError(f"Unsupported const.MODE={const.MODE}")

    trainloader, valloader, testloader = get_loaders(
        raw_trainset, raw_testset, batch_size=const.TRAIN_BS
    )
    train_vae(vae, trainloader, valloader, testloader, ckpt_path, save=args.save)


if __name__ == "__main__":
    main()
