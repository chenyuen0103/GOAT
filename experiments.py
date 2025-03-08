import torch
from model import *
import torch.optim as optim
import torchvision
from train_model import *
from util import *
from ot_util import ot_ablation
from da_algo import *
from ot_util import generate_domains
from expansion_util import *
from dataset import *
import copy
import argparse
import random
import torch.backends.cudnn as cudnn
import time
import os 
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pickle
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_progress(log_file, step, step_type, domain_idx, dataset, acc1=None, acc2=None, acc3=None, target_acc=None):
    """
    Logs training progress to a CSV file.

    Args:
        log_file (str): Path to the CSV log file.
        step (int): The sequential step number.
        step_type (str): Type of domain (Real_Intermediate, Synthetic_Intermediate, Final_Adaptation).
        domain_idx (int): The index of the domain (0,1,2,...).
        dataset (str): Dataset type (Ground-Truth, Synthetic, Target, etc.).
        acc1, acc2, acc3 (float, optional): Accuracy values.
        target_acc (float, optional): Accuracy on the target domain.
    """
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "Type", "Domain_Index", "Dataset", "Direct_Acc", "ST_Acc", "Generated_Acc", "Target_Acc", "Timestamp"])

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            step, step_type, domain_idx, dataset, 
            round(acc1, 4) if acc1 is not None else "", 
            round(acc2, 4) if acc2 is not None else "", 
            round(acc3, 4) if acc3 is not None else "", 
            round(target_acc, 4) if target_acc is not None else "", 
            time.time()
        ])


def init_tensorboard(log_dir='logs/tensorboard'):
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def get_source_model(args, trainset, testset, n_class, mode, encoder=None, epochs=50, verbose=True, model_path="cache/source_model.pth"):

    model = Classifier(encoder, MLP(mode=mode, n_class=n_class, hidden=1024)).to(device)
    if os.path.exists(model_path):
        print(f"✅ Loading cached trained model from {model_path}")
        model = Classifier(encoder, MLP(mode=mode, n_class=n_class, hidden=1024)).to(device)
        model.load_state_dict(torch.load(model_path))
        return model


    print("Start training source model")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    for epoch in range(1, epochs+1):
        train(epoch, trainloader, model, optimizer, verbose=verbose)
        if epoch % 5 == 0:
            test(testloader, model, verbose=verbose)

    print("🔄 Training new source model...")
    model = get_source_model_old(args, trainset, testset, n_class, mode, encoder, epochs=epochs)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)  # ✅ Save model for future runs
    
    return model

def get_source_model_old(args, trainset, testset, n_class, mode, encoder=None, epochs=50, verbose=True):
    print("Start training source model")
    model = Classifier(encoder, MLP(mode=mode, n_class=n_class, hidden=1024)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    for epoch in range(1, epochs+1):
        train(epoch, trainloader, model, optimizer, verbose=verbose)
        if epoch % 5 == 0:
            test(testloader, model, verbose=verbose)
    
    return model

def run_goat(model_copy, source_model, src_trainset, tgt_trainset, all_sets, generated_domains, epochs=10):
    step_counter = 0
    # get the performance of direct adaptation from the source to target, st involves self-training on target
    direct_acc, st_acc = self_train(args, model_copy, [tgt_trainset], epochs=epochs)
    # get the performance of GST from the source to target, st involves self-training on target
    direct_acc_all, st_acc_all = self_train(args, source_model, all_sets, epochs=epochs)
    cache_dir = "cache/"
    # encode the source and target domains
    e_src_trainset, e_tgt_trainset = get_encoded_dataset(source_model.encoder, src_trainset, cache_path=os.path.join(cache_dir, "encoded_source.pt")), get_encoded_dataset(source_model.encoder, tgt_trainset, cache_path=os.path.join(cache_dir, "encoded_target.pt"))

    # encode the intermediate ground-truth domains
    intersets = all_sets[:-1]
    encoded_intersets = [e_src_trainset]
    for i, interset in enumerate(intersets):
        cache_path = os.path.join(cache_dir, f"encoded_inter_{i}.pt")
        encoded_intersets.append(get_encoded_dataset(source_model.encoder, interset), cache_path=cache_path)
    encoded_intersets.append(e_tgt_trainset)

    # generate intermediate domains
    generated_acc = 0
    if generated_domains > 0:
        all_domains = []
        for i in range(len(encoded_intersets)-1):
            all_domains += generate_domains(generated_domains, encoded_intersets[i], encoded_intersets[i+1])

        _, generated_acc = self_train(args, source_model.mlp, all_domains, epochs=epochs)
    
    return direct_acc, st_acc, direct_acc_all, st_acc_all, generated_acc



def run_main_algo(model_copy, source_model, src_trainset, tgt_trainset, all_sets, generated_domains, epochs=10, log_file="main_algo_log.csv"):
    """
    Runs the main algorithm following the same structure as run_goat.
    """
    domain_indices = []
    domain_types = []
    # First, process source and real intermediate datasets
    datasets = [src_trainset] + all_sets  # Include source and real domains
    domain_indices.extend(range(len(datasets)))
    domain_types.extend(["source"] + ["real"] * (len(all_sets) - 1) + ["target"])  
    cache_dir = "cache/"
    os.makedirs(cache_dir, exist_ok=True)

    # Get the performance of direct adaptation from the source to target
    # direct_acc, st_acc = self_train(args, model_copy, [tgt_trainset], [len(all_sets)-1], ['target'], epochs=epochs, log_file="direct_"+log_file)

    # Get the performance of self-training across all ground-truth domains
    # direct_acc_all, st_acc_all = self_train(args, source_model, all_sets, domain_indices, domain_types, epochs=epochs, log_file="pool_"+log_file)

    # encode the source and target domains
    e_src_trainset, e_tgt_trainset = get_encoded_dataset(source_model.encoder, src_trainset, cache_path=os.path.join(cache_dir, "encoded_source.pt")), get_encoded_dataset(source_model.encoder, tgt_trainset, cache_path=os.path.join(cache_dir, "encoded_target.pt"))
    # breakpoint()
    # Encode the intermediate ground-truth domains
    intersets = all_sets[:-1]
    encoded_intersets = [e_src_trainset]
    # breakpoint()
    for i, interset in enumerate(intersets):
        cache_path = os.path.join(cache_dir, f"encoded_inter_{i}.pt")
        encoded_intersets.append(get_encoded_dataset(source_model.encoder, interset, cache_path=cache_path))
    encoded_intersets.append(e_tgt_trainset)

    # Generate intermediate domains using Wasserstein interpolation
    generated_acc = 0
    if generated_domains > 0:
        all_domains = []
        synthetic_indices = []
        print(f"Generating {generated_domains} intermediate domains using Wasserstein interpolation...")
        base_idx = len(domain_indices) - 1
        # ✅ Generate synthetic domains for adaptation
        for i in range(len(encoded_intersets) - 1):
            inter_domains = generate_gauss_domains(
                source_dataset=encoded_intersets[i],
                target_dataset=encoded_intersets[i + 1],
                n_wsteps=generated_domains,
                device=device
            )
            all_domains.extend(inter_domains)
            synthetic_indices.extend([base_idx + i + 1] * generated_domains)

        # Train on synthetic domains and log
        domain_indices.extend(synthetic_indices)
        domain_types.extend(["synthetic"] * len(synthetic_indices))
        _, generated_acc = self_train(args, source_model.mlp, all_domains,domain_indices, domain_types, epochs=epochs, log_file="gradual_"+log_file)

    return 0, 0, 0, 0, generated_acc


def run_mnist_experiment(target, gt_domains, generated_domains):
    t = time.time()

    src_trainset, tgt_trainset = get_single_rotate(False, 0), get_single_rotate(False, target)
    # breakpoint()
    encoder = ENCODER().to(device)
    source_model = get_source_model(args, src_trainset, src_trainset, 10, "mnist", encoder=encoder, epochs=5, model_path = f"cache/mnist_{target}_source_model.pth")
    model_copy = copy.deepcopy(source_model)
    all_sets = []
    for i in range(1, gt_domains+1):
        all_sets.append(get_single_rotate(False, i*target//(gt_domains+1)))
        print(i*target//(gt_domains+1))
    all_sets.append(tgt_trainset)
    _, _, _, _, main_algo_acc = run_main_algo(model_copy, source_model,
        src_trainset, tgt_trainset, all_sets, generated_domains, epochs=5)

    direct_acc, st_acc, direct_acc_all, st_acc_all, generated_acc = run_goat(model_copy, source_model, src_trainset, tgt_trainset, all_sets, generated_domains, epochs=5)
    # ✅ Run Main Algorithm (Wasserstein-based adaptation)

    # ✅ Run our main algorithm (Wasserstein-based adaptation)
    # your code here
    elapsed = round(time.time() - t, 2)
    print(elapsed)
    with open(f"logs/mnist_{target}_{gt_domains}_layer.txt", "a") as f:
        f.write(f"seed{args.seed}with{gt_domains}gt{generated_domains}generated,{round(direct_acc, 2)},{round(st_acc, 2)},{round(direct_acc_all, 2)},{round(st_acc_all, 2)},{round(generated_acc, 2)}, Main Algorithm: {round(main_algo_acc, 2)}\n")


def run_mnist_ablation(target, gt_domains, generated_domains):

    encoder = ENCODER().to(device)
    src_trainset, tgt_trainset = get_single_rotate(False, 0), get_single_rotate(False, target)
    source_model = get_source_model(args, src_trainset, src_trainset, 10, "mnist", encoder=encoder, epochs=20)
    model_copy = copy.deepcopy(source_model)

    all_sets = []
    for i in range(1, gt_domains+1):
        all_sets.append(get_single_rotate(False, i*target//(gt_domains+1)))
        print(i*target//(gt_domains+1))
    all_sets.append(tgt_trainset)

    direct_acc, st_acc = self_train(args, model_copy, [tgt_trainset], epochs=10)
    direct_acc_all, st_acc_all = self_train(args, source_model, all_sets, epochs=10)
    model_copy1 = copy.deepcopy(source_model)
    model_copy2 = copy.deepcopy(source_model)
    model_copy3 = copy.deepcopy(source_model)
    model_copy4 = copy.deepcopy(source_model)

    e_src_trainset, e_tgt_trainset = get_encoded_dataset(source_model.encoder, src_trainset), get_encoded_dataset(source_model.encoder, tgt_trainset)
    intersets = all_sets[:-1]
    encoded_intersets = [e_src_trainset]
    for i in intersets:
        encoded_intersets.append(get_encoded_dataset(source_model.encoder, i))
    encoded_intersets.append(e_tgt_trainset)

    # random plan
    all_domains1 = []
    for i in range(len(encoded_intersets)-1):
        plan = ot_ablation(len(src_trainset), "random")
        all_domains1 += generate_domains(generated_domains, encoded_intersets[i], encoded_intersets[i+1], plan=plan)[0]
    _, generated_acc1 = self_train(args, model_copy1.mlp, all_domains1, epochs=10)
    
    # uniform plan
    all_domains4 = []
    for i in range(len(encoded_intersets)-1):
        plan = ot_ablation(len(src_trainset), "uniform")
        all_domains4 += generate_domains(generated_domains, encoded_intersets[i], encoded_intersets[i+1], plan=plan)[0]
    _, generated_acc4 = self_train(args, model_copy4.mlp, all_domains4, epochs=10)
    
    # OT plan
    all_domains2 = []
    for i in range(len(encoded_intersets)-1):
        all_domains2 += generate_domains(generated_domains, encoded_intersets[i], encoded_intersets[i+1])[0]
    _, generated_acc2 = self_train(args, model_copy2.mlp, all_domains2, epochs=10)

    # ground-truth plan
    all_domains3 = []
    for i in range(len(encoded_intersets)-1):
        plan = np.identity(len(src_trainset))
        all_domains3 += generate_domains(generated_domains, encoded_intersets[i], encoded_intersets[i+1])[0]
    _, generated_acc3 = self_train(args, model_copy3.mlp, all_domains3, epochs=10)

    with open(f"logs/mnist_{target}_{generated_domains}_ablation.txt", "a") as f:
        f.write(f"seed{args.seed}generated{generated_domains},{round(direct_acc, 2)},{round(st_acc, 2)},{round(st_acc_all, 2)},{round(generated_acc1, 2)},{round(generated_acc4.item(), 2)},{round(generated_acc2, 2)},{round(generated_acc3, 2)}\n")


def run_portraits_experiment(gt_domains, generated_domains):
    t = time.time()

    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y) = make_portraits_data(1000, 1000, 14000, 2000, 1000, 1000)
    tr_x, tr_y = np.concatenate([src_tr_x, src_val_x]), np.concatenate([src_tr_y, src_val_y])
    ts_x, ts_y = np.concatenate([trg_val_x, trg_test_x]), np.concatenate([trg_val_y, trg_test_y])

    encoder = ENCODER().to(device)
    transforms = ToTensor()

    src_trainset = EncodeDataset(tr_x, tr_y.astype(int), transforms)
    tgt_trainset = EncodeDataset(ts_x, ts_y.astype(int), transforms)
    source_model = get_source_model(args, src_trainset, src_trainset, 2, mode="portraits", encoder=encoder, epochs=20)
    model_copy = copy.deepcopy(source_model)

    def get_domains(n_domains):
        domain_set = []
        n2idx = {0:[], 1:[3], 2:[2,4], 3:[1,3,5], 4:[0,2,4,6], 7:[0,1,2,3,4,5,6]}
        domain_idx = n2idx[n_domains]
        for i in domain_idx:
            start, end = i*2000, (i+1)*2000
            domain_set.append(EncodeDataset(inter_x[start:end], inter_y[start:end].astype(int), transforms))
        return domain_set

    all_sets = get_domains(gt_domains)
    all_sets.append(tgt_trainset)
    
    direct_acc, st_acc, direct_acc_all, st_acc_all, generated_acc = run_goat(model_copy, source_model, src_trainset, tgt_trainset, all_sets, generated_domains, epochs=5)

    elapsed = round(time.time() - t, 2)
    with open(f"logs/portraits_exp_time.txt", "a") as f:
        f.write(f"seed{args.seed}with{gt_domains}gt{generated_domains}generated,{round(direct_acc, 2)},{round(st_acc, 2)},{round(direct_acc_all, 2)},{round(st_acc_all, 2)},{round(generated_acc, 2)}\n")
        # f.write(f"seed{args.seed}with{gt_domains}gt{generated_domains}generated,"
        #         f"{round(direct_acc.item(), 2)},{round(st_acc.item(), 2)},{round(direct_acc_all.item(), 2)},"
        #         f"{round(st_acc_all.item(), 2)},{round(generated_acc.item(), 2)}\n")


def run_covtype_experiment(gt_domains, generated_domains):
    data = make_cov_data(40000, 10000, 400000, 50000, 25000, 20000)
    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y) = data
    
    src_trainset = EncodeDataset(torch.from_numpy(src_val_x).float(), src_val_y.astype(int))
    tgt_trainset = EncodeDataset(torch.from_numpy(trg_test_x).float(), torch.tensor(trg_test_y.astype(int)))

    encoder = MLP_Encoder().to(device)
    source_model = get_source_model(args, src_trainset, src_trainset, 2, mode="covtype", encoder=encoder, epochs=5)
    model_copy = copy.deepcopy(source_model)

    def get_domains(n_domains):
        domain_set = []
        n2idx = {0:[], 1:[6], 2:[3,7], 3:[2,5,8], 4:[2,4,6,8], 5:[1,3,5,7,9], 10: range(10), 200: range(200)}
        domain_idx = n2idx[n_domains]
        # domain_idx = range(n_domains)
        for i in domain_idx:
            # start, end = i*2000, (i+1)*2000
            # start, end = i*10000, (i+1)*10000
            start, end = i*40000, i*40000 + 2000
            domain_set.append(EncodeDataset(torch.from_numpy(inter_x[start:end]).float(), inter_y[start:end].astype(int)))
        return domain_set
    
    all_sets = get_domains(gt_domains)
    all_sets.append(tgt_trainset)

    direct_acc, st_acc, direct_acc_all, st_acc_all, generated_acc = run_goat(model_copy, source_model, src_trainset, tgt_trainset, all_sets, generated_domains, epochs=5)

    with open(f"logs/covtype_exp_{args.log_file}.txt", "a") as f:
            f.write(f"seed{args.seed}with{gt_domains}gt{generated_domains}generated,{round(direct_acc, 2)},{round(st_acc, 2)},{round(st_acc_all, 2)},{round(generated_acc, 2)}\n")


def run_color_mnist_experiment(gt_domains, generated_domains):
    shift = 1
    total_domains = 20

    src_tr_x, src_tr_y, src_val_x, src_val_y, dir_inter_x, dir_inter_y, dir_inter_x, dir_inter_y, trg_val_x, trg_val_y, trg_test_x, trg_test_y = ColorShiftMNIST(shift=shift)
    inter_x, inter_y = transform_inter_data(dir_inter_x, dir_inter_y, 0, shift, interval=len(dir_inter_x)//total_domains, n_domains=total_domains)

    src_x, src_y = np.concatenate([src_tr_x, src_val_x]), np.concatenate([src_tr_y, src_val_y])
    tgt_x, tgt_y = np.concatenate([trg_val_x, trg_test_x]), np.concatenate([trg_val_y, trg_test_y])
    src_trainset, tgt_trainset = EncodeDataset(src_x, src_y.astype(int), ToTensor()), EncodeDataset(trg_val_x, trg_val_y.astype(int), ToTensor())

    encoder = ENCODER().to(device)
    vae = VAE(x_dim=28*28, z_dim=16).to(device)
    vae_path = f'models/colored_mnist/vae.pt'
    if os.path.exists(vae_path):
        vae.load_state_dict(torch.load(vae_path))
    else:
        train_vae(vae, trainloader, valloader, testloader, vae_path, save=True)

    source_model = get_source_model(args, src_trainset, src_trainset, 10, "mnist", encoder=vae.encoder, epochs=20)
    model_copy = copy.deepcopy(source_model)

    def get_domains(n_domains):
        domain_set = []
        domain_idx = []
        if n_domains == total_domains:
            domain_idx = range(n_domains)
        else:
            for i in range(1, n_domains+1):
                domain_idx.append(total_domains // (n_domains+1) * i)
                
        interval = 42000 // total_domains
        for i in domain_idx:
            start, end = i*interval, (i+1)*interval
            domain_set.append(EncodeDataset(inter_x[start:end], inter_y[start:end].astype(int), ToTensor()))
        return domain_set

    all_sets = get_domains(gt_domains)
    all_sets.append(tgt_trainset)

    images = tgt_trainset.data[:5]

    # Ensure the images are in the correct shape for plotting
    # Remove the last dimension if it's a single channel (grayscale)
    images = images.squeeze(-1)

    # Plot the images
    plt.figure(figsize=(10, 2))  # Set a wide figure size for 5 images
    for i, img in enumerate(images):
        plt.subplot(1, 5, i + 1)  # Create a subplot (1 row, 5 columns)
        plt.imshow(img, cmap="gray")  # Display the image in grayscale
        plt.axis("off")  # Turn off the axis
        plt.title(f"Image {i + 1}")

    plt.tight_layout()  # Adjust spacing
    plt.show()

    # breakpoint()
    direct_acc, st_acc, direct_acc_all, st_acc_all, generated_acc = run_goat(model_copy, source_model, src_trainset, tgt_trainset, all_sets, generated_domains, epochs=10)
        
    with open(f"logs/color{args.log_file}.txt", "a") as f:
        values = [direct_acc, st_acc, direct_acc_all, st_acc_all, generated_acc]
        values = [round(v.item(), 2) if isinstance(v, torch.Tensor) else round(v, 2) for v in values]
        f.write(f"seed{args.seed}with{gt_domains}gt{generated_domains}generated,{','.join(map(str, values))}\n")

        # f.write(f"seed{args.seed}with{gt_domains}gt{generated_domains}generated,{round(direct_acc, 2)},{round(st_acc, 2)},{round(direct_acc_all, 2)},{round(st_acc_all, 2)},{round(generated_acc, 2)}\n")
        # f.write(f"seed{args.seed}with{gt_domains}gt{generated_domains}generated,"
        #         f"{round(direct_acc, 2)},{round(st_acc, 2)},{round(direct_acc_all, 2)},"
        #         f"{round(st_acc_all, 2)},{round(generated_acc.item(), 2)}\n")




def log_generated_images_tensorboard(writer, images, epoch, tag='Generated Images'):
    """
    Log a grid of generated images to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter.
        images (torch.Tensor or np.ndarray): Generated images.
        epoch (int): Current epoch number for logging.
        tag (str): Tag name for the images in TensorBoard.
    """
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    
    # Handle different tensor shapes
    if images.ndim == 3 and images.shape[2] == 32:
        # Shape: (H, W, N) where N=32 is batch size
        H, W, N = images.shape
        C = 1  # Assuming grayscale; adjust if necessary

        # Rearrange to (N, C, H, W)
        images = images.transpose(2, 0, 1)  # Now (N, H, W)
        images = images.reshape(N, C, H, W)  # Now (N, C, H, W)
        print(f"Rearranged images to shape: {images.shape}")
    elif images.ndim == 4:
        # Shape: (N, C, H, W)
        pass  # Already in the correct format
    else:
        print(f"Unsupported image shape for TensorBoard logging: {images.shape}")
        return  # Exit the function to prevent errors

    # Handle images with different channel counts
    if images.shape[1] > 3:
        # Select the first 3 channels for RGB visualization
        images = images[:, :3, :, :]
        print(f"Selected first 3 channels from {images.shape[1]} channels for TensorBoard visualization.")
    elif images.shape[1] == 1:
        # Duplicate the single channel to create RGB images
        images = images.repeat(1, 3, 1, 1)
        print("Duplicated single channel to 3 channels for TensorBoard visualization.")
    elif images.shape[1] == 3:
        pass  # RGB images are fine
    else:
        # For unexpected channel counts, select the first 3 or adjust as needed
        images = images[:, :3, :, :] if images.shape[1] > 3 else images.repeat(1, 3, 1, 1)
        print(f"Adjusted images to 3 channels for TensorBoard visualization.")

    # Log images (grid format)
    grid = torchvision.utils.make_grid(images[:16], nrow=4, normalize=True)
    writer.add_image(tag, grid, epoch)
    print(f"Logged generated images for epoch {epoch} to TensorBoard under tag '{tag}'.")





def main(args):
    writer = init_tensorboard()
    print(args)
    if args.dataset == "mnist":
        if args.mnist_mode == "normal":
            run_mnist_experiment(args.rotation_angle, args.gt_domains, args.generated_domains) 
        else:
           run_mnist_ablation(args.rotation_angle, args.gt_domains, args.generated_domains)
    else:
        eval(f"run_{args.dataset}_experiment({args.gt_domains}, {args.generated_domains})")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="GOAT experiments")
    parser.add_argument("--dataset", choices=["mnist", "portraits", "covtype", "color_mnist"],default="mnist")
    parser.add_argument("--gt-domains", default=1, type=int)
    parser.add_argument("--generated-domains", default=2, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--mnist-mode", default="normal", choices=["normal", "ablation"])
    parser.add_argument("--rotation-angle", default=90, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--log-file", default="")
    args = parser.parse_args()

    main(args)