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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# ===== K-MEANS++ BASELINE (drop-in) =====
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def plot_encoded_domains(encoded_source, encoded_inter, encoded_target, title_src="Encoded Source", title_inter = "Encoded Inter",
                         title_tgt="Encoded Target", method='goat', save_dir = 'plots',pca=None):

    src_data = torch.tensor(encoded_source.data) if not torch.is_tensor(encoded_source.data) else encoded_source.data
    inter_data = torch.tensor(encoded_inter.data) if not torch.is_tensor(encoded_inter.data) else encoded_inter.data
    tgt_data = torch.tensor(encoded_target.data) if not torch.is_tensor(encoded_target.data) else encoded_target.data
    src_data = src_data.reshape(src_data.shape[0], -1)
    inter_data = inter_data.reshape(inter_data.shape[0], -1)
    tgt_data = tgt_data.reshape(tgt_data.shape[0], -1)


    all_data = torch.cat([src_data, inter_data, tgt_data], dim=0).view(len(src_data) + len(inter_data) + len(tgt_data), -1)
    fit_data = torch.cat([src_data, tgt_data], dim=0).view(len(src_data) + len(tgt_data), -1)
    if pca is None:
        fit_data = torch.cat([src_data, tgt_data], dim=0)
        pca = PCA(n_components=2)
        pca.fit(fit_data.cpu().numpy())

    z_all = pca.transform(all_data.cpu().numpy())

    z_src = z_all[:len(src_data)]
    z_inter = z_all[len(src_data):len(src_data) + len(inter_data)]
    z_tgt = z_all[len(src_data) + len(inter_data):]




    y_src = encoded_source.targets.cpu().numpy()
    y_inter = encoded_inter.targets.cpu().numpy()
    y_tgt = encoded_target.targets.cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)

    for c in np.unique(y_src):
        axs[0].scatter(z_src[y_src == c, 0], z_src[y_src == c, 1], label=f"Class {c}", alpha=0.6, s=10)
    axs[0].set_title(title_src)
    axs[0].set_xlabel("PC 1")
    axs[0].set_ylabel("PC 2")
    axs[0].legend()
    axs[0].grid(True)

    for c in np.unique(y_inter):
        axs[1].scatter(z_inter[y_inter == c, 0], z_inter[y_inter == c, 1], label=f"Class {c}", alpha=0.6, s=10)
    axs[1].set_title(title_inter)
    axs[1].set_xlabel("PC 1")
    axs[1].legend()
    axs[1].grid(True)

    for c in np.unique(y_tgt):
        axs[2].scatter(z_tgt[y_tgt == c, 0], z_tgt[y_tgt == c, 1], label=f"Class {c}", alpha=0.6, s=10)
    axs[2].set_title(title_tgt)
    axs[2].set_xlabel("PC 1")
    axs[2].legend()
    axs[2].grid(True)
    os.makedirs(save_dir, exist_ok=True)
    plt.suptitle("Encoded Source vs Target Projections")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/encoded_domains_{method}.png")
    plt.close()

    return pca  # optionally return the fitted PCA


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

def get_source_model(args, trainset, testset, n_class, mode, encoder=None, epochs=50, verbose=True, model_path="cache{args.ssl_weight}/source_model.pth", target_dataset = None, ssl_weight = 0.5, force_recompute=False):

    model = Classifier(encoder, MLP(mode=mode, n_class=n_class, hidden=1024)).to(device)
    if os.path.exists(model_path) and not force_recompute:
        print(f"✅ Loading cached trained model from {model_path}")
        model.load_state_dict(torch.load(model_path))
        return model

    model = get_source_model_old(args, trainset, testset, n_class, mode, encoder, epochs=epochs, augment_fn=augment,target_dataset=target_dataset, ssl_weight=args.ssl_weight, verbose=verbose)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)  # ✅ Save model for future runs
    
    return model



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


def train_joint(model, trainloader, tgt_loader, optimizer, device, augment_fn, ssl_weight=0.1):
    model.train()
    total_sup_loss = 0
    total_ssl_loss = 0

    tgt_iter = iter(tgt_loader)  # Create iterator for SSL samples

    for batch in trainloader:
        if len(batch) == 2:
            data, labels = batch
            weights = None
        else:
            data, labels, weights = batch
            weights = weights.to(device)

        data, labels = data.to(device), labels.to(device)

        # --- Supervised Forward ---
        outputs = model(data)
        if weights is None:
            sup_loss = F.cross_entropy(outputs, labels)
        else:
            criterion = nn.CrossEntropyLoss(reduction='none')
            sup_loss = criterion(outputs, labels)
            sup_loss = (sup_loss * weights).mean()

        # --- SSL Forward ---
        try:
            x_tgt, _ = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            x_tgt, _ = next(tgt_iter)

        x_tgt = x_tgt.to(device)
        x1, x2 = augment_fn(x_tgt), augment_fn(x_tgt)
        z1 = F.normalize(model.encoder(x1).view(x1.size(0), -1) + 1e-6, dim=1)
        z2 = F.normalize(model.encoder(x2).view(x2.size(0), -1) + 1e-6, dim=1)
        ssl_loss = nt_xent_loss(z1, z2)

        # --- Combine Losses ---
        total_loss = sup_loss + ssl_weight * ssl_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_sup_loss += sup_loss.item()
        total_ssl_loss += ssl_loss.item()

    return total_sup_loss / len(trainloader), total_ssl_loss / len(trainloader)

def train_supervised(model, trainloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in trainloader:
        if len(batch) == 2:
            data, labels = batch
            weights = None
        else:
            data, labels, weights = batch
            weights = weights.to(device)

        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)

        if weights is None:
            loss = F.cross_entropy(outputs, labels)
        else:
            criterion = nn.CrossEntropyLoss(reduction='none')
            loss = criterion(outputs, labels)
            loss = (loss * weights).mean()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(trainloader)


def train_contrastive_ssl(model, tgt_loader, optimizer, augment_fn, device, ssl_weight=0.5):

    model.train()
    total_ssl_loss = 0

    for x_tgt, _ in tgt_loader:
        x_tgt = x_tgt.to(device)

        x1 = augment_fn(x_tgt)
        x2 = augment_fn(x_tgt)

        z1 = model.encoder(x1)
        z2 = model.encoder(x2)

        z1 = F.normalize(z1.view(z1.size(0), -1) + 1e-6, dim=1)
        z2 = F.normalize(z2.view(z2.size(0), -1) + 1e-6, dim=1)

        ssl_loss = nt_xent_loss(z1, z2) * ssl_weight

        optimizer.zero_grad()
        ssl_loss.backward()
        optimizer.step()
        total_ssl_loss += ssl_loss.item()

    return total_ssl_loss / len(tgt_loader)


def get_source_model_old(args, trainset, testset, n_class, mode, encoder=None, epochs=50, augment_fn=None, target_dataset=None, ssl_weight = 0.1,verbose=True, diet=True):
    print("🔧 Training source model...")
    model = Classifier(encoder, MLP(mode=mode, n_class=n_class, hidden=1024)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    tgt_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) if target_dataset else None

    # for epoch in range(1, epochs + 1):
    #     sup_loss = train_supervised(model, trainloader, optimizer, device)
    #     msg = f"[Epoch {epoch}] Supervised Loss: {sup_loss:.4f}"

    #     if tgt_loader and augment_fn:
    #         ssl_loss = train_contrastive_ssl(model, tgt_loader, optimizer, augment_fn, device, ssl_weight=ssl_weight)
    #         msg += f" | SSL Loss: {ssl_loss:.4f}"
    #     print(msg)

    #     if epoch % 5 == 0:
    #         test(testloader, model, verbose=verbose)

    if diet:
        train_encoder_diet(model, trainset, testset, optimizer, device)
    else:
        for epoch in range(1, epochs + 1):
            if tgt_loader and augment_fn:
                sup_loss, ssl_loss = train_supervised(model, trainloader, tgt_loader, optimizer, device, augment_fn, ssl_weight=ssl_weight)
            msg = f"[Epoch {epoch}] Supervised Loss: {sup_loss:.4f} | SSL Loss: {ssl_loss:.4f}"

           
        else:
            sup_loss = train_supervised(model, trainloader, optimizer, device)
            msg = f"[Epoch {epoch}] Supervised Loss: {sup_loss:.4f}"

        print(msg)

        if epoch % 5 == 0:
            test(testloader, model, verbose=verbose)

    return model


# def get_source_model_old(args, trainset, testset, n_class, mode, encoder=None, epochs=50, verbose=True, target_dataset=None):
#     print("Start training source model (with optional SSL)")
#     model = Classifier(encoder, MLP(mode=mode, n_class=n_class, hidden=1024)).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
#     trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
#     testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
#     if target_dataset is not None:
#         tgt_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
#         tgt_iter = iter(tgt_loader)  # infinite loop handled later

#     for epoch in range(1, epochs + 1):
#         model.train()
#         train_loss = 0
#         for _, batch in enumerate(trainloader):
#             if len(batch) == 2:
#                 data, labels = batch
#                 weight = None
#             else:
#                 data, labels, weight = batch
#                 weight = weight.to(device)

#             data = data.to(device)
#             labels = labels.to(device)
#             optimizer.zero_grad()

#             output = model(data)
#             if weight is None:
#                 loss = F.cross_entropy(output, labels)
#             else:
#                 criterion = nn.CrossEntropyLoss(reduction='none')
#                 loss = criterion(output, labels)
#                 loss = (loss * weight).mean()

#             loss.backward()
#             train_loss += loss.item()
#             optimizer.step()
#             # 🔹 Get target batch for contrastive loss

#         if target_dataset is not None:
#             unsup_loss_total = 0
#             for _, tgt_batch in enumerate(tgt_loader):

#                 x_tgt, _ = tgt_batch
#                 x_tgt = x_tgt.to(device)


#                 x1 = augment(x_tgt)
#                 x2 = augment(x_tgt)
#                 z1 = model.encoder(x1)
#                 z2 = model.encoder(x2)
#                 z1 = F.normalize(z1.view(z1.size(0), -1) + 1e-6, dim=1)
#                 z2 = F.normalize(z2.view(z2.size(0), -1) + 1e-6, dim=1)

#                 unsup_loss = nt_xent_loss(z1, z2) * 0.1
#                 optimizer.zero_grad()
#                 unsup_loss.backward()
#                 optimizer.step()
#                 unsup_loss_total += unsup_loss.item()

#             print(f"[Epoch {epoch}] Classification Loss: {train_loss / len(trainloader):.4f} | Contrastive Loss (unsup): {unsup_loss_total / len(tgt_loader):.4f}")
#         else:
#             print(f"[Epoch {epoch}] Total Loss: {train_loss / len(trainloader):.4f}")

#         if epoch % 5 == 0:
#             test(testloader, model, verbose=verbose)
    
#     return model

# def get_source_model_old(args, trainset, testset, n_class, mode, encoder=None, epochs=50, verbose=True):
#     print("Start training source model")
#     model = Classifier(encoder, MLP(mode=mode, n_class=n_class, hidden=1024)).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
#     trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
#     testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

#     for epoch in range(1, epochs+1):
#         train(epoch, trainloader, model, optimizer, verbose=verbose)
#         if epoch % 5 == 0:
#             test(testloader, model, verbose=verbose)
#     return model


def nt_xent_loss(z_i, z_j, temperature=1):
    z = torch.cat([z_i, z_j], dim=0)  # (2N, d)
    z = F.normalize(z + 1e-6, dim=1)
    similarity = z @ z.T  # cosine similarity matrix
    N = z_i.size(0)

    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels, labels], dim=0)

    mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
    similarity = similarity.masked_fill(mask, -9e15)

    similarity = similarity / temperature
    similarity = torch.clamp(similarity, min=-100, max=100)
    loss = F.cross_entropy(similarity, labels)
    return loss

import kornia.augmentation as K

augment = nn.Sequential(
    K.RandomResizedCrop((28, 28), scale=(0.8, 1.0)),
    K.RandomHorizontalFlip(),
    K.RandomAffine(degrees=10, translate=(0.1, 0.1)),
).to(device)




# Ensure dataset returns unique indices
class DatasetWithIndices(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x, _ = self.dataset[index]  # discard original label
        return x, index  # use index as pseudo-label (DIET's principle)

    def __len__(self):
        return len(self.dataset)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

def extract_features(encoder, dataset, device, batch_size=128):
    encoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = encoder(x).view(x.size(0), -1)
            features.append(z.cpu())
            labels.append(y)

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()
    return features, labels

def evaluate_linear_probe(encoder, trainset, testset, device, batch_size=128):
    X_train, y_train = extract_features(encoder, trainset, device, batch_size)
    X_test, y_test = extract_features(encoder, testset, device, batch_size)

    clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    return acc

def train_encoder_diet(model, trainset, testset, optimizer=None, device='cuda', epochs=1000, batch_size=128, lr=1e-3, label_smoothing=0.8, weight_decay=1e-5, eval_interval=20):
    print("🔁 DIET self-supervised training on target domain")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    encoder = model.encoder.to(device)
    encoder.eval()

    dummy_input = torch.randn(1, *trainset[0][0].shape).to(device)
    with torch.no_grad():
        dummy_output = encoder(dummy_input)
    flattened_dim = dummy_output.view(1, -1).shape[1]

    encoder.train()

    num_classes = len(trainset)
    W = nn.Linear(flattened_dim, num_classes, bias=False).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if optimizer is None:
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(W.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

    wrapped_dataset = DatasetWithIndices(trainset)
    loader = DataLoader(wrapped_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        encoder.train()
        W.train()
        total_loss = 0

        for x, idx in loader:
            x, idx = x.to(device), idx.to(device)
            z = encoder(x).view(x.size(0), -1)
            logits = W(z)
            loss = loss_fn(logits, idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[DIET] Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

        if testset and (epoch + 1) % eval_interval == 0:
            acc = evaluate_linear_probe(encoder, trainset, testset, device, batch_size)
            print(f"🔍 [Eval] Epoch {epoch+1}: Linear Probe Accuracy = {acc * 100:.2f}%")

    return encoder


def train_encoder_self_supervised(model, target_dataset, epochs=10, batch_size=128, lr=1e-3):
    print("🔁 Self-supervised training on target domain")

    encoder = model.encoder.to(device)
    encoder.eval()  # Just to get output shape from one forward pass

    # Get encoder output dimension dynamically
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    with torch.no_grad():
        dummy_output = encoder(dummy_input)
    flattened_dim = dummy_output.view(1, -1).shape[1]  # e.g. 32*28*28 = 25088

    encoder.train()  # Back to train mode

    projection_head = model.mlp.to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(projection_head.parameters()), lr=lr)
    loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        encoder.train()
        projection_head.train()
        total_loss = 0
        for batch_idx, x in enumerate(loader):
            if len(x) == 3:
                imgs, _, _ = x
            else:
                imgs = x[0]

            imgs1 = augment(imgs).to(device)
            imgs2 = augment(imgs).to(device)

            z1 = projection_head(encoder(imgs1))
            z2 = projection_head(encoder(imgs2))
            # print("z1 mean:", z1.mean().item(), "std:", z1.std().item())
            # print("z2 mean:", z2.mean().item(), "std:", z2.std().item())

            loss = nt_xent_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Batch loss: {loss.item():.4f}")
            # if batch_idx % 20 == 0:
            #     print(f"[Epoch {epoch+1}, Batch {batch_idx}] Grad norms:")
            #     for name, param in encoder.named_parameters():
            #         if param.requires_grad and param.grad is not None:
            #             print(f"  {name}: {param.grad.norm().item():.4e}")


        print(f"[SSL] Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")

    return encoder


def run_goat(model_copy, source_model, src_trainset, tgt_trainset, all_sets, deg_idx, generated_domains, epochs=10, target = 60):
    step_counter = 0
    # get the performance of direct adaptation from the source to target, st involves self-training on target
    # print("------------Direct adapt performance----------")
    direct_acc, st_acc = self_train(args, model_copy, [tgt_trainset], epochs=epochs)
    # get the performance of GST from the source to target, st involves self-training on target
    # Also self-train on unlabeled target 
    print("------------Self-train on pooled domains----------")
    direct_acc_all, st_acc_all = self_train(args, source_model, all_sets, epochs=epochs)
    cache_dir = f"cache{args.ssl_weight}/target{target}/"
    # encode the source and target domains
    e_src_trainset, e_tgt_trainset = get_encoded_dataset(src_trainset, cache_path=os.path.join(cache_dir, "encoded_0.pt"), encoder =source_model.encoder,  force_recompute=False), get_encoded_dataset(tgt_trainset, cache_path=os.path.join(cache_dir, f"encoded_{target}.pt"), encoder = source_model.encoder, force_recompute=False)
    pca_model = plot_encoded_domains(e_src_trainset, e_src_trainset, e_tgt_trainset, method = 'goat')

    # encode the intermediate ground-truth domains
    intersets = all_sets[:-1]
    encoded_intersets = [e_src_trainset]
    for i, interset in enumerate(intersets):
        cache_path = os.path.join(cache_dir, f"encoded_{deg_idx[i]}.pt")
        encoded_intersets.append(get_encoded_dataset(interset, cache_path=cache_path, encoder =source_model.encoder, force_recompute=False))
    encoded_intersets.append(e_tgt_trainset)

    # generate intermediate domains
    generated_acc = 0
    if generated_domains > 0:
        all_domains = []
        all_y_transported = []
        for i in range(len(encoded_intersets)-1):
            output = generate_domains(generated_domains, encoded_intersets[i], encoded_intersets[i+1])
            new_domains = output[0]
            all_domains += new_domains
            for j in range(len(new_domains)):
                # breakpoint()
                plot_encoded_domains(encoded_intersets[i], new_domains[j], encoded_intersets[i+1],
                                        title_src=f"Encoded {deg_idx[i-1] if i < len(deg_idx) else 'X'}",
                                        title_inter=f"Generated {j+1}",
                                        title_tgt=f"Encoded {deg_idx[i] if i < len(deg_idx) else 'X'}",
                                        save_dir='plots/target{}/'.format(target),
                                        method=f"goat_pair{i}_step{j}",
                                        pca=pca_model)
            
        print(f"Generating {generated_domains} beteween each pair of the {len(intersets)} ground-truth domains...")
        # # Plotting all generated intermediate domains
        # print("Plotting all generated intermediate domains...")
        # domain_idx = 0
        # for i in range(len(encoded_intersets) - 1):
        #     src_dom = encoded_intersets[i]
        #     tgt_dom = encoded_intersets[i + 1]
        #     for j in range(generated_domains):
        #         gen_dom = all_domains[domain_idx]
        #         plot_encoded_domains(src_dom, gen_dom, tgt_dom,
        #                              title_src=f"Encoded {deg_idx[i-1] if i < len(deg_idx) else 'X'}",
        #                              title_inter=f"Generated {j+1}",
        #                              title_tgt=f"Encoded {deg_idx[i] if i < len(deg_idx) else 'X'}",
        #                              save_dir='plots/target{}/'.format(target),
        #                              method=f"goat_pair{i}_step{j}",
        #                                 pca=pca_model)
        #         domain_idx += 1
        # breakpoint()
        _, generated_acc = self_train(args, source_model.mlp, all_domains, epochs=epochs, use_labels = args.use_labels)
        # plot all intermediate domains
            
    
    return direct_acc, st_acc, direct_acc_all, st_acc_all, generated_acc



def run_main_algo_oracle(model_copy, source_model, src_trainset, tgt_trainset, all_sets, deg_idx, generated_domains, epochs=10, log_file="main_algo_log.csv", target = 60):
    """
    Runs the main algorithm following the same structure as run_goat.
    """
    domain_indices = []
    domain_types = []
    # First, process source and real intermediate datasets
    datasets = [src_trainset] + all_sets  # Include source and real domains
    domain_indices.extend(range(len(datasets)))
    domain_types.extend(["source"] + ["real"] * (len(all_sets) - 1) + ["target"])  
    print("------------Direct adapt performance----------")
    direct_acc, st_acc = self_train(args, model_copy, [tgt_trainset], epochs=epochs)
    # get the performance of GST from the source to target, st involves self-training on target
    # Also self-train on unlabeled target 
    print("------------Self-train on pooled domains----------")
    direct_acc_all, st_acc_all = self_train(args, source_model, all_sets, epochs=epochs)
    cache_dir = f"cache{args.ssl_weight}/target{target}/"
    plot_dir = f"plots/target{target}/"

    os.makedirs(cache_dir, exist_ok=True)
    e_src_trainset, e_tgt_trainset = get_encoded_dataset(src_trainset, cache_path=os.path.join(cache_dir, "encoded_0.pt"), encoder =source_model.encoder, force_recompute=False), get_encoded_dataset(tgt_trainset, cache_path=os.path.join(cache_dir, f"encoded_{target}.pt"),encoder =source_model.encoder,  force_recompute=False)
    # breakpoint()
    pca_model = plot_encoded_domains(e_src_trainset, e_src_trainset, e_tgt_trainset, save_dir = plot_dir, method = 'main_algo_oracle')

    # Get the performance of direct adaptation from the source to target
    # direct_acc, st_acc = self_train(args, model_copy, [tgt_trainset], [len(all_sets)-1], ['target'], epochs=epochs, log_file="direct_"+log_file)

    # Get the performance of self-training across all ground-truth domains
    # direct_acc_all, st_acc_all = self_train(args, source_model, all_sets, domain_indices, domain_types, epochs=epochs, log_file="pool_"+log_file)

    # Encode the intermediate ground-truth domains
    intersets = all_sets[:-1]
    encoded_intersets = [e_src_trainset]
    # breakpoint()
    for i, interset in enumerate(intersets):
        cache_path = os.path.join(cache_dir, f"encoded_{deg_idx[i]}.pt")
        encoded_intersets.append(get_encoded_dataset( interset, cache_path=cache_path, encoder =source_model.encoder,))
    encoded_intersets.append(e_tgt_trainset)

    # breakpoint()
    print(f"Generating {generated_domains} beteween each pair of the {len(intersets)} ground-truth domains...")
    # Generate intermediate domains using Wasserstein interpolaftion
    generated_acc = 0
    if generated_domains > 0:
        all_domains = []
        synthetic_indices = []
        base_idx = len(domain_indices) - 1
        # ✅ Generate synthetic domains for adaptation
        for i in range(len(encoded_intersets) - 1):
            # output = generate_gauss_domains(
            #     source_dataset=encoded_intersets[i],
            #     target_dataset=encoded_intersets[i + 1],
            #     n_wsteps=generated_domains,
            #     device=device
            # )
            # generate pseudo labels for the target domain
            tgt_pseudo_labels,_ = get_pseudo_labels(encoded_intersets[i + 1], source_model.mlp, device=device)
            inter_domains = generate_domains_find_next(
                source_dataset=encoded_intersets[i],
                target_dataset=encoded_intersets[i + 1],
                tgt_pseudo_labels=tgt_pseudo_labels,
                n_wsteps=generated_domains,
                device=device
            )
            all_domains.extend(inter_domains)

            # all_domains.extend(inter_domains)
            synthetic_indices.extend([base_idx + i + 1] * generated_domains)
        print("Plotting all generated intermediate domains...")
        domain_idx = 0
        for i in range(len(encoded_intersets) - 1):
            src_dom = encoded_intersets[i]
            tgt_dom = encoded_intersets[i + 1]
            for j in range(generated_domains):
                gen_dom = all_domains[domain_idx]
                # breakpoint()
                plot_encoded_domains(src_dom, gen_dom, tgt_dom,
                                    title_src=f"Encoded {deg_idx[i-1] if i < len(deg_idx) else 'X'}",
                                    title_inter=f"Generated {j+1}",
                                    title_tgt=f"Encoded {deg_idx[i] if i < len(deg_idx) else 'X'}",
                                    save_dir=plot_dir,
                                    method=f"oracle_gen_pair{i}_step{j}",
                                    pca=pca_model)
                domain_idx += 1

        # Train on synthetic domains and log
        domain_indices.extend(synthetic_indices)
        domain_types.extend(["synthetic"] * len(synthetic_indices))
        # breakpoint()
        _, generated_acc = self_train(args, source_model.mlp, all_domains, epochs=epochs)

    return 0, 0, 0, 0, generated_acc



def run_main_algo(model_copy, source_model, src_trainset, tgt_trainset, all_sets, deg_idx, generated_domains, epochs=10, log_file="main_algo_log.csv", target = 60, stop_threshold = 3):
    """
    Runs the main algorithm following the same structure as run_goat.
    """
    domain_indices = []
    domain_types = []
    # First, process source and real intermediate datasets
    # breakpoint()
    datasets = [src_trainset] + all_sets  # Include source and real domains
    domain_indices.extend(range(len(datasets)))
    domain_types.extend(["source"] + ["real"] * (len(all_sets) - 1) + ["target"])
    direct_acc, st_acc = self_train(args, model_copy, [tgt_trainset], epochs=epochs)
    # get the performance of GST from the source to target, st involves self-training on target
    # Also self-train on unlabeled target 
    print("------------Self-train on pooled domains----------")
    direct_acc_all, st_acc_all = self_train(args, source_model, all_sets, epochs=epochs)
    cache_dir = f"cache{args.ssl_weight}/target{target}/"
    plot_dir = f"plots/target{target}/"
    os.makedirs(cache_dir, exist_ok=True)
    e_src_trainset, e_tgt_trainset = get_encoded_dataset(src_trainset, cache_path=os.path.join(cache_dir, "encoded_0.pt"), encoder = source_model.encoder),get_encoded_dataset(tgt_trainset, cache_path=os.path.join(cache_dir, f"encoded_{target}.pt"),encoder =source_model.encoder)
    pca_model = plot_encoded_domains(e_src_trainset, e_src_trainset, e_tgt_trainset, method = 'main_algo')
    # Encode the intermediate ground-truth domains
    intersets = all_sets[:-1]
    encoded_intersets = [e_src_trainset]
    # breakpoint()
    for i, interset in enumerate(intersets):
        cache_path = os.path.join(cache_dir, f"encoded_{deg_idx[i]}.pt")
        encoded_intersets.append(get_encoded_dataset( interset, cache_path=cache_path, encoder =source_model.encoder, force_recompute=False))
    encoded_intersets.append(e_tgt_trainset)

    # breakpoint()
    print(f"Generating {generated_domains} beteween each pair of the {len(intersets)} ground-truth domains...")
    # Generate intermediate domains using Wasserstein interpolaftion
    generated_acc = 0
    if generated_domains > 0:
        all_domains = []
        synthetic_indices = []
        base_idx = len(domain_indices) - 1
        teacher = source_model.mlp
        student = copy.deepcopy(teacher)
        student = student.to(device)
        # ✅ Generate synthetic domains for adaptation
        dom_count = 1
        for i in range(len(encoded_intersets) - 1):
            w2_target = 99
            src_encoded = encoded_intersets[i]
            tgt_encoded = encoded_intersets[i + 1]
            source_loader = DataLoader(encoded_intersets[i], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            tgt_loader = DataLoader(encoded_intersets[i + 1], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            tgt_pseudo_labels,_ = get_pseudo_labels(tgt_loader, teacher)
            if i == 0:
                src_pseudo_labels = encoded_intersets[i].targets
            else:
                breakpoint()
                teacher_before = copy.deepcopy(teacher)
                src_pseudo_labels,_ = get_pseudo_labels(source_loader, teacher)
                def models_equal(model1, model2):
                    for p1, p2 in zip(model1.parameters(), model2.parameters()):
                        if not torch.equal(p1, p2):
                            return False
                    return True

                print("Teacher was modified:", not models_equal(teacher, teacher_before))

                src_encoded.targets = src_pseudo_labels
            # for step in range(generated_domains):
            while w2_target > stop_threshold and dom_count <= generated_domains * 3:
                # Generate one intermediate domain
                # w2_target, domain = generate_gauss_domains(
                #     src_encoded,
                #     tgt_encoded,
                #     n_wsteps=1,
                #     device=device,
                # )  # pick the first step only
                
                domain, w2_target = generate_domains_find_next(
                    src_encoded,
                    src_pseudo_labels,
                    tgt_encoded,
                    tgt_pseudo_labels,
                    n_wsteps=1,
                    device=device,)
                domain = domain[0]
                src_encoded = domain
                # breakpoint()
                
                # Self-train on the generated domain
                st_acc, teacher = self_train_one_domain(
                    args,
                    teacher,
                    domain,
                    tgt_encoded,
                    epochs=epochs,
                    source_idx = dom_count,
                )
                dom_count += 1
                print(f"W2 distance to target: {w2_target}")

                all_domains.append(domain)
            # all_domains.extend(inter_domains)

            synthetic_indices.extend([base_idx + i + 1] * generated_domains)
        
        tgt_loader = DataLoader(encoded_intersets[-1], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        tgt_pseudo_labels,_ = get_pseudo_labels(tgt_loader, teacher)
        tgt_pseudo_labeled = copy.deepcopy(encoded_intersets[-1])
        tgt_pseudo_labeled.targets = tgt_pseudo_labels

        st_acc, teacher = self_train_one_domain(
            args,
            teacher,
            tgt_pseudo_labeled,
            e_tgt_trainset,
            epochs=epochs,
            source_idx = dom_count,
        )
        dom_count += 1

        print("Plotting all generated intermediate domains...")
        domain_idx = 0
        for i in range(len(encoded_intersets) - 1):
            src_dom = encoded_intersets[i]
            tgt_dom = encoded_intersets[i + 1]
            for j in range(generated_domains):
                gen_dom = all_domains[domain_idx]
                # breakpoint()
                plot_encoded_domains(src_dom, gen_dom, tgt_dom,
                                    title_src=f"Encoded {deg_idx[i-1] if i < len(deg_idx) else 'X'}",
                                    title_inter=f"Generated {j+1}",
                                    title_tgt=f"Encoded {deg_idx[i] if i < len(deg_idx) else 'X'}",
                                    save_dir=plot_dir,
                                    method=f"main_algo_gen_pair{i}_step{j}",
                                    pca=pca_model)
                domain_idx += 1

        # Train on synthetic domains and log
        domain_indices.extend(synthetic_indices)
        domain_types.extend(["synthetic"] * len(synthetic_indices))
    else:
        breakpoint()
        _, generated_acc = self_train(args, source_model.mlp, all_domains, epochs=epochs)
        

    return 0, 0, 0, 0, generated_acc


# ===== K-MEANS++ BASELINE (drop-in) =====
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

@torch.no_grad()
def predict_source_labels(model, dataset, device, batch_size=256):
    """Proxy labels on target from the SOURCE model (unsupervised wrt ground truth)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    for x, _ in loader:
        x = x.to(device)
        logit = model(x)
        preds.append(logit.argmax(dim=1).cpu())
    return torch.cat(preds).numpy()

def hungarian_map_clusters_to_labels(cluster_ids, proxy_labels, n_classes):
    """
    Map cluster -> label by maximizing agreement with proxy_labels (from source model).
    Returns mapped_labels (same length as cluster_ids) and the mapping dict.
    """
    C = np.zeros((n_classes, n_classes), dtype=int)  # [cluster, label]
    for c in range(n_classes):
        for y in range(n_classes):
            C[c, y] = np.sum((cluster_ids == c) & (proxy_labels == y))

    # maximize matches -> minimize negative counts
    row_ind, col_ind = linear_sum_assignment(C.max() - C)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    mapped = np.vectorize(mapping.get)(cluster_ids)
    return mapped, mapping

def nearest_centroid_predict(X, centroids):
    # returns cluster index per row
    # X: (N,d), centroids: (K,d)
    d2 = ((X[:, None, :] - centroids[None, :, :])**2).sum(axis=2)
    return d2.argmin(axis=1)


def _to_2d_numpy(x, pool="flatten"):
    """
    x: torch.Tensor or np.ndarray with shape (N, …).
    Returns np.ndarray of shape (N, D) suitable for sklearn.
    pool:
      - "flatten": flatten all non-batch dims
      - "gap": global average pool over spatial dims, keeps channels (N, C)
    """
    t = torch.as_tensor(x)  # handles numpy, too
    if t.ndim > 2:
        if pool == "gap":
            # average over all non-(N,C) dims
            # common case: (N, C, H, W) → mean over H,W
            reduce_dims = tuple(range(2, t.ndim))
            t = t.mean(dim=reduce_dims)
        else:
            t = t.view(t.size(0), -1)
    return t.detach().cpu().numpy()

def run_kmeanspp_baseline(args, target_angle, n_classes=10, pool="gap", n_init=10):
    # load cached encoded datasets
    cache_dir = f"cache{args.ssl_weight}/target{target_angle}/"
    e_src = torch.load(f"{cache_dir}/encoded_0.pt")
    e_tgt = torch.load(f"{cache_dir}/encoded_{target_angle}.pt")

    # features → 2D numpy
    X_tr  = _to_2d_numpy(e_src.data, pool=pool)   # (N, D)
    X_tgt = _to_2d_numpy(e_tgt.data, pool=pool)   # (N, D)

    # (optional) get labels if you want to report true acc
    y_tgt = e_tgt.targets.cpu().numpy() if torch.is_tensor(e_tgt.targets) else np.asarray(e_tgt.targets)

    # k-means++ on target (or on source if you prefer)
    km = KMeans(
        n_clusters=n_classes,
        init="k-means++",
        n_init=n_init,
        max_iter=300,
        random_state=args.seed
    )
    cluster_ids = km.fit_predict(X_tgt)

    # if you want a quick unsupervised accuracy proxy:
    # map clusters → labels by majority vote (needs y_tgt only for reporting)
    if y_tgt is not None:
        mapping = {}
        for c in range(n_classes):
            mask = (cluster_ids == c)
            if mask.any():
                vals, cnts = np.unique(y_tgt[mask], return_counts=True)
                mapping[c] = vals[cnts.argmax()]
            else:
                mapping[c] = 0
        y_hat = np.vectorize(mapping.get)(cluster_ids)
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_tgt, y_hat)
        print(f"[KMeans++] pooled='{pool}'  acc={acc:.4f}")

    return km, cluster_ids


def run_mnist_experiment(target, gt_domains, generated_domains):

    src_trainset, tgt_trainset = get_single_rotate(False, 0), get_single_rotate(False, target)
    # breakpoint()
    encoder = ENCODER().to(device)

    # 🧠 Step 1: Train on source (supervised)
    source_model = get_source_model(
        args,
        src_trainset,
        tgt_trainset,
        n_class=10,
        mode="mnist",
        encoder=encoder,
        epochs=5,
        model_path=f"diet_source_models/target{target}/mnist_source_model_{target}.pth",
        target_dataset=tgt_trainset,
        force_recompute=False
    )

    # 🧪 Step 2: (Optional) Improve encoder using self-supervised learning on target domain
    # encoder = train_encoder_self_supervised(
    #     model=source_model,
    #     target_dataset=tgt_trainset,
    #     epochs=10,  # you can tune this
    #     batch_size=args.batch_size,
    #     lr=1e-3
    # )

    # 🔁 Step 3: Reattach classifier head to updated encoder
    # source_model = Classifier(encoder, MLP(mode="mnist", n_class=10, hidden=1024)).to(device)

    # ✅ Step 4: Copy for adaptation algorithm

    source_model_main_algo = copy.deepcopy(source_model)
    source_model_goat = copy.deepcopy(source_model)


    main_model_copy = copy.deepcopy(source_model_main_algo)  # used in run_main_algo
    goat_model_copy = copy.deepcopy(source_model_goat)  # used in run_goat



    all_sets = []
    deg_idx = []
    for i in range(1, gt_domains+1):
        all_sets.append(get_single_rotate(False, i*target//(gt_domains+1)))
        deg_idx.append(i*target//(gt_domains+1))
        print(i*target//(gt_domains+1))
    all_sets.append(tgt_trainset)
    deg_idx.append(target)
    # breakpoint()
    _, _, _, _, main_algo_acc = run_main_algo(main_model_copy, source_model_main_algo, src_trainset, tgt_trainset, all_sets, deg_idx, generated_domains, epochs=5, target = target)
    # _, _, _, _, main_algo_acc = run_main_algo(main_model_copy, source_model_main_algo, src_trainset, tgt_trainset, all_sets, deg_idx, generated_domains, epochs=5, target = target)
    # direct_acc, st_acc, direct_acc_all, st_acc_all, generated_acc = run_goat(goat_model_copy, source_model_goat, src_trainset, tgt_trainset, all_sets, deg_idx,generated_domains, epochs=5, target = target)

    # with open(f"logs/mnist_{target}_{gt_domains}_layer.txt", "a") as f:
    #     f.write(f"seed{args.seed}with{gt_domains}gt{generated_domains}generated,{round(direct_acc, 2)},{round(st_acc, 2)},{round(direct_acc_all, 2)},{round(st_acc_all, 2)},{round(generated_acc, 2)}, Main Algorithm: {round(main_algo_acc, 2)}\n")



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

    e_src_trainset, e_tgt_trainset = get_encoded_dataset(src_trainset,encoder=source_model.encoder), get_encoded_dataset(tgt_trainset, encoder =source_model.encoder)
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

    os.makedirs("logs", exist_ok=True)
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
    source_model_goat = get_source_model(args, src_trainset, src_trainset, 2, mode="portraits", encoder=encoder, epochs=20, model_path=f"portraits/cache{args.ssl_weight}/source_model.pth", target_dataset=tgt_trainset, force_recompute=False)
    source_model_main = copy.deepcopy(source_model_goat)
    model_copy_goat = copy.deepcopy(source_model_goat)
    model_copy_main = copy.deepcopy(source_model_main)

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
    
    direct_acc, st_acc, direct_acc_all, st_acc_all, generated_acc = run_goat(model_copy_goat, source_model_goat, src_trainset, tgt_trainset, all_sets, 0,generated_domains, epochs=5)
    direct_acc, st_acc, direct_acc_all, st_acc_all, generated_acc = run_main_algo(model_copy_main, source_model_main, src_trainset, tgt_trainset, all_sets, generated_domains, epochs=5)

    elapsed = round(time.time() - t, 2)
    os.makedirs("logs", exist_ok=True)
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

    os.makedirs("logs", exist_ok=True)
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
    
    os.makedirs("logs", exist_ok=True)
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
    # run_kmeanspp_baseline(args, target_angle=args.rotation_angle, n_classes=10)
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
    parser.add_argument("--gt-domains", default=0, type=int)
    parser.add_argument("--generated-domains", default=3, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--mnist-mode", default="normal", choices=["normal", "ablation"])
    parser.add_argument("--rotation-angle", default=45, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--log-file", default="")
    parser.add_argument("--ssl-weight", default=0.1, type=float)
    parser.add_argument("--use-labels", action="store_true", help="Use translable")
    args = parser.parse_args()

    main(args)