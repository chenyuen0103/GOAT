import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split, Subset, TensorDataset
import torch.optim as optim
from train_model import *
from util import *
from dataset import *
from ot_util import *
from model import *
import copy
import csv
import os 
from sklearn.decomposition import PCA


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def plot_encoded_domains(encoded_source, encoded_target, title_src="Encoded Source", title_tgt="Encoded Target", save_dir = 'plots',method='goat', pca=None):
    """
    Plots 2D projections of encoded source and target domains using PCA.
    If a PCA object is provided, uses it instead of fitting a new one.
    """
    src_data = torch.tensor(encoded_source.data) if not torch.is_tensor(encoded_source.data) else encoded_source.data
    tgt_data = torch.tensor(encoded_target.data) if not torch.is_tensor(encoded_target.data) else encoded_target.data

    all_data = torch.cat([src_data, tgt_data], dim=0).view(len(src_data) + len(tgt_data), -1)

    if pca is None:
        pca = PCA(n_components=2)
        z_all = pca.fit_transform(all_data.cpu().numpy())
    else:
        z_all = pca.transform(all_data.cpu().numpy())

    # Split back
    z_src = z_all[:len(src_data)]
    z_tgt = z_all[len(src_data):]

    y_src = encoded_source.targets.cpu().numpy()
    y_tgt = encoded_target.targets.cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for c in np.unique(y_src):
        axs[0].scatter(z_src[y_src == c, 0], z_src[y_src == c, 1], label=f"Class {c}", alpha=0.6, s=10)
    axs[0].set_title(title_src)
    axs[0].set_xlabel("PC 1")
    axs[0].set_ylabel("PC 2")
    axs[0].legend()
    axs[0].grid(True)

    for c in np.unique(y_tgt):
        axs[1].scatter(z_tgt[y_tgt == c, 0], z_tgt[y_tgt == c, 1], label=f"Class {c}", alpha=0.6, s=10)
    axs[1].set_title(title_tgt)
    axs[1].set_xlabel("PC 1")
    axs[1].legend()
    axs[1].grid(True)

    plt.suptitle("Encoded Source vs Target Projections")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/encoded_domains_{method}.png")
    plt.close()

    return pca  # optionally return the fitted PCA

def get_pseudo_labels(dataloader, model, confidence_q=0.1, temperature: float = 1.0, device_override=None):
    """Return high-confidence pseudo labels using probability margins."""

    if isinstance(dataloader, Dataset):
        dataloader = DataLoader(dataloader, batch_size=256, shuffle=False, num_workers=0)

    logits_list = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                data = batch[0]
            else:
                data = batch
            data = data.to(device if device_override is None else device_override)
            logits_list.append(model(data))

    logits = torch.cat(logits_list, dim=0)
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=1)
    top_probs, top_idx = probs.topk(k=min(2, probs.size(1)), dim=1)
    if top_probs.size(1) == 1:
        margins = top_probs[:, 0]
    else:
        margins = top_probs[:, 0] - top_probs[:, 1]

    keep_fraction = max(0.0, min(1.0, 1.0 - confidence_q))
    threshold = torch.quantile(margins, keep_fraction) if margins.numel() > 0 else 0.0
    keep = torch.nonzero(margins >= threshold, as_tuple=False).squeeze(1)

    labels = top_idx[:, 0].to(torch.int64).cpu()
    return labels, list(keep.cpu().numpy())


def _head_module_for_training(source_model: nn.Module) -> nn.Module:
    if hasattr(source_model, "predict"):
        module = source_model.predict
    elif hasattr(source_model, "classifier"):
        module = source_model.classifier
    else:
        raise AttributeError("Expected source_model to have .predict or .classifier")
    if not isinstance(module, nn.Module):
        raise TypeError("Head module must be an nn.Module")
    return module


def _feature_dataset_from_encoded(ds) -> TensorDataset:
    if isinstance(ds, TensorDataset):
        tensors = ds.tensors
        if len(tensors) < 2:
            raise ValueError("TensorDataset must contain features and labels")
        feats = tensors[0]
        labels = tensors[1]
        return TensorDataset(feats.float(), labels.long())

    data = getattr(ds, "data", None)
    labels = getattr(ds, "targets", None)
    if labels is None:
        labels = getattr(ds, "targets_em", None)
    if data is None or labels is None:
        raise ValueError("Encoded dataset must provide data and targets/targets_em")

    data_tensor = torch.as_tensor(data).float()
    labels_tensor = torch.as_tensor(labels).long()
    return TensorDataset(data_tensor, labels_tensor)


def self_train_head(args, source_model, feature_domains, encoded_target, epochs=10):
    if len(feature_domains) == 0:
        return None, 0.0

    teacher = copy.deepcopy(source_model).to(device)
    head = _head_module_for_training(teacher)

    head_param_ids = {id(p) for p in head.parameters()}
    for param in teacher.parameters():
        param.requires_grad = id(param) in head_param_ids

    head_lr = getattr(args, "head_lr", args.lr)
    label_smoothing = getattr(args, "label_smoothing", 0.0)
    optimizer = optim.Adam(head.parameters(), lr=head_lr, weight_decay=1e-4)

    def _make_loader(domain, shuffle):
        dataset = domain if isinstance(domain, TensorDataset) else _feature_dataset_from_encoded(domain)
        return DataLoader(dataset, batch_size=getattr(args, "batch_size", 256), shuffle=shuffle, num_workers=0)

    for domain in feature_domains:
        loader = _make_loader(domain, shuffle=True)
        for epoch in range(epochs):
            head.train()
            for z_batch, y_batch in loader:
                z_batch = z_batch.to(device).float()
                y_batch = y_batch.to(device).long()
                logits = head(z_batch)
                if label_smoothing > 0.0:
                    loss = F.cross_entropy(logits, y_batch, label_smoothing=label_smoothing)
                else:
                    loss = F.cross_entropy(logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    head.eval()
    target_loader = _make_loader(encoded_target, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for z_batch, y_batch in target_loader:
            z_batch = z_batch.to(device).float()
            y_batch = y_batch.to(device).long()
            logits = head(z_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    acc = correct / total if total > 0 else 0.0
    return None, acc


def train_head_soft_on_encoded(
    args,
    head: nn.Module,
    encoded_target: Dataset,
    soft_targets: np.ndarray,
    epochs: int = 5,
    lr: float = None,
    weight_decay: float = 1e-4,
    temperature: float = 1.0,
):
    """Fine-tune a head (z->logits) on encoded features with soft class targets.

    encoded_target: EncodeDataset or TensorDataset with .data being features (N,d)
    soft_targets: (N,C) numpy/torch array aligned with encoded_target order
    """
    # Prepare tensors
    data_attr = getattr(encoded_target, "data", None)
    if data_attr is None:
        raise ValueError("encoded_target must expose .data (features)")

    feats = torch.as_tensor(data_attr).float()
    if isinstance(soft_targets, np.ndarray):
        soft_t = torch.from_numpy(soft_targets).float()
    else:
        soft_t = torch.as_tensor(soft_targets).float()
    if feats.size(0) != soft_t.size(0):
        raise ValueError(f"soft_targets length {soft_t.size(0)} != data length {feats.size(0)}")

    device = next(head.parameters()).device
    ds = TensorDataset(feats, soft_t)
    loader = DataLoader(ds, batch_size=getattr(args, "batch_size", 256), shuffle=True, num_workers=0)

    # Optimizer on head only
    opt = optim.Adam(head.parameters(), lr=(getattr(args, "head_lr", None) or lr or getattr(args, "lr", 1e-3)), weight_decay=weight_decay)

    T = max(float(temperature), 1e-6)
    head.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for z, target in loader:
            z = z.to(device)
            target = target.to(device)
            logits = head(z)
            logp = F.log_softmax(logits / T, dim=1)
            loss = (-(target * logp).sum(dim=1)).mean() * (T * T)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        print(f"[Head-Soft] Epoch {epoch}/{epochs}: loss={total/len(loader):.4f}")

    return head


def self_train_label(args, source_model, datasets, epochs=10):
    """Self-train on datasets that already store the labels to use (e.g., EM)."""
    steps = max(len(datasets) - 1, 0)
    teacher = copy.deepcopy(source_model).to(device)
    targetset = datasets[-1]
    # breakpoint()
    targetloader = DataLoader(targetset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("------------Direct adapt performance----------")
    direct_loss, direct_acc = test(targetloader, teacher)
    st_acc = direct_acc

    for i in range(steps):
        print(f"--------Training on the {i}th domain--------")
        trainset = datasets[i]

        labels_attr = getattr(trainset, "targets_em", None)
        if labels_attr is None and isinstance(trainset, TensorDataset) and len(trainset.tensors) > 1:
            labels_attr = trainset.tensors[1]
        if labels_attr is None and hasattr(trainset, "targets"):
            labels_attr = trainset.targets
        if labels_attr is None:
            raise ValueError("Dataset must carry labels (targets_em or targets) for self_train_label.")

        if isinstance(trainset, TensorDataset):
            data_tensor = trainset.tensors[0].detach().cpu()
            labels_tensor = torch.as_tensor(labels_attr, dtype=torch.long).detach().cpu()
            pseudo_dataset = TensorDataset(data_tensor.float(), labels_tensor)
        else:
            data_attr = getattr(trainset, "data", None)
            if data_attr is None:
                raise ValueError("Dataset must expose .data when not using TensorDataset.")
            transform = getattr(trainset, "transform", None)

            if torch.is_tensor(data_attr):
                feats = data_attr.detach().cpu()
            else:
                feats = torch.as_tensor(np.asarray(data_attr))
            labels_tensor = torch.as_tensor(labels_attr, dtype=torch.long)

            if transform is None:
                pseudo_dataset = TensorDataset(feats.float(), labels_tensor)
            else:
                pseudo_dataset = EncodeDataset(feats.numpy(), labels_tensor.cpu().numpy(), transform)

        trainloader = DataLoader(pseudo_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        student = copy.deepcopy(teacher).to(device)
        optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-4)

        for epoch in range(1, epochs + 1):
            train(epoch, trainloader, student, optimizer)
            if epoch % 5 == 0:
                test(targetloader, student)

        print("------------Performance on the current domain----------")
        test(trainloader, student)

        print("------------Performance on the target domain----------")
        st_loss, st_acc = test(targetloader, student)

        teacher = copy.deepcopy(student)

    if steps > 0:
        st_loss, st_acc = test(targetloader, teacher)

    return direct_acc, st_acc

# def self_train(args, source_model, datasets, domain_indices, domain_types, epochs=10, log_file="log.csv"):
#     steps = len(datasets)
#     teacher = source_model
#     targetset = datasets[-1]

#     log_header = ["Domain Index", "Domain Type", "Epoch", "Train Set Size", 
#                 "Train Loss", "Train Acc", "Target Loss", "Target Acc"]
#     if not os.path.exists(log_file):
#         with open(log_file, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(log_header)
        
#     targetloader = DataLoader(targetset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
#     print("------------Direct adapt performance----------")
#     # Log direct adaptation performance (before training starts)
#     direct_loss, direct_acc = test(targetloader, teacher)

#     with open(log_file, "a", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([0, "source", 0, len(datasets[0]), "-", "-", round(target_loss, 4), round(target_acc, 4)])


#     # start self-training on intermediate domains
#     for i in range(steps):
#         domain_idx = domain_indices[i]
#         domain_type = domain_types[i]
#         print(f"--------Training on domain {domain_idx} ({domain_type}) --------")

#         print(f"--------Training on the {i}th domain--------")
#         trainset = datasets[i]
#         ogloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                
#         test(targetloader, teacher)
#         train_labs, train_idx = get_pseudo_labels(ogloader, teacher)

#         if torch.is_tensor(trainset.data):
#             data = trainset.data.cpu().detach().numpy()
#         else:
#             data = trainset.data
#         trainset  = EncodeDataset(data, train_labs, trainset.transform)
        
#         # filter out the least 10% confident data
#         filter_trainset = Subset(trainset, train_idx)
#         print("Trainset size: " + str(len(filter_trainset)))

#         trainloader =  DataLoader(filter_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

#         # initialize and train student model
#         student = copy.deepcopy(teacher)
#         optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-4)

#         for i in range(1, epochs+1):
            
#             if i % 5 == 0:
#                 train_loss = train(i, trainloader, student, optimizer)
#                 _, train_acc = test(trainloader, student)  # Accuracy on current domain
#                 target_loss, target_acc = test(targetloader, student)  # Accuracy & loss on target domain
#                 # Append log to CSV
#                 with open(log_file, "a", newline="") as f:
#                     writer = csv.writer(f)
#                     writer.writerow([domain_idx, domain_type, i, len(filter_trainset), 
#                                     round(train_loss, 4), round(train_acc, 4), 
#                                     round(target_loss, 4), round(target_acc, 4)])


#         print("------------Performance on the current domain----------")
#         test(trainloader, student)

#         # test on the target domain
#         print("------------Performance on the target domain----------")
#         st_acc = test(targetloader, student)

#         teacher = copy.deepcopy(student)
    
#     return direct_acc, st_acc


def self_train(args, source_model, datasets, epochs=10, use_labels=False):
    """Image-space self-training that holds out the last dataset for evaluation.

    Training labels:
      - Use targets_em or TensorDataset labels when provided (synthetic/encoded sets).
      - For raw image datasets, default to pseudo-labels; only use .targets if use_labels=True.
    """
    steps = max(len(datasets) - 1, 0)
    teacher = copy.deepcopy(source_model).to(device)
    targetset = datasets[-1]

    targetloader = DataLoader(targetset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("------------Direct adapt performance----------")
    direct_loss, direct_acc = test(targetloader, teacher)
    st_acc = direct_acc

    # start self-training on training domains
    for i in range(steps):
        print(f"--------Training on the {i}th domain--------")
        trainset = datasets[i]
        
        # Choose label source
        labels_attr = getattr(trainset, "targets_em", None)
        if labels_attr is None and isinstance(trainset, TensorDataset) and len(trainset.tensors) > 1:
            labels_attr = trainset.tensors[1]
        if labels_attr is None and use_labels and hasattr(trainset, "targets"):
            labels_attr = trainset.targets

        if labels_attr is not None:
            labels_tensor = torch.as_tensor(labels_attr, dtype=torch.long).cpu()
            idx_tensor = torch.arange(labels_tensor.size(0), dtype=torch.long)
        else:
            ogloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            pseudo_labels, pseudo_indices = get_pseudo_labels(ogloader, teacher)
            if len(pseudo_indices) == 0:
                print("[self_train] Skipping domain: no high-confidence pseudo-labels.")
                continue
            train_idx = np.asarray(pseudo_indices, dtype=int)
            labels_tensor = pseudo_labels.cpu()[train_idx].long()
            idx_tensor = torch.from_numpy(train_idx).long()
            # Optional filtering if true labels exist
            if use_labels and hasattr(trainset, "targets"):
                targets_np = np.asarray(trainset.targets)
                mask = targets_np[idx_tensor.numpy()] == labels_tensor.numpy()
                idx_tensor = idx_tensor[mask]
                labels_tensor = labels_tensor[mask]
                if idx_tensor.numel() == 0:
                    print("[self_train] Skipping domain after label filtering (use_labels).")
                    continue

        if isinstance(trainset, TensorDataset):
            data_tensor = trainset.tensors[0]
            pseudo_dataset = TensorDataset(
                data_tensor.index_select(0, idx_tensor).float(),
                labels_tensor.clone(),
            )
        else:
            data_attr = getattr(trainset, "data", None)
            if data_attr is None:
                raise TypeError("trainset must expose .data or be a TensorDataset when features are fixed.")

            transform = getattr(trainset, "transform", None)
            idx_numpy = idx_tensor.cpu().numpy()

            if torch.is_tensor(data_attr):
                data_tensor = data_attr.detach().cpu()
                if transform is None:
                    selected = data_tensor.index_select(0, idx_tensor)
                    pseudo_dataset = EncodeDataset(
                        selected,
                        labels_tensor.clone(),
                        None,
                    )
                else:
                    selected = data_tensor.numpy()[idx_numpy]
                    pseudo_dataset = EncodeDataset(
                        selected,
                        labels_tensor.clone(),
                        transform,
                    )
            else:
                data_np = np.asarray(data_attr)
                selected = data_np[idx_numpy]
                pseudo_dataset = EncodeDataset(
                    selected,
                    labels_tensor.clone(),
                    transform,
                )

        trainloader = DataLoader(pseudo_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        # initialize and train student model
        student = copy.deepcopy(teacher).to(device)
        optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-4)

        for i in range(1, epochs+1):
            train(i, trainloader, student, optimizer)
            if i % 5 == 0:
                 test(targetloader, student)
        print("------------Performance on the current domain----------")
        test(trainloader, student)
        # breakpoint()

        # test on the target domain
        print("------------Performance on the target domain----------")
        st_loss, st_acc = test(targetloader, student)

        teacher = copy.deepcopy(student)

    if steps > 0:
        st_loss, st_acc = test(targetloader, teacher)

    return direct_acc, st_acc




def self_train_og(args, source_model, datasets, epochs=10, use_labels=False):
    steps = len(datasets)
    teacher = source_model
    targetset = datasets[-1]
        
    targetloader = DataLoader(targetset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("------------Direct adapt performance----------")
    direct_loss, direct_acc = test(targetloader, teacher)
    st_acc = direct_acc

    # start self-training on intermediate domains
    for i in range(steps):
        print(f"--------Training on the {i}th domain--------")
        trainset = datasets[i]
        ogloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                
        test(targetloader, teacher)
        # breakpoint()
        train_labs, train_idx = get_pseudo_labels(ogloader, teacher)



        # If transported labels are available, and not the last domain
        if use_labels and i < steps - 1:
            if hasattr(trainset, "targets"):
                # breakpoint()
                train_targets = np.array(trainset.targets)
                train_idx = np.array(train_idx)  # Ensure it's a NumPy array
                train_labs = np.array(train_labs)
                match_mask = train_targets == train_labs
                keep_idx = [idx for idx in train_idx if match_mask[idx]]
                train_idx = np.array(keep_idx)

                print(f"Filtered trainset size: {len(train_labs)}")
            else:
                raise ValueError("Transported labels not available in the dataset.")


        if torch.is_tensor(trainset.data):
            data = trainset.data.cpu().detach().numpy()
        else:
            data = trainset.data
        trainset  = EncodeDataset(data, train_labs, trainset.transform)
        
        # filter out the least 10% confident data
        filter_trainset = Subset(trainset, train_idx)
        print("Trainset size: " + str(len(filter_trainset)))

        trainloader =  DataLoader(filter_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        # initialize and train student model
        student = copy.deepcopy(teacher)
        optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-4)

        for i in range(1, epochs+1):
            train(i, trainloader, student, optimizer)
            if i % 5 == 0:
                 test(targetloader, student)
        print("------------Performance on the current domain----------")
        test(trainloader, student)

        # test on the target domain
        print("------------Performance on the target domain----------")
        st_loss, st_acc = test(targetloader, student)

        teacher = copy.deepcopy(student)

    if steps > 0:
        st_loss, st_acc = test(targetloader, teacher)

    return direct_acc, st_acc

def self_train_one_domain(args, teacher_model, trainset, targetset, epochs=10, source_idx=0, use_labels: bool = False):
    """
    Self-trains on a single domain and returns the updated student model (new teacher).
    
    Args:
        args: training args
        teacher_model: current model (used to generate pseudo-labels)
        trainset: unlabeled synthetic domain (DomainDataset)
        targetset: full target dataset (for evaluation)
        epochs: number of training epochs
    
    Returns:
        st_acc: final self-training accuracy on the target set
        student: updated model after self-training
    """
    print(f"--------Training on the {source_idx}th domain--------")
    # Evaluation loader for target
    targetloader = DataLoader(targetset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Get pseudo-labels on current synthetic domain
    ogloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    train_labs, train_idx = get_pseudo_labels(ogloader, teacher_model)
    if use_labels and hasattr(trainset, "targets"):
        train_labs = np.array(trainset.targets)

    # Convert trainset to EncodeDataset with pseudo-labels
    if torch.is_tensor(trainset.data):
        data = trainset.data.cpu().detach().numpy()
    else:
        data = trainset.data
    pseudo_trainset = EncodeDataset(data, train_labs, trainset.transform)

    # Filter by confidence
    confident_subset = Subset(pseudo_trainset, train_idx)
    print(f"Trainset size: {len(confident_subset)}")
    # breakpoint()
    trainloader = DataLoader(confident_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Train a new student from the current teacher
    student = copy.deepcopy(teacher_model)
    optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        train(epoch, trainloader, student, optimizer)
        if epoch % 5 == 0:
            test(trainloader, student)

    st_acc = test(targetloader, student)  # Final accuracy on target
    teacher = copy.deepcopy(student)

    return st_acc, teacher
