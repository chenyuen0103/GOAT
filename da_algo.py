import torch
from torch.utils.data import DataLoader, random_split, Subset
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

def get_pseudo_labels(dataloader, model, confidence_q=0.1):
    logits = []
    model.eval()
    with torch.no_grad():
        for x in dataloader:
            if len(x) == 3:
                data, _, _ = x
            else:
                data, _ = x
            data = data.to(device)
            logits.append(model(data))
    
    logits = torch.cat(logits)
    confidence = torch.max(logits, dim=1)[0] - torch.min(logits, dim=1)[0]
    alpha = torch.quantile(confidence, confidence_q)
    indices = torch.where(confidence >= alpha)[0].to("cpu")
    labels = torch.argmax(logits, axis=1) #[indices]
    # breakpoint()
    return labels.cpu().detach().type(torch.int64), list(indices.detach().numpy())

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
    steps = len(datasets)
    teacher = source_model
    targetset = datasets[-1]
        
    targetloader = DataLoader(targetset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("------------Direct adapt performance----------")
    direct_acc = test(targetloader, teacher)

    # start self-training on intermediate domains
    for i in range(steps):
        print(f"--------Training on the {i}th domain--------")
        trainset = datasets[i]
        ogloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        train_labs, train_idx = get_pseudo_labels(ogloader, teacher)



        # Filter labels and indices if needed
        if use_labels and i < steps - 1:
            if hasattr(trainset, "targets"):
                train_labs = np.array(trainset.targets)
                # train_targets = np.array(trainset.targets)
                # train_idx = np.array(train_idx)
                # train_labs_full = np.array(train_labs)[train_idx]
                # transported_subset = train_targets[train_idx]

                # match_mask = transported_subset == train_labs_full
                # train_idx = train_idx[match_mask]
                # train_labs = train_labs_full[match_mask]
            else:
                raise ValueError("Transported labels not available in the dataset.")
        else:
            train_idx = np.array(train_idx)
            train_labs = np.array(train_labs)[train_idx]

        # Apply filtering to data
        if torch.is_tensor(trainset.data):
            data = trainset.data.cpu().detach().numpy()
        else:
            data = trainset.data

        data = data[train_idx]  # ⬅️ aligned with filtered labels

        trainset = EncodeDataset(data, train_labs, trainset.transform)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        # initialize and train student model
        student = copy.deepcopy(teacher)
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
        st_acc = test(targetloader, student)

        teacher = copy.deepcopy(student)
    
    return direct_acc, st_acc


def self_train_og(args, source_model, datasets, epochs=10, use_labels=False):
    steps = len(datasets)
    teacher = source_model
    targetset = datasets[-1]
        
    targetloader = DataLoader(targetset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("------------Direct adapt performance----------")
    direct_acc = test(targetloader, teacher)

    # start self-training on intermediate domains
    for i in range(steps):
        print(f"--------Training on the {i}th domain--------")
        trainset = datasets[i]
        ogloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                
        test(targetloader, teacher)
        breakpoint()
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
        breakpoint()

        # test on the target domain
        print("------------Performance on the target domain----------")
        st_acc = test(targetloader, student)

        teacher = copy.deepcopy(student)
    
    return direct_acc, st_acc

def self_train_one_domain(args, teacher_model, trainset, targetset, epochs=10, source_idx=0):
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
    if hasattr(trainset, "targets"):
        # breakpoint()
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
