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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    return labels.cpu().detach().type(torch.int64), list(indices.detach().numpy())


def self_train(args, source_model, datasets, domain_indices, domain_types, epochs=10, log_file="log.csv"):
    steps = len(datasets)
    teacher = source_model
    targetset = datasets[-1]

    log_header = ["Domain Index", "Domain Type", "Epoch", "Train Set Size", 
                "Train Loss", "Train Acc", "Target Loss", "Target Acc"]
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(log_header)
        
    targetloader = DataLoader(targetset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("------------Direct adapt performance----------")
    # Log direct adaptation performance (before training starts)
    target_loss, target_acc = test(targetloader, teacher)

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([0, "source", 0, len(datasets[0]), "-", "-", round(target_loss, 4), round(target_acc, 4)])


    # start self-training on intermediate domains
    for i in range(steps):
        domain_idx = domain_indices[i]
        domain_type = domain_types[i]
        print(f"--------Training on domain {domain_idx} ({domain_type}) --------")

        print(f"--------Training on the {i}th domain--------")
        trainset = datasets[i]
        ogloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                
        test(targetloader, teacher)
        train_labs, train_idx = get_pseudo_labels(ogloader, teacher)

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
            
            if i % 5 == 0:
                train_loss = train(i, trainloader, student, optimizer)
                _, train_acc = test(trainloader, student)  # Accuracy on current domain
                target_loss, target_acc = test(targetloader, student)  # Accuracy & loss on target domain
                # Append log to CSV
                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([domain_idx, domain_type, i, len(filter_trainset), 
                                    round(train_loss, 4), round(train_acc, 4), 
                                    round(target_loss, 4), round(target_acc, 4)])


        print("------------Performance on the current domain----------")
        test(trainloader, student)

        # test on the target domain
        print("------------Performance on the target domain----------")
        st_acc = test(targetloader, student)

        teacher = copy.deepcopy(student)
    
    return direct_acc, st_acc



