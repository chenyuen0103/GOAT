import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    BCE = F.binary_cross_entropy(recon_x, x.view(recon_x.shape), reduction='sum')

    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def calculate_modal_val_accuracy(model, valloader):
    model.eval()
    correct = 0.
    total = 0.

    with torch.no_grad():
        for x in valloader:
            if len(x) == 3:
                images, labels, weight = x
            else:
                images, labels = x

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    return 100 * correct / total



def weighted_val_accuracy(model, valloader):
    correct = 0.
    total = 0.

    for (images, labels, weight) in valloader:

        images, labels, weight = images.to(device), labels.to(device), weight.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    return 100*correct/total

def train(epoch, train_loader, model, optimizer, lr_scheduler=None, vae=False, verbose=True):
    model.train()
    train_loss = 0
    for _, x in enumerate(train_loader):
        if len(x) == 2:
            data, labels = x
        elif len(x) == 3:
            data, labels, weight = x
            weight = weight.to(device)

        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        if vae:
            recon_batch, mu, log_var = model(data)
            loss = loss_function(recon_batch, data, mu, log_var)
        else:
            output = model(data)
            if labels.min() < 0 or labels.max() >= output.shape[1]:
                breakpoint()
                print(f"[DEBUG] Invalid label detected. Label range: [{labels.min().item()}, {labels.max().item()}], Expected: [0, {output.shape[1] - 1}]")
                print(f"[DEBUG] Sample invalid labels: {labels[:10]}")
                print(f"[DEBUG] Output shape: {output.shape}")
                import sys
                sys.exit(1)  # or raise an Exception

            if len(x) == 2:
                loss = F.cross_entropy(output, labels.long())
            elif len(x) == 3:
                criterion = nn.CrossEntropyLoss(reduction='none')
                loss = criterion(output, labels)
                loss = (loss * weight).mean()

        loss.backward()
        try:
            train_loss += loss.item()
        except:
            breakpoint()
            print(f"[DEBUG] loss = {loss}")
            print(f"[DEBUG] loss.item() = {loss.item()}")
            print(f"[DEBUG] loss.shape = {loss.shape}")
            print(f"[DEBUG] loss.mean() = {loss.mean()}")
            print(f" loss.sum() = {loss.sum()}")

        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

    if verbose:
        print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss / len(train_loader.dataset)))
    return train_loss


def test(val_loader, model, vae=False, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0.
    total = 0.

    with torch.no_grad():
        for x in val_loader:
            if len(x) == 2:
                data, labels = x
            elif len(x) == 3:
                data, labels, weight = x
                weight = weight.to(device)
            data = data.to(device)
            labels = labels.to(device)

            if vae:
                recon, mu, log_var = model(data)
                test_loss += loss_function(recon, data, mu, log_var).item()
            else:
                output = model(data)
                if len(x) == 2:
                    criterion = nn.CrossEntropyLoss()
                    test_loss += criterion(output, labels).item()
                elif len(x) == 3:
                    criterion = nn.CrossEntropyLoss(reduction='none')

                    loss = criterion(output, labels.long())
                    test_loss += (loss * weight).mean().item()

                predicted = output.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

    test_loss /= len(val_loader.dataset)
    val_accuracy = 100 * correct / total
    val_accuracy = val_accuracy.item()
    if verbose:
        print('====> Test loss: {:.8f}'.format(test_loss))
        if not vae:
            print('====> Test Accuracy %.4f' % (val_accuracy))

    return test_loss, val_accuracy



def weighted_train(epoch, train_loader, model, optimizer, verbose=True):
    model.train()
    train_loss = 0
    for _, (data, labels, weight) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        weight = weight.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(output, labels)
        loss = (loss * weight).mean()

        # criterion = nn.CrossEntropyLoss()
        # loss = criterion(output, labels)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    if verbose:
        print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss / len(train_loader.dataset)))


def val(val_loader, model, vae=False, verbose=True):
    model.eval()
    test_loss= 0
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device)
            labels = labels.to(device)

            if vae:
                recon, mu, log_var = model(data)
                test_loss += loss_function(recon, data, mu, log_var).item()
                # output = model(data)
                # criterion = nn.MSELoss()
                # test_loss += criterion(output, data)
            else:
                output = model(data)
                criterion = nn.CrossEntropyLoss()
                test_loss += criterion(output, labels).item()
        
    test_loss /= len(val_loader.dataset)

    if verbose:
        print('====> Validation loss: {:.8f}'.format(test_loss))

        if not vae:
            val_accuracy = calculate_modal_val_accuracy(model, val_loader)
            print('====> Validation Accuracy %.4f' % (val_accuracy))


def weighted_val(val_loader, model, vae=False, verbose=True):
    model.eval()
    test_loss= 0
    for _, (data, labels, weight) in enumerate(val_loader):
        data = data.to(device)
        labels = labels.to(device)
        weight = weight.to(device)
        
        output = model(data)
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(output, labels)
        test_loss += (loss * weight).mean().item()
        
    test_loss /= len(val_loader.dataset)

    if verbose:
        print('====> Validation loss: {:.8f}'.format(test_loss))

        if not vae:
            val_accuracy = weighted_val_accuracy(model, val_loader)
            print('====> Validation Accuracy %.4f' % (val_accuracy))




# def test(test_loader, model, vae=False, verbose=True):
#     model.eval()
#     test_loss= 0
#     with torch.no_grad():
#         for data, labels in test_loader:
#             data = data.to(device)
#             labels = labels.to(device)

#             if vae:
#                 recon, mu, log_var = model(data)
#                 test_loss += loss_function(recon, data, mu, log_var).item()
#                 # output = model(data)
#                 # criterion = nn.MSELoss()
#                 # test_loss += criterion(output, data)
#             else:
#                 output = model(data)
#                 criterion = nn.CrossEntropyLoss()
#                 test_loss += criterion(output, labels).item()
        
#     test_loss /= len(test_loader.dataset)

#     if verbose:
#         print('====> Test set loss: {:.4f}'.format(test_loss))

#         if not vae:
#             test_accuracy = calculate_modal_val_accuracy(model, test_loader)
#             print('====> Test Accuracy %.4f' % (test_accuracy))
        
#     if not vae:
#         test_accuracy = calculate_modal_val_accuracy(model, test_loader)
#         return test_accuracy

