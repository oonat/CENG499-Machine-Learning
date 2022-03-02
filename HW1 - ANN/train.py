import torch
import torch.nn as nn
from tqdm import tqdm



def train(model, device, train_loader, val_loader, num_epochs, learning_rate, fpath):

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = 10000
    best_epoch = -1
    train_loss_list = []
    val_loss_list = []

    for epoch in tqdm(range(num_epochs)):
        # Training
        model.train()
        accum_train_loss = 0
        for i, (imgs, labels) in enumerate(train_loader, start=1):
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            loss = loss_function(output, labels)

            # accumulate the loss
            accum_train_loss += loss.item()

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        accum_val_loss = 0
        with torch.no_grad():
            for j, (imgs, labels) in enumerate(val_loader, start=1):
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
                accum_val_loss += loss_function(output, labels).item()


        train_loss_list.append(accum_train_loss / i) 
        val_loss_list.append(accum_val_loss / j)

        if((accum_val_loss / j) < best_val_loss):
            torch.save(model.state_dict(), fpath + "_BEST")
            best_val_loss = (accum_val_loss / j)
            best_epoch = epoch


    return train_loss_list, val_loss_list, best_val_loss, best_epoch



def validation_test(model, device, val_loader):

    # Compute Validation Accuracy
    model.eval()
    with torch.no_grad():
        correct = total = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            
            _, predicted_labels = torch.max(output, 1)
            correct += (predicted_labels == labels).sum()
            total += labels.size(0)

    return (100 * correct/total)




def test(model, device, test_loader):

    # Compute Test Accuracy
    model.eval()
    with torch.no_grad():
        correct = total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            
            _, predicted_labels = torch.max(output, 1)
            correct += (predicted_labels == labels).sum()
            total += labels.size(0)

    print(f'Test Accuracy = {100 * correct/total :.3f}%')

