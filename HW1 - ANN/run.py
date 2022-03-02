import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader

from model import ANN
from train import train, test, validation_test
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import os


def train_test_split():

    train_transform = T.Compose([
        # can add additional transforms on images
        T.ToTensor (), # convert images to PyTorch tensors
        T.Grayscale (), # RGB to grayscale
        T.Normalize(mean=(0.5,), std=(0.5,)) # normalization
                # speeds up the convergence
                # and improves the accuracy
    ])

    val_transform = test_transform = T.Compose([
        T.ToTensor (),
        T.Grayscale (),
        T.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_set = CIFAR10(root='CIFAR10', train=True ,
                        transform=train_transform , download=True)
                        
    test_set = CIFAR10(root='CIFAR10', train=False ,
                        transform=test_transform , download=True)


    train_set_size = int(0.8 * len(train_set))
    val_set_size = len(train_set) - train_set_size
    train_set, val_set = random_split(train_set, [train_set_size, val_set_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)

    return train_loader, val_loader, test_loader



def plot_graph(train_loss_list, val_loss_list, num_epochs, path):

    epochs = range(1, num_epochs + 1)

    fig, ax = plt.subplots()

    ax.plot(epochs, train_loss_list)
    ax.plot(epochs, val_loss_list)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.legend(["Training loss", "Validation loss"])
    fig.savefig(path + "_PLOT.png")
    plt.close(fig)



if __name__ == "__main__":

    torch.manual_seed(1234)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_epochs = 30

    # hyperparameter settings
    layer_list =  [[32 * 32, 10],
        [32 * 32, 512, 10],
        [32 * 32, 1024, 10],
        [32 * 32, 512, 256, 10],
        [32 * 32, 1024, 512, 10]]
    learning_rate_list = [0.0001, 0.001, 0.01]
    activation_function_list = ["relu", "sigmoid", "tanh"]

    train_loader, val_loader, test_loader = train_test_split()


    saving_dir = os.path.join(os.getcwd(), r'savings')
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)


    for layer_features in layer_list:
        layer_num = len(layer_features) - 1
        neuron_num = 0 if layer_num == 1 else layer_features[1]
        for lr in learning_rate_list:
            for act_func in activation_function_list:
                model = ANN(layer_num, layer_features, act_func).to(device)
                fpath = "savings/" + str(layer_num) + "_" + str(neuron_num) + "_" + str(lr) + "_" + act_func
                train_loss_list, val_loss_list, best_val_loss, best_epoch = train(model, device, train_loader, val_loader, num_epochs, lr, fpath)
                
                model.load_state_dict(torch.load(fpath + "_BEST"))
                validation_acc = validation_test(model, device, val_loader)

                print(f'Layer_num = {layer_num} Neuron_num = {neuron_num} | Lr = {lr} | Function = {act_func} | Val_loss = {best_val_loss:.4f} Val_acc = {validation_acc:.4f} Epoch = {best_epoch}')
                plot_graph(train_loss_list, val_loss_list, num_epochs, fpath)


    """

    model = ANN(2, [32 * 32, 1024, 10], "relu").to(device)
    model.load_state_dict(torch.load("savings/2_1024_0.0001_relu_BEST"))
    test(model, device, test_loader)

    """