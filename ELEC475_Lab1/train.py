import datetime
import torch
import argparse
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch import nn
from torchsummary import summary
from torchinfo import summary
from torch.utils.data import DataLoader
from model import autoencoderML4Layer

# CLI parsing
parser = argparse.ArgumentParser(description='Train the MLP autoencoder.')
parser.add_argument('-z', type=int, help='bottleneck size')
parser.add_argument('-e', type=int, help='number of epochs')
parser.add_argument('-b', type=int, help='batch size')
parser.add_argument('-s', type=str, help='model weight path')
parser.add_argument('-p', type=str, help='loss plot')
args = parser.parse_args()


# train function
def train(n_epochs, optimizer, model, loss_fn, train_dataloader, scheduler, device, model_weight, plot_save_path):
    print('training...')
    model.to(device)
    model.train()
    losses_train = []
    print(n_epochs)
    for epoch in range(1, n_epochs + 1):
        print('epoch ', epoch)
        loss_train = 0.0
        for index, (img, target) in enumerate(train_dataloader):
            # print(img)
            img = img.to(device=device)  # use cpu or gpu
            # print(img.shape)
            img = img[0].reshape(1, 784)  # reshape the image to a 1x784 tensor
            output = model(img)  # forward propagation through network
            loss = loss_fn(output, img)  # calculate loss
            optimizer.zero_grad()  # reset optimization gradients to zero
            loss.backward()  # calculate loss gradients
            optimizer.step()  # iterate the optimization, based on loss gradients
            loss_train += loss.item()  # update value of losses

        scheduler.step()  # update optimization hyperparameters

        losses_train += [loss_train / len(train_dataloader)]  # update value of losses

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, loss_train / len(train_dataloader)))
    # save weight file as name given in commandline
    torch.save(model.state_dict(), model_weight)
    summary(model, (1, 28 * 28))

    # Loss curve
    plt.plot(losses_train)
    plt.savefig(plot_save_path)
    plt.show()


if __name__ == "__main__":
    # data preparation
    # read MNIST dataset

    # transforms.ToTensor() automatically scales the input data to the range of [0,1]
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = MNIST('./data/mnist', train=True, download=True,
                      transform=train_transform)
    train_dataloader = DataLoader(train_set, batch_size=args.b, shuffle=True)

    n_epochs = args.e
    model = autoencoderML4Layer(N_bottleneck=args.z)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    loss_fn = nn.MSELoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    model_weight = args.s
    plot_save_path = args.p

    train(n_epochs, optimizer, model, loss_fn, train_dataloader, scheduler, device, model_weight, plot_save_path)
