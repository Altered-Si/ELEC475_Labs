import torch
from torch import nn
from torch import optim
import argparse
import datetime
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import model
from custom_dataset import KittiROIDataset

parser = argparse.ArgumentParser(description='CIFAR100 Training')
parser.add_argument('-e', type=int, help='number of epochs')
parser.add_argument('-b', type=int, default=8, help='batch size')
parser.add_argument('-s', type=str, help='decoder weight')
parser.add_argument('-p', type=str, help='plot')
# parser.add_argument('-m', metavar='mode', type=str, help='[train/test]')
args = parser.parse_args()

# data preparation
data_dir = './data/Kitti8_ROIs/'

def train_transform():
    transform_list = [
        transforms.Resize(size=(150, 150)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


train_dataset = KittiROIDataset(data_dir, training=True, transform=train_transform())
test_dataset = KittiROIDataset(data_dir, training=False, transform=train_transform())

train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=False)

def train(n_epochs, decoder_weight, plot_save_path):
    print('training...')
    net.to(device)
    net.train()
    losses_train = []
    for epoch in range(1, n_epochs + 1):
        print('epoch', epoch)

        loss_train = 0.0
        correct_top1 = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            images, labels = inputs.to(device), targets.to(device)
            # print('batch_idx: ', batch_idx)
            # print('image: ', images)
            # print('labels: ', labels)
            # print('image.shape', images.shape)
            optimizer.zero_grad()
            outputs = net.forward(images)
            # print(outputs.shape)
            # print(labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)  # batch size, total would be len(trainloader)
            correct_top1 += predicted.eq(labels).sum().item()  # if highest prediction matches ground truth, award once

            # if batch_idx % args.b == 0:
            #     print('{} batch_idx: {}, Loss: {}, Accuracy: {:.3%}'.format(
            #         datetime.datetime.now(), batch_idx, loss_train / (batch_idx + 1), correct_top1 / total))
        scheduler.step()

        losses_train += [loss_train / (batch_idx+1)]
        torch.save(net.resnet.state_dict(), model_weight)

        print('{} Epoch {}, Loss: {}, Top 1 Accuracy: {:.3%}'.format(
            datetime.datetime.now(), epoch, loss_train / (batch_idx+1), correct_top1/total))
    # Loss curve
    plt.plot(losses_train)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(plot_save_path)
    plt.show()


if __name__ == "__main__":
    # parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = model.Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-2, weight_decay=5e-4)
    n_epochs = args.e
    batch_size = args.b
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)
    model_weight = args.s
    plot_save_path = args.p

    train(n_epochs, model_weight, plot_save_path)
    # test(decoder_weight)

