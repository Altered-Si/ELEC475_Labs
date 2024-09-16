import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
import datetime
from torch.utils.data import DataLoader
import model_mod as model


parser = argparse.ArgumentParser(description='CIFAR100 Training')
parser.add_argument('-e', type=int, help='number of epochs')
parser.add_argument('-b', type=int, default=8, help='batch size')
parser.add_argument('-l', type=str, help='encoder weight')
parser.add_argument('-s', type=str, help='decoder weight')
parser.add_argument('-p', type=str, help='plot')
args = parser.parse_args()

# Data preparation
transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=args.b, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size=args.b, shuffle=False, num_workers=4)


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

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            images, labels = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net.forward(images)

            print(labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)  # batch size, total would be len(trainloader)
            correct_top1 += predicted.eq(labels).sum().item()  # if highest prediction matches ground truth, award once

        scheduler.step()

        losses_train += [loss_train / (batch_idx+1)]
        # torch.save(net.features.state_dict(), encoder_weight)
        torch.save(net.classifier.state_dict(), decoder_weight)

        print('{} Epoch {}, Loss: {}, Top 1 Accuracy: {:.3%}'.format(
            datetime.datetime.now(), epoch, loss_train / (batch_idx+1), correct_top1/total))
    # Loss curve
    plt.plot(losses_train)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(plot_save_path)
    plt.show()


if __name__ == "__main__":
    classifier = model.encoder_decoder.decoder
    features = model.encoder_decoder.encoder

    features.load_state_dict(torch.load(args.l))

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    net = model.BottomNet(features, classifier)
    criterion = nn.CrossEntropyLoss()
    n_epochs = args.e
    batch_size = args.b
    optimizer = torch.optim.SGD(params=net.parameters(), lr=1e-2, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)
    decoder_weight = args.s
    plot_save_path = args.p

    train(n_epochs, decoder_weight, plot_save_path)

