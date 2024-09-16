import torch
from torch import optim
import argparse
import datetime
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from model import *
from custom_dataset import CustomDataset

parser = argparse.ArgumentParser()
parser.add_argument('-e', type=int, help='number of epochs')
parser.add_argument('-b', type=int, default=8, help='batch size')
parser.add_argument('-s', type=str, help='model weight')
parser.add_argument('-p', type=str, help='plot')
parser.add_argument('-m', type=str, help='[train/test]')
parser.add_argument('-d', metavar='display', type=str, help='display image [y/N]')
args = parser.parse_args()

# data preparation
data_dir = './data/images/'


def train_transform():
    transform_list = [
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


train_dataset = CustomDataset(data_dir, training=True, transform=train_transform())
test_dataset = CustomDataset(data_dir, training=False, transform=train_transform())

train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=True)


def euclidean_distance(predict, groud_truth):
    e_distance = torch.sqrt(torch.sum((predict - groud_truth) ** 2, dim=1))
    return e_distance


def train(n_epochs, criterion, model_weight, plot_save_path):
    print('training...')
    net.to(device)
    net.train()
    losses_train = []
    for epoch in range(1, n_epochs + 1):
        print('epoch', epoch)
        loss_train = 0.0
        total_distance_train = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            images, labels = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = net.forward(images)  # predicted nose central coordinate

            # loss = criterion(outputs, labels) ** 0.5  # criterion is MSE, used to make the loss be RMSE
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            dist = euclidean_distance(outputs, labels)
            total_distance_train += dist.sum().item()
        scheduler.step()

        losses_train += [loss_train / len(train_loader)]
        avg_distance_train = total_distance_train / len(train_dataset)

        torch.save(net.state_dict(), model_weight)

        print('{} Epoch {}, Loss: {:.6f}, Avg Distance: {:.4f}'.format(
            datetime.datetime.now(), epoch, loss_train / len(train_loader), avg_distance_train))
    # Loss curve
    plt.plot(losses_train)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(plot_save_path)
    plt.show()


def test(criterion, model_weight):
    print('{} testing...'.format(datetime.datetime.now()))
    net.load_state_dict(torch.load(model_weight))
    net.to(device)
    net.eval()
    with torch.no_grad():
        distances = []
        loss_test = 0

        for batch_idx, (inputs, targets) in enumerate(test_loader):
            images, labels = inputs.to(device), targets.to(device)

            outputs = net.forward(images)  # predicted nose central coordinate

            loss = criterion(outputs, labels)
            loss_test += loss.item()
            # calculate the Euclidean distance from predicted locations to ground truth nose locations
            dist = euclidean_distance(outputs, labels)
            distances.append(dist.item())

        # calculate the minimum Euclidean distance from predicted locations to ground truth nose locations
        min_dist_test = min(distances)
        # calculate the maximum Euclidean distance from predicted locations to ground truth nose locations
        max_dist_test = max(distances)
        # calculate the mean of Euclidean distance from predicted locations to ground truth nose locations
        avg_distance_test = sum(distances) / len(distances)
        # calculate the standard deviation of Euclidean distance from predicted locations to ground truth nose locations
        distances_tensor = torch.Tensor(distances)
        std_dev_test = distances_tensor.std()
        print('{} Loss: {:.6f}, Avg Dist: {:.4f}, Min Dist: {:.4f}, Max Dist: {:.4f}, Std Dev: {:.4f}'.format(
            datetime.datetime.now(),
            loss_test / len(test_dataset),
            avg_distance_test,
            min_dist_test,
            max_dist_test,
            std_dev_test))


if __name__ == "__main__":
    # parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net()

    criterion = nn.SmoothL1Loss()
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    n_epochs = args.e
    batch_size = args.b
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)
    model_weight = args.s
    plot_save_path = args.p
    mode = args.m
    show_images = True if args.d == 'y' or args.d == 'Y' else False

    if mode == 'train':
        train(n_epochs, criterion, model_weight, plot_save_path)
    else:
        test(criterion, model_weight)
