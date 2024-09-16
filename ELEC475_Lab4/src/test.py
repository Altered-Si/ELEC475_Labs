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

parser = argparse.ArgumentParser(description='YODA classifier')
parser.add_argument('-b', type=int, default=8, help='batch size')
parser.add_argument('-s', type=str, help='decoder weight')

args = parser.parse_args()

# data preparation
data_dir = './data/Kitti8_ROIs/'

def test_transform():
    transform_list = [
        transforms.Resize(size=(150, 150)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

test_dataset = KittiROIDataset(data_dir, training=False, transform=test_transform())
test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=False)


def test(decoder_weight):
    print('testing...')
    net.resnet.load_state_dict(torch.load(decoder_weight, map_location='cpu'))
    net.to(device)
    net.eval()

    correct_top1 = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            images, labels = inputs.to(device), targets.to(device)

            outputs = net(images)
            _, predicted = outputs.topk(k=2, dim=1, largest=True, sorted=True)

            labels = labels.view(labels.size(0), -1).expand_as(predicted)
            correct = predicted.eq(labels).float()

            # compute top 1
            correct_top1 += correct[:, :1].sum().item()
    top_1_error = 1 - correct_top1 / len(test_dataset)
    # top_5_error = 1 - correct_top5 / len(test_dataset)
    print('Top 1 Error: {:.3%}'.format(top_1_error))


if __name__ == "__main__":
    # parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = model.Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-2, weight_decay=5e-4)
    batch_size = args.b
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)
    decoder_weight = args.s

    test(decoder_weight)

