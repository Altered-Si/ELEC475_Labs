import datetime
import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataset import custom_dataset

import AdaIN_net as net

parser = argparse.ArgumentParser()
# training options
parser.add_argument('-content_dir', type=str, required=True,
                    help='Directory to content images')
parser.add_argument('-style_dir', type=str, required=True,
                    help='Directory to style images')
parser.add_argument('-gamma', type=float, default=1.0, help='gamma')
parser.add_argument('-e', type=int, help='number of epochs')
parser.add_argument('-b', type=int, help='batch size')
parser.add_argument('-l', type=str, help='encoder (vgg) weight')
parser.add_argument('-s', type=str, help='decoder weight')
parser.add_argument('-p', type=str, help='plot')
args = parser.parse_args()


# resize images size to 512*512, then randomly crop regions of 256*256
def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


# train function
def train(_n_epochs, _batch_size, _optimizer, _model, _gamma, _scheduler, _content_dataloader, _style_dataloader,
          _device, _decoder_weight, _plot_save_path):
    print('training...')
    model.to(_device)
    model.train()
    losses_train_content = []
    losses_train_style = []
    losses_train = []

    n_batches = len(_content_dataloader.dataset) // _batch_size
    print('n_batch per epoch = ' + str(n_batches))
    for epoch in range(1, _n_epochs + 1):
        print('epoch', epoch)
        loss_train_content = 0.0
        loss_train_style = 0.0
        loss_train = 0.0
        iter_count = 0

        for batch in range(n_batches):
            content_images = next(iter(_content_dataloader)).to(_device)
            print(len(content_images))
            style_images = next(iter(_style_dataloader)).to(_device)
            loss_c, loss_s = _model(content_images, style_images)
            loss = loss_c + _gamma * loss_s  # implement equation 11

            _optimizer.zero_grad()  # reset optimization gradients to zero
            loss.backward()  # calculate loss gradients
            _optimizer.step()  # iterate the optimization, based on loss gradients

            # update value of losses
            loss_train_content += loss_c.item()
            loss_train_style += loss_s.item()
            loss_train += loss.item()

        _scheduler.step()

        losses_train_content += [loss_train_content / n_batches]
        losses_train_style += [loss_train_style / n_batches]
        losses_train += [loss_train / n_batches]

        print('{} Epoch {}, Training loss {}, Content loss {}, Style loss {}'.format(
            datetime.datetime.now(), epoch, loss_train / n_batches, loss_train_content / n_batches,
            loss_train_style / n_batches))

        torch.save(_model.decoder.state_dict(), _decoder_weight)

    # Loss curve
    plt.plot(losses_train)
    plt.plot(losses_train_content)
    plt.plot(losses_train_style)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(['content+style', 'content', 'style'], loc="upper right")
    plt.savefig(_plot_save_path)
    plt.show()


if __name__ == "__main__":
    decoder = net.encoder_decoder.decoder
    encoder = net.encoder_decoder.encoder

    encoder.load_state_dict(torch.load(args.l))
    encoder = nn.Sequential(*list(encoder.children())[:31])
    network = net.AdaIN_net(encoder, decoder)

    # data preparation
    content_dataset = custom_dataset(args.content_dir, train_transform())
    style_dataset = custom_dataset(args.style_dir, train_transform())

    content_dataloader = DataLoader(content_dataset, batch_size=args.b, shuffle=True)
    style_dataloader = DataLoader(style_dataset, batch_size=args.b, shuffle=True)

    # training parameters
    n_epochs = args.e
    batch_size = args.b
    optimizer = torch.optim.Adam(params=net.encoder_decoder.decoder.parameters(), lr=1e-4)
    model = network
    gamma = args.gamma  # style loss weight
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    encoder_weight = args.l
    decoder_weight = args.s
    plot_save_path = args.p

    train(n_epochs, batch_size, optimizer, model, gamma, scheduler, content_dataloader, style_dataloader, device,
          decoder_weight, plot_save_path)
    
