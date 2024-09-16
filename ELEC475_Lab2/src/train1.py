import argparse
import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import custom_dataset

import AdaIN_net as net


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    print(iteration_count)
    learning_rate = lr / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def train(n_epochs, n_batches, optimizer, model, content_dataset, style_dataset, device):
    print('training ...')
    model.to(device=device)
    model.train()
    losses_train = []
    losses_c_train = []
    losses_s_train = []
    for epoch in range(n_epochs):
        content_dataloader = DataLoader(content_dataset, batch_size=n_batches, shuffle=True)
        style_dataloader = DataLoader(style_dataset, batch_size=n_batches, shuffle=True)
        loss_train = 0.0
        loss_c_train = 0.0
        loss_s_train = 0.0
        for batch in range(n_batches):
            iter_count = epoch * n_batches + batch
            adjust_learning_rate(optimizer, iteration_count=iter_count)
            content_images = next(iter(content_dataloader)).to(device)
            style_images = next(iter(style_dataloader)).to(device)
            loss_c, loss_s = model(content_images, style_images, alpha=gamma)
            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print('{} Batch {}, loss {}, loss_c {}, loss_s {}'.format(
            #     datetime.datetime.now(),
            #     batch,
            #     loss.item() / 20,
            #     loss_c.item() / 20,
            #     loss_s.item() / 20
            # ))

            loss_train += loss.item()
            loss_c_train += loss_c.item()
            loss_s_train += loss_s.item()

        losses_train += [loss_train / (n_batches * 20)]
        losses_c_train += [loss_c_train / (n_batches * 20)]
        losses_s_train += [loss_s_train / (n_batches * 20)]
        print('{} Epoch {}\n loss {}\n loss_c {}\n loss_s {}'.format(
            datetime.datetime.now(),
            epoch,
            loss_train / (n_batches * 20),
            loss_c_train / (n_batches * 20),
            loss_s_train / (n_batches * 20)
        ))

    torch.save(model.decoder.state_dict(), save_file)

    plt.plot(range(n_epochs), losses_train, label='content+style')
    plt.plot(range(n_epochs), losses_c_train, label='content')
    plt.plot(range(n_epochs), losses_s_train, label='style')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(save_plot)


if __name__ == "__main__":
    print(torch.cuda.is_available())

    parser = argparse.ArgumentParser()
    parser.add_argument('-content_dir', type=str, required=True, help='Directory path to a batch of content images')
    parser.add_argument('-style_dir', type=str, required=True, help='Directory path to a batch of style images')
    parser.add_argument('-gamma', type=float, default=1.0, help='gamma')
    parser.add_argument('-e', type=int, default=20, help='number of epoch')
    parser.add_argument('-b', type=int, default=32, help='batch size')
    parser.add_argument('-l', type=str, default='./encoder.pth', help='Directory to load the encoder weight')
    parser.add_argument('-s', type=str, default='./decoder.pth', help='Directory to save the decoder weight')
    parser.add_argument('-p', type=str, default='./decoder.png', help='Directory to save the loss plot')
    parser.add_argument('-cuda', type=str, default='N', help='[y/N]')

    args = parser.parse_args()

    lr = 1e-4
    lr_decay = 5e-5
    gamma = args.gamma
    n_epochs = args.e
    n_batches = args.b
    load_file = args.l
    save_file = args.s
    save_plot = args.p
    device = torch.device('cuda' if args.cuda == 'y' or args.cuda == 'Y' else 'cpu')

    encoder = net.encoder_decoder.encoder
    decoder = net.encoder_decoder.decoder

    encoder.load_state_dict(torch.load(load_file))
    encoder = nn.Sequential(*list(encoder.children())[:31])
    model = net.AdaIN_net(encoder, decoder)

    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = custom_dataset.custom_dataset(args.content_dir, content_tf)
    style_dataset = custom_dataset.custom_dataset(args.style_dir, style_tf)

    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=lr)

    train(n_epochs, n_batches, optimizer, model, content_dataset, style_dataset, device)
