import torch
import random
import argparse
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import autoencoderML4Layer


# CLI parsing
parser = argparse.ArgumentParser(description='Lab 1 results')
parser.add_argument('-l', type=str, help='model weight')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model_weight = args.l

# initialize model
model = autoencoderML4Layer()
model.load_state_dict(torch.load(model_weight))
model.to('cpu')

# data preparation
# read MNIST dataset
train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = train_transform
test_set = MNIST('./data/mnist', train=False, download=True, transform=test_transform)
test_dataloader = DataLoader(test_set, batch_size=1024, shuffle=False)

test_loader = []
for i, batch in enumerate(test_dataloader):
    test_loader += batch[0]

def test(test_loader, model):
    model.eval()
    # plot before and after training
    with torch.inference_mode():
        f = plt.figure()
        f.add_subplot(1, 2, 1)
        plt.imshow(test_loader[0].reshape(28, 28), cmap='gray')
        f.add_subplot(1, 2, 2)
        plt.imshow(model(test_loader[0].reshape(1, 784)).reshape(28, 28), cmap='gray')
        plt.show()

# Image denoising


def noise(test_loader, model):
    model.eval()
    # plot before and after training, three sample images
    with torch.inference_mode():
        f = plt.figure()
        f.add_subplot(3, 3, 1)
        t_img = test_loader[random.randrange(len(test_loader))].reshape(28, 28)  # reshape the tensor to 28*28 size
        plt.imshow(t_img, cmap='gray')
        f.add_subplot(3, 3, 2)
        noise_img = t_img + 0.4 * torch.randn(28, 28)  # overlay noise on the image
        plt.imshow(noise_img, cmap='gray')
        f.add_subplot(3, 3, 3)
        denoise_img = model(noise_img.reshape(1, 784)).reshape(28, 28)  # remove image noise then reshape
        plt.imshow(denoise_img, cmap='gray')

        f.add_subplot(3, 3, 4)
        t_img1 = test_loader[random.randrange(len(test_loader))].reshape(28, 28)
        plt.imshow(t_img1, cmap='gray')
        f.add_subplot(3, 3, 5)
        noise_img1 = t_img1 + 0.4 * torch.randn(28, 28)
        plt.imshow(noise_img1, cmap='gray')
        f.add_subplot(3, 3, 6)
        denoise_img1 = model(noise_img1.reshape(1, 784)).reshape(28, 28)
        plt.imshow(denoise_img1, cmap='gray')

        f.add_subplot(3, 3, 7)
        t_img2 = test_loader[random.randrange(len(test_loader))].reshape(28, 28)
        plt.imshow(t_img2, cmap='gray')
        f.add_subplot(3, 3, 8)
        noise_img2 = t_img2 + 0.4 * torch.randn(28, 28)
        plt.imshow(noise_img2, cmap='gray')
        f.add_subplot(3, 3, 9)
        denoise_img2 = model(noise_img2.reshape(1, 784)).reshape(28, 28)
        plt.imshow(denoise_img2, cmap='gray')
        plt.show()


if __name__ == "__main__":
    test(test_loader, model)
    noise(test_loader, model)

    # plot before and after training, interpolate
    with torch.inference_mode():
        f = plt.figure(figsize=(16, 10))
        f.add_subplot(3, 10, 1)
        f_img1 = test_loader[random.randrange(len(test_loader))].reshape(28, 28)  # reshape the tensor to 28*28 size
        plt.imshow(f_img1, cmap='gray')
        f.add_subplot(3, 10, 10)
        f_img2 = test_loader[random.randrange(len(test_loader))].reshape(28, 28)
        plt.imshow(f_img2, cmap='gray')

        for i in range(2, 10):
            f.add_subplot(3, 10, i)
            f_img = f_img1.reshape(1, 784)
            _f_img = model.encode(f_img)
            b_img = f_img2.reshape(1, 784)
            _b_img = model.encode(b_img)
            _inter_img = 0.125 * (8-(i-1)) * _f_img + 0.125 * (i-1) * _b_img
            out_img = model.decode(_inter_img)
            plt.imshow(out_img.reshape(28, 28), cmap='gray')

        f.add_subplot(3, 10, 11)
        f_img3 = test_loader[random.randrange(len(test_loader))].reshape(28, 28)
        plt.imshow(f_img3, cmap='gray')
        f.add_subplot(3, 10, 20)
        f_img4 = test_loader[random.randrange(len(test_loader))].reshape(28, 28)
        plt.imshow(f_img4, cmap='gray')

        for i in range(12, 20):
            f.add_subplot(3, 10, i)
            f_img = f_img3.reshape(1, 784)
            _f_img = model.encode(f_img)
            b_img = f_img4.reshape(1, 784)
            _b_img = model.encode(b_img)
            _inter_img = 0.125 * (8-(i-11)) * _f_img + 0.125 * (i-11) * _b_img
            out_img = model.decode(_inter_img)
            plt.imshow(out_img.reshape(28, 28), cmap='gray')

        f.add_subplot(3, 10, 21)
        f_img5 = test_loader[random.randrange(len(test_loader))].reshape(28, 28)
        plt.imshow(f_img5, cmap='gray')
        f.add_subplot(3, 10, 30)
        f_img6 = test_loader[random.randrange(len(test_loader))].reshape(28, 28)
        plt.imshow(f_img6, cmap='gray')

        for i in range(22, 30):
            f.add_subplot(3, 10, i)
            f_img = f_img5.reshape(1, 784)
            _f_img = model.encode(f_img)
            b_img = f_img6.reshape(1, 784)
            _b_img = model.encode(b_img)
            _inter_img = 0.125 * (8-(i-21)) * _f_img + 0.125 * (i-21) * _b_img
            out_img = model.decode(_inter_img)
            plt.imshow(out_img.reshape(28, 28), cmap='gray')
        plt.show()