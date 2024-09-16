import torch
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
import model_vanilla as model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR100 Testing')
    parser.add_argument('-encoder_file', type=str, help='encoder weight file')
    parser.add_argument('-decoder_file', type=str, help='decoder weight file')
    parser.add_argument('-b', type=int, default=16, help='batch size')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    # Data preparation
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)
    testloader = DataLoader(testset, batch_size=args.b, shuffle=False, num_workers=2)

    decoder_file = args.decoder_file
    encoder_file = args.encoder_file
    features = model.encoder_decoder.encoder
    features.load_state_dict(torch.load(encoder_file, map_location='cpu'))
    classifier = model.encoder_decoder.decoder
    classifier.load_state_dict(torch.load(decoder_file, map_location='cpu'))

    net = model.Vanilla(features, classifier)
    # test
    print('testing...')
    net.to(device)
    net.eval()

    correct_top1 = 0
    correct_top5 = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            images, labels = inputs.to(device), targets.to(device)

            outputs = net(images)
            _, predicted = outputs.topk(k=5, dim=1, largest=True, sorted=True)

            labels = labels.view(labels.size(0), -1).expand_as(predicted)
            correct = predicted.eq(labels).float()

            # compute top 5
            correct_top5 += correct[:, :5].sum().item()

            # compute top 1
            correct_top1 += correct[:, :1].sum().item()
    top_1_error = 1 - correct_top1 / len(testset)
    top_5_error = 1 - correct_top5 / len(testset)
    print('Top 1 Error: {:.3%}, Top 5 Error: {:.3%}'.format(top_1_error, top_5_error))