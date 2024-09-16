import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import re


class CustomDataset(Dataset):
    def __init__(self, root_dir, training=True, transform=None):
        self.root_dir = root_dir
        self.training = training
        self.mode = 'train'
        if not self.training:
            self.mode = 'test'
        self.transform = transform
        self.data, self.labels = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')

        # img.size[0] is width of the img in pixel, and img.size[1] is the height of the img in pixel
        # scaled aligning with the image transform (resize)
        label_scaled = torch.Tensor([label[0].item() * 224 / img.size[0], label[1].item() * 224 / img.size[1]])
        if self.transform:
            img = self.transform(img)
        
        return img, label_scaled

    def load_data(self):
        data = []
        labels = []

        if self.mode == 'train':
            label_path = '../train_noses.3.txt'  # label file for training
        else:
            label_path = '../test_noses.txt'  # label file for testing

        # read the coordinate of the nose
        with open(os.path.join(self.root_dir, label_path), 'r') as label_file:
            lines = label_file.readlines()
            for line in lines:
                line = re.findall(r'[^,"\s()]+', line)
                image_name = line[0]  # line[0] is the image file name
                image_path = os.path.join(self.root_dir, image_name)
                x, y = int(line[1]), int(line[2])

                nose_label = torch.Tensor([x, y])

                data.append(image_path)
                labels.append(nose_label)

        return data, labels
