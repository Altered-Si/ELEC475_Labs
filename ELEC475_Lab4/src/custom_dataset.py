import os
import fnmatch
import torch
from torch.utils.data import Dataset
from PIL import Image
from numpy import array
import cv2

class KittiROIDataset(Dataset):
    def __init__(self, dir, training=True, transform=None):
        self.dir = dir
        self.training = training
        self.mode = 'train'
        if self.training == False:
            self.mode = 'test'
        self.img_dir = os.path.join(dir, self.mode)
        self.label_dir = os.path.join(dir, self.mode)
        self.transform = transform
        self.num = 0
        self.img_files = []
        self.labels = []
        for file in os.listdir(self.img_dir):
            if fnmatch.fnmatch(file, '*.png'):
                self.img_files += [file]

        label_path = os.path.join(self.label_dir, 'labels.txt')
        labels_string = None

        with open(label_path, 'r') as label_file:
            # reads all the lines in the file and stores them as a list of strings in labels_string
            labels_string = label_file.readlines()

            for i in range(len(labels_string)):
                lsplit = labels_string[i].split(
                    ' ')  # splits each line by spaces and stores the resulting list in lsplit
                label = int(lsplit[1])  # get the class label (second element) from the split line
                self.labels += [label]  # Appending this list to labels

        self.max = len(self)
        # print('break 12: ', self.img_dir)
        # print('break 12: ', self.label_dir)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # filename = os.path.splitext(self.img_files[idx])[0]
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # Convert numpy array to PIL image
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.transform(image)

        label = self.labels[idx]
        # convert list to array
        # labels = torch.tensor(labels)
        return image, label

    def __iter__(self):
        self.num = 0
        return self

    def __next__(self):
        if (self.num >= self.max):
            raise StopIteration
        else:
            self.num += 1
            return self.__getitem__(self.num-1)


    # class_label = {'NoCar': 0, 'Car': 1}

    def strip_ROIs(self, class_ID, label_list):
        ROIs = []
        for i in range(len(label_list)):
            ROI = label_list[i]
            if ROI[1] == class_ID:
                pt1 = (int(ROI[3]),int(ROI[2]))
                pt2 = (int(ROI[5]), int(ROI[4]))
                ROIs += [(pt1,pt2)]
        return ROIs
