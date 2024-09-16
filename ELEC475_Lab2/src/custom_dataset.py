import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile

class custom_dataset(Dataset):
    def __init__(self, dir, transform=None):
        super().__init__()
        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # self.transform = transforms.Compose([
        #     transforms.Resize(size=(size,size), interpolation=Image.BICUBIC),
        #     # transforms.RandomCrop(crop_size),
        #     transforms.ToTensor()
        #     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])
        self.transform = transform
        self.image_files = [dir + file_name for file_name in os.listdir(dir)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = Image.open(self.image_files[index]).convert('RGB')
        image_sample = self.transform(image)
        # print('break 27: ', index, image, image_sample.shape)
        return image_sample
