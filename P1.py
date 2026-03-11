import torch.utils.data as data
from torchvision.io import read_image
import torchvision.transforms as transforms
import glob
import os
import pandas as pd

class PortDataset(data.Dataset):

    def __init__(self, image_path, csv_path, transform = None):
        super().__init__()
        # Load all the filenames with extension jpg from the image_path directory
        self.img_files = glob.glob(os.path.join(image_path, '*.jpg'))
        self.mask_files = []

        self.df = pd.read_csv(csv_path)

        # Load the filenames of the masks (it is assumed that each mask
        # has the same name as the corresponding image).
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(csv_path,os.path.basename(img_path)))

        # DATA AUGMENTATION --> REVISAR AJUSTAR TAMAÑOS
        if transform:
          self.transform = transform
        else:
          self.transform = transforms.Compose([
                                                transforms.ToPILImage(),
                                                transforms.ToTensor()])


    # Returns the n-th image with its corresponding mask and image name
    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]

        # Get label from imagename
        name = os.path.splitext(os.path.basename(img_path))[0]

        # Get image and mask
        image = read_image(img_path)
        image = self.transform(image)

        mask = read_image(mask_path)
        mask = self.transform(mask)

        return image, mask, name

    def __len__(self):
        return len(self.img_files)