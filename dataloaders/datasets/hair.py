import os
import numpy as np
import cv2
from torchvision import transforms
from dataloaders import custom_transforms as tr
from torch.utils.data import Dataset
from PIL import Image


class HairDataset(Dataset):

    NUM_CLASSES = 21
    def __init__(self, args, split):
        self.img_dir  = args.img_dir
        self.anno_dir = args.anno_dir
        self.split    = split
        with open(split + '.txt') as f:
            self.filenames = f.read().splitlines()

    def __len__(self):
        return len(self.filenames)

    def visualize(self, idx, percent=50):
        sample = self.__getitem__(idx)
        img = sample['image']
        ano = sample['label']
        ano[ano == 0] = 255
        ano[ano == 1] = percent
        img = cv2.resize(img, ano.shape)
        ano = np.expand_dims(ano, axis=-1)
        merge = np.concatenate([img, ano], axis=-1)
        return merge

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        basename = os.path.splitext(filename)[0]
        basename = "{:05d}".format(int(basename))
        img = cv2.imread(os.path.join(self.img_dir, filename))
        anno = cv2.imread(os.path.join(self.anno_dir, basename + '_hair.png'), cv2.IMREAD_GRAYSCALE)
        if anno is None:
            anno = cv2.imread(os.path.join(self.anno_dir, 'no_hair.png'), cv2.IMREAD_GRAYSCALE)
        anno[anno > 0] = 1
        img  = Image.fromarray(img)
        anno = Image.fromarray(anno)
        img.resize(anno.size, Image.BILINEAR)
        sample = {'image': img, 'label': anno}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

        return sample

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)