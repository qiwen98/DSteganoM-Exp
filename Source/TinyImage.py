import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
from torchvision.io import read_image, ImageReadMode

class TrainTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("tiny-imagenet-200/train/*/*/*.JPEG")
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        # print(img_path)
        # print(img_path.split("\\")[1])
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        label = self.id_dict[img_path.split("\\")[1]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label

class TestTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("tiny-imagenet-200//val//images//*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open('tiny-imagenet-200//val//val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0],a[1]
            self.cls_dic[img] = self.id_dict[cls_id]
 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        label = self.cls_dic[img_path.split('\\')[1]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label