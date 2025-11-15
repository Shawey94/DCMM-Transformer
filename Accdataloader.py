from __future__ import print_function
from re import L
import re
from parameters import *
import torch.utils.data as data
import random
import os
import numpy as np
import torch
import albumentations as A
import cv2
import pdb
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import RandAugment
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


def img_resize(img, img_resize):
    min_size = min(img.shape[0:2])
    retio = float(img_resize / min_size)
    width = int(img.shape[1] * retio)
    height = int(img.shape[0] * retio)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized_img


class CP_Net_Dataset(data.Dataset):
    def __init__(self, data_training, data_testing, train=True, test=False, img_resize=224):
        self.train = train  # training set or val set
        self.test = test
        self.img_resize = img_resize

        # pdb.set_trace()
        if self.train:
            self.data = data_training
            self.transform = A.Compose([A.GaussNoise(p=0.2), 
                                        A.Resize(height = 224, width = 224, p =1.0), 
                                        A.HorizontalFlip(p = 0.5), 
                                        A.RandomBrightnessContrast(p=0.2), 
                                        A.Flip(p=0.2), 
                                        A.Normalize(mean=(0.5,0.5,0.5),std=(0.3,0.3,0.3),p=1.0),
                                        ])
        if self.test:
            self.data = data_testing
            self.transform = A.Compose([A.GaussNoise(p=0.2), 
                                        A.Resize(height = 224, width = 224, p =1.0), 
                                        A.HorizontalFlip(p = 0.5), 
                                        A.RandomBrightnessContrast(p=0.2), 
                                        A.Flip(p=0.2), 
                                        A.Normalize(mean=(0.5,0.5,0.5),std=(0.3,0.3,0.3),p=1.0),
                                        ])
        random.shuffle(self.data)


    def __getitem__(self, index):
        img_path, label_ID = self.data[index]
        img = cv2.imread(img_path)
        #resized_img = img_resize(img, self.img_resize)
        #resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=img)
        transformed_img = transformed["image"]
        #transformed_normalized_img = (transformed_img/255.)*2 - 1

        return transformed_img, label_ID

    def debug_getitem__(self, index=0):
        img_path, label_ID = self.data[index]
        img = cv2.imread(img_path)
        # if not self.train and not self.test:
        #     print(img_path)
        #     print(img.shape)
        # cv2.imwrite('./image_orig_resize_transform/' + str(index) + '_orig_(' + str(img.shape[0]) + '_' + str(img.shape[1]) + ')' + '.jpg', img)
        #resized_img = img_resize(img, self.img_resize)
        # cv2.imwrite('./image_orig_resize_transform/' + str(index) + '_resize_(' + str(resized_img.shape[0]) + '_' + str(resized_img.shape[1]) + ')' + '.jpg', resized_img)
        #resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=img)
        transformed_img = transformed["image"]
        # cv2.imwrite('./image_orig_resize_transform/' + str(index) + '_transformed_(' + str(transformed_img.shape[0]) + '_' + str(transformed_img.shape[1]) + ')' + '.jpg', cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR))
        #transformed_normalized_img = (transformed_img/255.)*2 - 1

        print(transformed_img.shape)
        #pdb.set_trace()

        return img_path, transformed_img, label_ID

    def __len__(self):
        return len(self.data)

'''
def get_loader():
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    trainset = datasets.ImageNet(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
    testset = datasets.ImageNet(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) 
    
    train_sampler = RandomSampler(trainset) 
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=opt.batch_size,
                              num_workers=4,
                              )
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=opt.batch_size,
                             num_workers=4,
                             ) if testset is not None else None
    return train_loader, test_loader
'''

def get_imagenet(root, target_transform = None):
        # transform_train = transforms.Compose([
        #     transforms.RandomResizedCrop((opt.res, opt.res), scale=(0.05, 1.0)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

        transform_train = transforms.Compose([
            #transforms.RandomHorizontalFlip(),   # Randomly flip the image horizontally
            #transforms.RandomRotation(10),        # Randomly rotate the image by up to 10 degrees
            RandAugment(),
            transforms.RandomResizedCrop((opt.res, opt.res), scale=(0.05, 1.0)),
            transforms.ToTensor(),                # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_val = transforms.Compose([
            transforms.Resize((opt.res, opt.res)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tra_root = os.path.join(root,'train')
        trainset = datasets.ImageFolder(root=tra_root,
                                transform=transform_train,
                                target_transform=target_transform)
        val_root = os.path.join(root,'val')
        valset = datasets.ImageFolder(root=val_root,
                                transform=transform_val,
                                target_transform=target_transform)
        return trainset,valset


def get_loader(root):
    trainset, testset = get_imagenet(root)

    train_sampler = RandomSampler(trainset) 
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=opt.batch_size,
                              num_workers=4,
                              pin_memory = True,
                              #shuffle = True,  #ValueError: DistributedSampler option is mutually exclusive with shuffle
                              )
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=opt.te_batch_size,
                             num_workers=4,
                             pin_memory = True,
                             #shuffle = True
                             ) if testset is not None else None
    return train_loader, test_loader



class CustomedINbreast(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.endswith('.jpg')
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        # Get label from filename
        if(filename.split('_')[2][0] == 'l'):
                label = int(filename.split('_')[2][1])
        else: 
                label = int(filename.split('_')[3][1])
        
        image = Image.fromarray(cv2.imread(img_path))
        if self.transform:
            image = self.transform(image)
        return image, label


class SIIMACRDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        """
        Args:
            image_paths (list): List of full image file paths.
            transform (callable, optional): Transform to apply to the images.
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)

        try:
            label = int(filename.split("_")[3].split(".")[0])
        except Exception as e:
            raise ValueError(f"Error extracting label from {filename}: {e}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label
    

class ADNIDataset(Dataset):
    def __init__(self, root_dir, label_csv, transform=None):
        self.transform = transform
        self.tensor_paths = []
        self.labels = []

        # Load CSV using column headers
        df = pd.read_csv(label_csv, dtype=str)

        # Filter to relevant classes
        df["subject_id"] = df["subject_id"].astype(str).str.strip()
        df = df[df["DX_bl"].isin(["CN", "EMCI", "LMCI"])]

        # Map label to binary: CN → 0, EMCI/LMCI → 1
        df["binary_label"] = df["DX_bl"].map(lambda x: 0 if x == "CN" else 1)

        # Make a dict for fast lookup: subject_id → binary label
        label_dict = dict(zip(df["subject_id"], df["binary_label"]))

        #print("Total subjects in label CSV:", len(label_dict))
        #print("Available folders in root_dir:", len(os.listdir(root_dir)))

        for subject_id in os.listdir(root_dir):
            subject_id = str(subject_id).strip()
            if subject_id not in label_dict:
                continue  # skip subjects without label

            tensor_file = os.path.join(root_dir, subject_id, "chosen_sequence", "T1_tensor.pt")
            if os.path.isfile(tensor_file):
                self.tensor_paths.append(tensor_file)
                self.labels.append(label_dict[subject_id])

    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        tensor_path = self.tensor_paths[idx]
        label = self.labels[idx]

        volume = torch.load(tensor_path)  # (31, 224, 224)
        image = torch.stack([volume[14], volume[15], volume[16]], dim=0)

        image = (image - image.min()) / (image.max() - image.min() + 1e-5)

        if self.transform:
            image = self.transform(image)

        return image, label
    

    

if __name__ == '__main__':

    transform_train = transforms.Compose([
        transforms.Resize((opt.res, opt.res)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    transform_adni = transforms.Compose([
        transforms.Resize((opt.res, opt.res)),
        #transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

        # Load datasets
    train_dataset = datasets.ImageFolder(root=opt.ChestXray_path+'/train', transform=transform_train)
    test_dataset = datasets.ImageFolder(root=opt.ChestXray_path+'/test', transform=transform_train)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)

    for i, (tra_transformed_normalized_img, tra_labels) in enumerate(train_loader):
        print(i)
        print(tra_labels)
        plt.imshow( np.transpose( tra_transformed_normalized_img[0], (1,2,0)))
        plt.show()



    # Shuffle and split
    random.seed(42)  # for reproducibility
    adni_path = opt.ADNI_path

        # Datasets
    adni_dataset = ADNIDataset(adni_path, label_csv = '/media/xw_stuff/CP-DCMM-Transformer/AD_labels.csv', transform=transform_adni)
    print("Dataset size:", len(adni_dataset))

    # Dataloaders
    train_loader = DataLoader(adni_dataset, batch_size=2, shuffle=True)

    for i, (tra_transformed_normalized_img, tra_labels) in enumerate(train_loader):
        print(i)
        print(tra_labels)
        plt.imshow( np.transpose( tra_transformed_normalized_img[0], (1,2,0)))
        plt.show()


    # All image paths
    all_image_paths = [
        os.path.join(opt.SIIMACR_path, fname)
        for fname in os.listdir(opt.SIIMACR_path)
        if fname.endswith('.jpg')
    ]

    # Shuffle and split
    random.seed(42)  # for reproducibility
    random.shuffle(all_image_paths)
    train_paths = all_image_paths[:1000]
    test_paths = all_image_paths[1000:]

        # Datasets
    train_dataset = SIIMACRDataset(train_paths, transform=transform_train)
    test_dataset = SIIMACRDataset(test_paths, transform=transform_train)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    for i, (tra_transformed_normalized_img, tra_labels) in enumerate(train_loader):
        print(i)
        print(tra_labels)
        plt.imshow( np.transpose( tra_transformed_normalized_img[0], (1,2,0)))
        plt.show()


    root_path = opt.INbreast_path
    dataset = CustomedINbreast(root_path+'/train', transform = transform_train)
    train_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
    for i, (tra_transformed_normalized_img, tra_labels) in enumerate(train_loader):
        print(i)
        print(tra_labels)
        plt.imshow( np.transpose( tra_transformed_normalized_img[0], (1,2,0)))
        plt.show()

    root_path = opt.imagenet_path
    train_loader, test_loader = get_loader(root_path)
    for i, (tra_transformed_normalized_img, tra_labels) in enumerate(train_loader):
        print(i)
        print(tra_labels)
        plt.imshow( np.transpose( tra_transformed_normalized_img[0], (1,2,0)))
        plt.show()




