import os
import os.path
from collections import defaultdict
from torch_geometric_temporal.dataset import MTMDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import torchvision
from torchvision import datasets, transforms
from loguru import logger
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.utils.data as data
torch.manual_seed(0)

def filter_OutZero(data):
    data = [i for i in data if i.sum() != 0]
    return data

def mtmLoader():
    loader = MTMDatasetLoader()

    dataset = loader.get_dataset(frames=24)

    train_dataset, val_test_dataset = temporal_signal_split(dataset, train_ratio=0.7)
    val_dataset, test_dataset = temporal_signal_split(val_test_dataset, train_ratio=0.5)

    filter_train_dataset = filter_OutZero(train_dataset.features)
    filter_test_dataset = filter_OutZero(test_dataset.features)
    filter_val_dataset = filter_OutZero(val_dataset.features)

    # train_loader = DataLoader(filter_train_dataset, batch_size=2, shuffle=True, num_workers=2, drop_last=True)
    # test_loader = DataLoader(filter_test_dataset, batch_size=2, shuffle=True, num_workers=2, drop_last=True)
    # val_loader = DataLoader(filter_val_dataset, batch_size=2, shuffle=True, num_workers=2, drop_last=True)
    # dataset_loader = DataLoader(filter_train_dataset, batch_size=2, shuffle=True, num_workers=2, drop_last=True)

    # Investigating the dataset
    print("Dataset type: ", type(dataset))
    print("Index tensor of edges ", dataset.edge_index.transpose())

    print("Edge weight tensor ", dataset.edge_weight)
    print("Length of node feature tensors: ",
          len(dataset.features))  # Length of node feature tensors:  14453 means there is 14453 samples in this dataset
    print("List of node feature tensors: ", dataset.features[
        0].shape)  # each sample have the shape of (3, 21, 16) 3 channel-> x,y,z , 21 node points, with 16 time frame
    print("List of node label (target) tensors: ", dataset.targets[
        0].shape)  # each sample have the the target shape of (16, 6) 16 time frame and target label class per frame  (in this case all 16 times frame should have the same class)

    # 6classes

    # Grasp, Release, Move, Reach, Position plus a negative class for frames without graph signals (no hand present).

    total = 0
    for time, snapshot in enumerate(train_dataset):
        # print(time) # the time here is just an index,, not neccesary to be a time
        if time == 20:
            # print(snapshot.x.shape,snapshot.y.shape, snapshot.edge_index.shape, snapshot.edge_attr.shape)
            break

    print(type(snapshot.x))
    print(snapshot.x[0])
    numpy_x = snapshot.x.numpy()
    print(type(numpy_x))
    # print(numpy_x[0])

    print(snapshot.y)

    return filter_train_dataset,filter_test_dataset,filter_val_dataset

def otherLoader():
    raise NotImplementedError

def mnistLoader():


    transformations = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train = datasets.CIFAR10('.', train=True, download=True,
                           transform=transformations)

    test = datasets.CIFAR10('.', train=False, download=True,
                           transform=transformations)

    logger.info("Mnist loaded successfully!")
    return train,test,test