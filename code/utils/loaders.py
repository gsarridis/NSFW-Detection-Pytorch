import os

import torch
from torchvision import transforms

from utils.custom_nsfw_dataset import Nsfw


def custom_nsfw_data_loader(batch = 128, path = './data/', path2 = './data/nude_net_data/nude_sexy_safe_v1_x320', return_ids = True, size=240):
    # Data
    print('==> Preparing data..')
    if size == 240:
        precrop = 300
        crop = 240
    elif size == 224:
        precrop = 256
        crop = 224
        
    transform_train = transforms.Compose([

    transforms.Resize((precrop, precrop)),
    transforms.RandomCrop((crop, crop)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_test = transforms.Compose([

        transforms.Resize((crop, crop)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    trainset = Nsfw(fold_dir=os.path.join(path, 'splits', 'train_90.txt'), root_dir=os.path.join(path, '/pornography-2k/images'), root_dir2=os.path.join(path2, 'training'), transform=transform_train, return_ids=return_ids)
    valset = Nsfw(fold_dir=os.path.join(path, 'splits', 'test_10.txt'), root_dir=os.path.join(path, '/pornography-2k/images'), root_dir2=None, transform=transform_test, return_ids=return_ids)


    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch, shuffle=True, num_workers=8, pin_memory=True)

    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch, shuffle=False, num_workers=8, pin_memory=True)
    
    return trainloader, valloader
