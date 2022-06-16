from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import numpy as np
from PIL import Image

class Nsfw(Dataset):
    """Pornography Images dataset."""

    def __init__(self, fold_dir, root_dir, root_dir2, transform=None, return_ids=True):
        """Initializes the dataset

        Args:
            fold_dir (str): the directory of the csvs
            root_dir (str): the path that contains the pornography-2k images
            root_dir2 (str): the path that contains the Nudenet-data
            transform (_type_, optional): The torch transformations. Defaults to None.
            return_ids (bool, optional): If yes, dataloaders will return the frame ids exept for the inputs and the targets. Defaults to True.
        """
        self.data = pd.read_csv(fold_dir,names=["FileList", "Label"]) # Labels
        self.root_dir = root_dir
        self.root_dir2 = root_dir2
        self.transform = transform
        self.return_ids = return_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        name = self.data.iloc[idx, 0]
        # define the image path.
        if 'vPorn' in name:
            cat = 'porn'
            img_name = os.path.join(self.root_dir, cat,
                                name)
        elif 'vNonPorn' in name:
            cat = 'nonporn'
            img_name = os.path.join(self.root_dir, cat,
                                name)
        elif self.data.iloc[idx, 1] == 1:
            cat = 'nude'
            img_name = os.path.join(self.root_dir2, cat,
                                name)
        elif self.data.iloc[idx, 1] == 0:
            cat = 'safe'
            img_name = os.path.join(self.root_dir2, cat,
                                name)

        image = Image.open(img_name)
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        if self.return_ids:
            return image, label, name
        else: 
            return image, label



