from PIL import Image
from torch.utils.data import Dataset
import pandas as pd 
import numpy as np 



class ImageDataset(Dataset):
    image_list = []
    label_list = []
    def __init__(self, data, label):
        super().__init__()
        self.image_list = list(data.drop("id", axis=1).to_numpy().reshape(-1, 28, 28))
        self.label_list = list(label.to_numpy())

    def __len__(self):
        return len(image_list)

    def __getitem__(self, idx):
        label = self.label_list[idx]
        img_arr = self.image_list[idx]

        img = Image.fromarray(img_arr, 'GRAY')

        return img, label