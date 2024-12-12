# house_dataset.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class HouseRegressionData(Dataset):
    def __init__(self, csv_file, image_folder, price_col, transform=None, use_metadata=False, classification=False, threshold=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.price_col = price_col
        self.transform = transform
        self.use_metadata = use_metadata
        self.classification = classification
        self.threshold = threshold

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        row = self.data.iloc[idx]
        image_filename = f"{str(row['image_id'])}.jpg"
        value = row[self.price_col]

        
        if self.classification:
 
            target = 1 if value > self.threshold else 0
        else:
  
            target = value

        image_path = os.path.join(self.image_folder, image_filename)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)


        return image, torch.tensor(target, dtype=torch.float32)


class HouseDataImage(Dataset):
    def __init__(self, csv_file, image_folder, price_col=None, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.price_col = price_col 
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
    
        row = self.data.iloc[idx]
        image_filename = f"{str(row['image_id'])}.jpg"
        image_path = os.path.join(self.image_folder, image_filename)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.price_col:
      
            log_price = row[self.price_col]
            return image, torch.tensor(log_price, dtype=torch.float32), row['image_id']
        else:

            return image, row['image_id']
