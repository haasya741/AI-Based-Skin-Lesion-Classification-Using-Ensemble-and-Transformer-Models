

import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform(image_size, heavy=False):
    if heavy:
        return A.Compose([
            
            A.RandomResizedCrop(
                height=image_size, 
                width=image_size, 
                size=(image_size, image_size), 
                scale=(0.8, 1.0)
            ), 
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(),
            ToTensorV2(),
        ])


class PADUFESDataset(Dataset):
    def __init__(
        self, df, img_dir, transform, class_names,
        use_metadata=True, metadata_cols=None, label_col="diagnostic",
        image_file_map=None
    ):
        """
        Dataset for PAD-UFES images + metadata.
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.use_metadata = use_metadata
        self.label_col = label_col
        self.class_map = {name: idx for idx, name in enumerate(class_names)}
        self.metadata_cols = metadata_cols if metadata_cols else ["age", "gender"]

        
        if image_file_map is not None:
            self.image_map = image_file_map
        else:
            
            print("WARNING: Building image map inside DataLoader worker. This can be slow.")
            self.image_map = {
                fname: os.path.join(root, fname)
                for root, _, files in os.walk(self.img_dir)
                for fname in files
            }

        
        if self.use_metadata and "gender" in self.df.columns:
            if self.df["gender"].dtype == object:
                
                self.df["gender"] = self.df["gender"].map({"male": 0, "female": 1}).fillna(-1).astype(float)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

       
        img_path = self.image_map.get(row.img_id)
        if img_path is None:
            raise FileNotFoundError(f"Image {row.img_id} not found in pre-computed map.")

      
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        img = self.transform(image=img)["image"]

        
        if row[self.label_col] not in self.class_map:
            raise ValueError(f"Invalid class {row[self.label_col]} in dataset.")
        label = self.class_map[row[self.label_col]]

      
        if self.use_metadata:
            # Use safe extraction with fallback values
            metadata_values = []
            for col in self.metadata_cols:
                if col in row.index and not pd.isna(row[col]):
                    metadata_values.append(float(row[col]))
                else:
                    metadata_values.append(0.0)  # Default value for missing data
            
            metadata = torch.tensor(metadata_values, dtype=torch.float32)
            return img, metadata, label
        else:
            # Correctly return tuple of length 2 when metadata is disabled
            return img, label 

    def __len__(self):
        return len(self.df)


