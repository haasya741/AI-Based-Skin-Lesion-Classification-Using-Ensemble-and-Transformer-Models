# import os
# import pandas as pd
# from torch.utils.data import Dataset
# from PIL import Image
# import numpy as np
# import torch
# import albumentations as A
# from albumentations.pytorch import ToTensorV2


# def get_transform(image_size, heavy=False):
# # ... (transforms code remains unchanged) ...
#     if heavy:
#         return A.Compose([
#             A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0)),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.RandomBrightnessContrast(p=0.2),
#             A.Normalize(),
#             ToTensorV2(),
#         ])
#     else:
#         return A.Compose([
#             A.Resize(height=image_size, width=image_size),
#             A.Normalize(),
#             ToTensorV2(),
#         ])


# class PADUFESDataset(Dataset):
#     def __init__(
#         self, df, img_dir, transform, class_names,
#         use_metadata=True, metadata_cols=None, label_col="diagnostic",
#         image_file_map=None # ADDED ARGUMENT
#     ):
#         """
#         Dataset for PAD-UFES images + metadata.
#         """
#         self.df = df.reset_index(drop=True)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.use_metadata = use_metadata
#         self.label_col = label_col
#         self.class_map = {name: idx for idx, name in enumerate(class_names)}
#         self.metadata_cols = metadata_cols if metadata_cols else ["age", "gender"]

#         # MODIFIED: Use the passed map instead of rebuilding it
#         if image_file_map is not None:
#             self.image_map = image_file_map
#         else:
#             # Fallback (original logic) if map isn't passed, though not recommended with DataLoader workers
#             print("WARNING: Building image map inside DataLoader worker. This can be slow.")
#             self.image_map = {
#                 fname: os.path.join(root, fname)
#                 for root, _, files in os.walk(self.img_dir)
#                 for fname in files
#             }

#         # Encode gender column if it exists
#         if self.use_metadata and "gender" in self.df.columns:
#             if self.df["gender"].dtype == object:
#                 self.df["gender"] = self.df["gender"].map({"male": 0, "female": 1}).fillna(-1).astype(float)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]

#         # Use the full path stored in image_map
#         img_path = self.image_map.get(row.img_id)
#         if img_path is None:
#             # Should have been filtered out in train.py, but good to check
#             raise FileNotFoundError(f"Image {row.img_id} not found in pre-computed map.")

#         # Load and transform image
#         img = Image.open(img_path).convert("RGB")
#         img = np.array(img)
#         img = self.transform(image=img)["image"]

#         # Map diagnostic to class index
#         if row[self.label_col] not in self.class_map:
#             raise ValueError(f"Invalid class {row[self.label_col]} in dataset.")
#         label = self.class_map[row[self.label_col]]

#         # Handle metadata if enabled
#         if self.use_metadata:
#             # Fixed: Use safe extraction with fallback values
#             metadata_values = []
#             for col in self.metadata_cols:
#                 if col in row.index and not pd.isna(row[col]):
#                     metadata_values.append(float(row[col]))
#                 else:
#                     metadata_values.append(0.0)  # Default value for missing data
            
#             metadata = torch.tensor(metadata_values, dtype=torch.float32)
#             return img, metadata, label
#         else:
#             # If not using metadata, return None for metadata
#             return img, None, label

#     def __len__(self):
#         return len(self.df)


# import os
# import pandas as pd
# from torch.utils.data import Dataset
# from PIL import Image
# import numpy as np
# import torch
# import albumentations as A
# from albumentations.pytorch import ToTensorV2


# def get_transform(image_size, heavy=False):
#     if heavy:
#         return A.Compose([
#             # FIXED: Explicitly pass height and width (and size argument if needed by version)
#             A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0)), 
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.RandomBrightnessContrast(p=0.2),
#             A.Normalize(),
#             ToTensorV2(),
#         ])
#     else:
#         return A.Compose([
#             A.Resize(height=image_size, width=image_size),
#             A.Normalize(),
#             ToTensorV2(),
#         ])


# class PADUFESDataset(Dataset):
#     def __init__(
#         self, df, img_dir, transform, class_names,
#         use_metadata=True, metadata_cols=None, label_col="diagnostic",
#         image_file_map=None
#     ):
#         """
#         Dataset for PAD-UFES images + metadata.
#         """
#         self.df = df.reset_index(drop=True)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.use_metadata = use_metadata
#         self.label_col = label_col
#         self.class_map = {name: idx for idx, name in enumerate(class_names)}
#         self.metadata_cols = metadata_cols if metadata_cols else ["age", "gender"]

#         # MODIFIED: Use the passed map instead of rebuilding it
#         if image_file_map is not None:
#             self.image_map = image_file_map
#         else:
#             # Fallback (original logic)
#             print("WARNING: Building image map inside DataLoader worker. This can be slow.")
#             self.image_map = {
#                 fname: os.path.join(root, fname)
#                 for root, _, files in os.walk(self.img_dir)
#                 for fname in files
#             }

#         # Encode gender column if it exists
#         if self.use_metadata and "gender" in self.df.columns:
#             if self.df["gender"].dtype == object:
#                 self.df["gender"] = self.df["gender"].map({"male": 0, "female": 1}).fillna(-1).astype(float)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]

#         # Use the full path stored in image_map
#         img_path = self.image_map.get(row.img_id)
#         if img_path is None:
#             raise FileNotFoundError(f"Image {row.img_id} not found in pre-computed map.")

#         # Load and transform image
#         img = Image.open(img_path).convert("RGB")
#         img = np.array(img)
#         img = self.transform(image=img)["image"]

#         # Map diagnostic to class index
#         if row[self.label_col] not in self.class_map:
#             raise ValueError(f"Invalid class {row[self.label_col]} in dataset.")
#         label = self.class_map[row[self.label_col]]

#         # Handle metadata if enabled
#         if self.use_metadata:
#             # Use safe extraction with fallback values
#             metadata_values = []
#             for col in self.metadata_cols:
#                 if col in row.index and not pd.isna(row[col]):
#                     metadata_values.append(float(row[col]))
#                 else:
#                     metadata_values.append(0.0)  # Default value for missing data
            
#             metadata = torch.tensor(metadata_values, dtype=torch.float32)
#             return img, metadata, label
#         else:
#             # MODIFIED: Return tuple of length 2 when metadata is disabled
#             return img, label 

#     def __len__(self):
#         return len(self.df)



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
            # FIX APPLIED: Explicitly including 'size' to satisfy strict validation 
            # in newer albumentations/pydantic versions.
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

        # MODIFIED: Use the passed map instead of rebuilding it
        if image_file_map is not None:
            self.image_map = image_file_map
        else:
            # Fallback (original logic)
            print("WARNING: Building image map inside DataLoader worker. This can be slow.")
            self.image_map = {
                fname: os.path.join(root, fname)
                for root, _, files in os.walk(self.img_dir)
                for fname in files
            }

        # Encode gender column if it exists
        if self.use_metadata and "gender" in self.df.columns:
            if self.df["gender"].dtype == object:
                # Use .fillna(-1) for unmapped/missing gender values
                self.df["gender"] = self.df["gender"].map({"male": 0, "female": 1}).fillna(-1).astype(float)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Use the full path stored in image_map
        img_path = self.image_map.get(row.img_id)
        if img_path is None:
            raise FileNotFoundError(f"Image {row.img_id} not found in pre-computed map.")

        # Load and transform image
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        img = self.transform(image=img)["image"]

        # Map diagnostic to class index
        if row[self.label_col] not in self.class_map:
            raise ValueError(f"Invalid class {row[self.label_col]} in dataset.")
        label = self.class_map[row[self.label_col]]

        # Handle metadata if enabled
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


# import os
# import pandas as pd
# from torch.utils.data import Dataset
# from PIL import Image
# import numpy as np
# import torch
# import albumentations as A
# from albumentations.pytorch import ToTensorV2


# def get_transform(image_size, heavy=False):
#     # Standard ImageNet normalization parameters (often works well with pre-trained models)
#     IMAGENET_MEAN = [0.485, 0.456, 0.406]
#     IMAGENET_STD = [0.229, 0.224, 0.225]

#     if heavy:
#         return A.Compose([
#             # FIX: Explicitly including 'size' to satisfy strict validation 
#             A.RandomResizedCrop(
#                 height=image_size, 
#                 width=image_size, 
#                 size=(image_size, image_size), 
#                 scale=(0.75, 1.0) # Slightly more aggressive crop scale
#             ), 
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
            
#             # ADDED: Stronger geometric augmentation
#             A.ShiftScaleRotate(
#                 shift_limit=0.0625, 
#                 scale_limit=0.1, 
#                 rotate_limit=15, 
#                 p=0.5, 
#                 border_mode=cv2.BORDER_CONSTANT # Ensure clean borders after transformation
#             ), 
#             # ADDED: Stronger color augmentation
#             A.RandomBrightnessContrast(
#                 brightness_limit=0.3, 
#                 contrast_limit=0.3, 
#                 p=0.5
#             ),
            
#             A.Normalize(
#                 mean=IMAGENET_MEAN,
#                 std=IMAGENET_STD
#             ), 
#             ToTensorV2(),
#         ])
#     else:
#         return A.Compose([
#             A.Resize(height=image_size, width=image_size),
#             A.Normalize(
#                 mean=IMAGENET_MEAN,
#                 std=IMAGENET_STD
#             ),
#             ToTensorV2(),
#         ])


# class PADUFESDataset(Dataset):
#     def __init__(
#         self, df, img_dir, transform, class_names,
#         use_metadata=True, metadata_cols=None, label_col="diagnostic",
#         image_file_map=None
#     ):
#         """
#         Dataset for PAD-UFES images + metadata.
#         """
#         self.df = df.reset_index(drop=True)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.use_metadata = use_metadata
#         self.label_col = label_col
#         self.class_map = {name: idx for idx, name in enumerate(class_names)}
#         self.metadata_cols = metadata_cols if metadata_cols else ["age", "gender"]

#         # MODIFIED: Use the passed map instead of rebuilding it
#         if image_file_map is not None:
#             self.image_map = image_file_map
#         else:
#             # Fallback (original logic)
#             print("WARNING: Building image map inside DataLoader worker. This can be slow.")
#             self.image_map = {
#                 fname: os.path.join(root, fname)
#                 for root, _, files in os.walk(self.img_dir)
#                 for fname in files
#             }

#         # Encode gender column if it exists
#         if self.use_metadata and "gender" in self.df.columns:
#             if self.df["gender"].dtype == object:
#                 # Use .fillna(-1) for unmapped/missing gender values
#                 self.df["gender"] = self.df["gender"].map({"male": 0, "female": 1}).fillna(-1).astype(float)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]

#         # Use the full path stored in image_map
#         img_path = self.image_map.get(row.img_id)
#         if img_path is None:
#             raise FileNotFoundError(f"Image {row.img_id} not found in pre-computed map.")

#         # Load and transform image
#         # Note: We import cv2 temporarily here to handle the ShiftScaleRotate dependency, 
#         # though ideally it's imported at the top. Since it's not in the existing imports, 
#         # we'll assume it's available or add it if necessary.
#         try:
#             import cv2
#         except ImportError:
#             print("Warning: cv2 not imported. ShiftScaleRotate may fail if not imported globally.")
            
#         img = Image.open(img_path).convert("RGB")
#         img = np.array(img)
#         img = self.transform(image=img)["image"]

#         # Map diagnostic to class index
#         if row[self.label_col] not in self.class_map:
#             raise ValueError(f"Invalid class {row[self.label_col]} in dataset.")
#         label = self.class_map[row[self.label_col]]

#         # Handle metadata if enabled
#         if self.use_metadata:
#             # Use safe extraction with fallback values
#             metadata_values = []
#             for col in self.metadata_cols:
#                 if col in row.index and not pd.isna(row[col]):
#                     metadata_values.append(float(row[col]))
#                 else:
#                     metadata_values.append(0.0)  # Default value for missing data
            
#             metadata = torch.tensor(metadata_values, dtype=torch.float32)
#             return img, metadata, label
#         else:
#             # Correctly return tuple of length 2 when metadata is disabled
#             return img, label 

#     def __len__(self):
#         return len(self.df)
