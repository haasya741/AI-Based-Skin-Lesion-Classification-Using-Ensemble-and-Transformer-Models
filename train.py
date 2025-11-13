


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
# NOTE: Assuming these imports point to your custom files
from data import PADUFESDataset, get_transform 
from model import EnsembleModel
from lit_model import LitEnsembleModel 
from utils import FocalLoss 

import torch.nn as nn
import yaml
import pandas as pd
import torch
import os
import timm


# -------------------------
# Models that require 224x224 - Standardizing this check
# -------------------------
MODELS_REQUIRE_224 = {
    # UPDATED FOR NEW ENSEMBLE
    "vit_base_patch16_224",         
    "swin_base_patch4_window7_224", 
    "swin_t",
    "swin_small_patch4_window7_224",
    "deit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
    "resnet50"
}


def main():
  
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    
    lr = float(config.get("lr", 1e-4))
    weight_decay = float(config.get("weight_decay", 1e-5))
    
    
    NUM_WORKERS = config.get("num_workers", 4) 
    PIN_MEMORY = config.get("pin_memory", True)
    PERSISTENT_WORKERS = NUM_WORKERS > 0
    
   
    img_dir = config.get("data_dir", "images")

    
    full_df = pd.read_csv(config.get("metadata_file", "metadata.csv"))
    full_df["diagnostic"] = full_df["diagnostic"].astype(str).str.strip().str.upper()

    # Build image map first by walking through all subdirectories
    image_map = {
        fname: os.path.join(root, fname)
        for root, _, files in os.walk(img_dir)
        for fname in files
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')) # Only map image files
    }

    # Filter metadata to only include rows with existing images
    initial_count = len(full_df)
    full_df = full_df[full_df["img_id"].isin(image_map.keys())].reset_index(drop=True)
    filtered_count = len(full_df)
    
    print(f"📊 Filtered dataset: {filtered_count}/{initial_count} samples have existing images")
    print(f"⚠️ Missing {initial_count - filtered_count} images from metadata")

    if filtered_count == 0:
        raise ValueError("❌ No matching images found in the data directory. Check 'img_dir' and 'img_id' in metadata.")

  
    gss = GroupShuffleSplit(n_splits=1, test_size=config.get("val_split", 0.2), random_state=42)
    # Ensure 'patient_id' column exists before splitting
    if "patient_id" not in full_df.columns:
        raise ValueError("❌ 'patient_id' column required for GroupShuffleSplit not found in metadata.")
        
    train_idx, val_idx = next(gss.split(full_df, groups=full_df["patient_id"]))

    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_df.iloc[val_idx].reset_index(drop=True)

   
    dataset_classes = sorted(full_df["diagnostic"].unique().tolist())
    if "class_names" not in config or set(config["class_names"]) != set(dataset_classes):
        print("⚠️ Config class names mismatched or missing, using dataset classes instead.")
        class_names = dataset_classes
    else:
        class_names = [c.strip().upper() for c in config["class_names"]]

    print(f"✅ Final Classes ({len(class_names)}): {class_names}")

 
    model_names = config.get("models", ["efficientnetv2_s", "resnet50", "vit_base_patch16_224"])
    current_image_size = config.get("image_size", 384)
    
    requires_224 = any(model_name in MODELS_REQUIRE_224 for model_name in model_names)
    
    if requires_224 and current_image_size != 224:
        print(f"⚠️ At least one model requires 224x224, overriding image_size from {current_image_size} to 224")
        current_image_size = 224
    
    config["image_size"] = current_image_size # Update config for dataset creation
    print(f"📌 Using image size: {config['image_size']}")

 
    metadata_cols = config.get("metadata_cols", ["age", "gender"])
    metadata_dim = len(metadata_cols) if config.get("use_metadata", False) else 0
    print(f"📌 Using metadata dimension: {metadata_dim} (Cols: {metadata_cols})")

   
    train_dataset = PADUFESDataset(
        df=train_df,
        img_dir=img_dir,
        transform=get_transform(config["image_size"], heavy=True),
        class_names=class_names,
        use_metadata=config["use_metadata"],
        metadata_cols=metadata_cols,
        label_col="diagnostic",
        image_file_map=image_map # PASSING THE MAP
    )

    val_dataset = PADUFESDataset(
        df=val_df,
        img_dir=img_dir,
        transform=get_transform(config["image_size"], heavy=False),
        class_names=class_names,
        use_metadata=config["use_metadata"],
        metadata_cols=metadata_cols,
        label_col="diagnostic",
        image_file_map=image_map # PASSING THE MAP
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )

  
    available_models = []
    pretrained_config = config.get("pretrained", True)
    
    for m in model_names:
        if m not in timm.list_models():
            print(f"⚠️ Model {m} not found in timm. Skipping.")
            continue
            
        # Test if pretrained weights are available
        if pretrained_config:
            try:
                # Try creating a small test model to check pretrained weights
                test_model = timm.create_model(m, pretrained=True, num_classes=1000)
                test_model = None # Clean up
                available_models.append(m)
                print(f"✅ {m} - pretrained weights available")
            except Exception as e:
                # This catches both RuntimeError (no weights) and other potential errors
                if "No pretrained weights exist" in str(e):
                    print(f"⚠️ {m} - no pretrained weights, will use random initialization")
                    available_models.append(m)
                else:
                    print(f"❌ {m} - error loading model: {e}")
        else:
            available_models.append(m)
            print(f"✅ {m} - using random initialization (as per config)")

    if not available_models:
        raise ValueError("❌ No valid models found. Please check model names.")

    print(f"📌 Final model list: {available_models}")

    # Create ensemble 
    ensemble = EnsembleModel(
        model_names=available_models,
        num_classes=len(class_names),
        use_metadata=config["use_metadata"],
        metadata_dim=metadata_dim,
        # Pass the config setting, and let EnsembleModel handle the per-model loading
        pretrained=pretrained_config 
    )

    print(f"✅ Created ensemble with {len(available_models)} models")

 
    loss_fn = FocalLoss() if config.get("loss", "crossentropy") == "focal" else nn.CrossEntropyLoss()

    lit_model = LitEnsembleModel(
        ensemble,
        loss_fn=loss_fn,
        lr=lr,
        weight_decay=weight_decay
    )

    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_macro_f1",
        mode="max",
        save_top_k=3,
        dirpath="checkpoints",
        filename="{epoch:02d}-{val_macro_f1:.2f}",
        save_on_train_epoch_end=False
    )

    early_stop_callback = EarlyStopping(
        monitor="val_macro_f1",
        mode="max",
        patience=config.get("patience", 5)
    )

  
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="gpu", 
        devices=config.get("devices", 1),
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=True
    )

   
    print("🚀 Starting training...")
    trainer.fit(lit_model, train_loader, val_loader)
    print("✅ Training Completed Successfully!")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True) 
    main()
