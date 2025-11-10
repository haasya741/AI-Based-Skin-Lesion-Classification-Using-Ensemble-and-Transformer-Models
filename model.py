import torch
import torch.nn as nn
import timm


class SingleModel(nn.Module):
    def __init__(self, model_name, num_classes, use_metadata, metadata_dim=3, pretrained=False):
        super().__init__()

        self.use_metadata = use_metadata
        
        # Try to create model with pretrained weights, fallback to random init if needed
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0  # 🔹 No classifier head, we want embeddings
            )
            if pretrained:
                print(f"✅ {model_name} loaded with pretrained weights")
            else:
                print(f"📌 {model_name} loaded with random initialization")
        except RuntimeError as e:
            if "No pretrained weights exist" in str(e) and pretrained:
                print(f"⚠️ {model_name} pretrained weights not available, using random initialization")
                self.backbone = timm.create_model(
                    model_name,
                    pretrained=False,
                    num_classes=0
                )
            else:
                raise e

        # Get number of features from backbone
        in_features = self.backbone.num_features

        # If we use metadata, final input = CNN_features + metadata_dim
        if self.use_metadata:
            self.fc = nn.Linear(in_features + metadata_dim, num_classes)
        else:
            self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x_img, metadata=None):
        # Extract image features from backbone
        img_features = self.backbone(x_img)

        # Combine image features + metadata if needed
        if self.use_metadata and metadata is not None:
            x = torch.cat([img_features, metadata], dim=1)
        else:
            x = img_features

        return self.fc(x)


class EnsembleModel(nn.Module):
    def __init__(self, model_names, num_classes, use_metadata, metadata_dim=3, pretrained=False, config_pretrained=None):
        super().__init__()
        
        # Use config_pretrained if provided, otherwise fall back to pretrained
        use_pretrained = config_pretrained if config_pretrained is not None else pretrained
        
        self.models = nn.ModuleList([
            SingleModel(m, num_classes, use_metadata, metadata_dim, pretrained=use_pretrained)
            for m in model_names
        ])
        
        print(f"🔥 Created ensemble with {len(self.models)} models")

    def forward(self, x_img, x_meta=None):
        logits = []
        for model in self.models:
            logits.append(model(x_img, x_meta))
        return torch.stack(logits).mean(0)