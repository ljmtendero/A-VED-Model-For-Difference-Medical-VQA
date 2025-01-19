import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torchvision import transforms
from transformers import AutoImageProcessor, SwinModel
import pandas as pd
import pathlib
from PIL import Image
import numpy as np
import albumentations as A
import cv2
import multiprocessing
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from paths import DICT_MIMICALL_OBS_TO_INT, IMAGES_MIMIC_PATH, MIMIC_PATH_TEST, MIMIC_PATH_TRAIN, MIMIC_PATH_VAL, SWINB_IMAGENET22K_WEIGHTS, DICT_MIMIC_OBSKEY_TO_INT

torch.set_float32_matmul_precision('medium')

# Define the Lightning Module
class SwinLightningModel(pl.LightningModule):
    def __init__(self, swin_weights, num_classes=14, lr=1e-4, weight_decay=0.05, epochs=30, use_weights=False):
        super().__init__()
        self.swin = SwinModel.from_pretrained(swin_weights)
        self.swin.train()
        self.processor = AutoImageProcessor.from_pretrained(swin_weights)
        self.classifier = nn.Linear(self.swin.config.hidden_size, num_classes * 2)
        if use_weights:
            self.class_weights = torch.load("class_weights.pt",weights_only=True).to("cuda")
        else:
            self.class_weights = torch.ones(14, 2).to("cuda")
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = nn.Dropout(0.5)
        self.epochs = epochs

    def forward(self, x):
        x = self.swin(x).pooler_output
        x = self.dropout(x)
        x = self.classifier(x)
        return x.view(-1, 14, 2)

    def training_step(self, batch, batch_idx):
        pixel_values, labels = batch
        outputs = self(pixel_values)
        loss = 0
        for i in range(14):
            loss += F.cross_entropy(outputs[:, i, :], labels[:, i], weight=self.class_weights[i], label_smoothing=0.1)
        loss /= 14
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values, labels = batch
        outputs = self(pixel_values)
        loss = 0
        for i in range(14):
            loss += F.cross_entropy(outputs[:, i, :], labels[:, i], weight=self.class_weights[i], label_smoothing=0.1)
        loss /= 14
        preds = torch.argmax(outputs, dim=2)
        accuracy = (preds == labels).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
model = SwinLightningModel(swin_weights=SWINB_IMAGENET22K_WEIGHTS)

# Define the Dataset
class MIMICDataset(Dataset):
    def __init__(self, transform, processor, partition, dataset_path, img_root_dir, label_map, labels):
        self.transform = transform
        self.processor = processor
        self.partition = partition
        self.dataset_df = pd.read_csv(dataset_path)
        if partition == "train":
            self.dataset_df = self.dataset_df[self.dataset_df["split"] == "train"]
        elif partition == "val":
            self.dataset_df = self.dataset_df[self.dataset_df["split"] == "validate"]
        elif partition == "test":
            self.dataset_df = self.dataset_df[self.dataset_df["split"] == "test"]
        else:
            raise ValueError("Unknown partition type.")
        
        self.img_root_dir = pathlib.Path(img_root_dir)
        self.label_map = label_map
        self.possible_labels = list(labels.keys())

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        img_name = self.img_root_dir / self.dataset_df.iloc[idx].image_path.split(",")[0]
        img = Image.open(img_name).convert("RGB")

        if isinstance(self.transform, transforms.Compose):
            img = self.transform(img)
        elif isinstance(self.transform, A.core.composition.Compose):
            img = self.transform(image=np.array(img))["image"]
        else:
            raise ValueError("Unknown transformation type.")

        img = self.processor(img, return_tensors="pt", size=384).pixel_values.squeeze()
        row = self.dataset_df.iloc[idx]
        labels = torch.zeros(14)
        for i in range(len(self.possible_labels)):
            inte_label = row[self.possible_labels[i]]
            if inte_label != inte_label:
                inte_label = -2
            labels[i] = self.label_map[inte_label]
        labels = labels.long()
        return img, labels

# Training Parameters
BATCH_SIZE = 24
num_workers = multiprocessing.cpu_count() - 1

# Define Transforms
def train_transforms():
    return A.Compose([
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-0.2, 0.1), rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, value=(105/256,), p=0.5),
        A.Resize(height=416, width=416),
        A.RandomCrop(height=384, width=384),
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5)
    ])

def val_transforms():
    return transforms.Compose([
        transforms.Resize(416),
        transforms.CenterCrop(384)
    ])

# Datasets and DataLoaders
train_dataset = MIMICDataset(transform=train_transforms(), 
                             processor=model.processor, 
                             partition="train", 
                             dataset_path="mimic_all_with_image_paths.csv", 
                             img_root_dir=IMAGES_MIMIC_PATH, 
                             label_map=DICT_MIMIC_OBSKEY_TO_INT,
                             labels=DICT_MIMICALL_OBS_TO_INT)
val_dataset = MIMICDataset(transform=val_transforms(),
                            processor=model.processor,
                            partition="val",
                            dataset_path="mimic_all_with_image_paths.csv",
                            img_root_dir=IMAGES_MIMIC_PATH,
                            label_map=DICT_MIMIC_OBSKEY_TO_INT,
                            labels=DICT_MIMICALL_OBS_TO_INT)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

# Callbacks
checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min", filename="swin_best")
lr_monitor = LearningRateMonitor(logging_interval="epoch")

# Trainer
trainer = pl.Trainer(
    max_epochs=30,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    callbacks=[checkpoint_callback, lr_monitor],
    log_every_n_steps=10,
    accumulate_grad_batches=3,
)

# Initialize and Train
trainer.fit(model, train_loader, val_loader)
