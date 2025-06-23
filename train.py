import os
import random
import numpy as np
import torch
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from datamodule import DataModule
from model import ConvolutionalNetwork

# Reproducibility
SEED = 1834579291
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    train_dir = "/kaggle/input/vggface2/train"
    val_dir   = "/kaggle/input/vggface2/val"

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    dm = DataModule(train_dir, val_dir, transform,
                    batch_size=64, num_workers=4, seed=SEED)

    model = ConvolutionalNetwork(num_classes=540, s=32.0, m=0.3, lr=1e-3)
   
    # Callbacks
    ckpt_cb = ModelCheckpoint(
        monitor='val_loss', mode='min', save_top_k=1,
        filename='best-{epoch:02d}-{val_loss:.2f}'
    )
    es_cb = EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=True)

    trainer = Trainer(
        max_epochs=5,
        accelerator="tpu",
        devices=1,
        callbacks=[ckpt_cb, es_cb],
    )

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()
