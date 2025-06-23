import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from facenet_pytorch import InceptionResnetV1
from losses import ArcFaceLoss

class ConvolutionalNetwork(LightningModule):
    def __init__(self, num_classes=540, s=32.0, m=0.3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Pretrained face-recognition backbone
        self.backbone = InceptionResnetV1(pretrained='vggface2', classify=False)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
        )

        # Final ArcFace layer
        self.fc  = nn.Linear(512, num_classes, bias=False)
        self.arc = ArcFaceLoss(n_classes=num_classes, s=s, m=m)

    def encode(self, x):
        emb = self.backbone(x)
        emb = self.projection(emb)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x):
        return self.encode(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        emb = self.encode(x)
        loss = self.arc(emb, self.fc.weight, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        emb = self.encode(x)
        loss = self.arc(emb, self.fc.weight, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
