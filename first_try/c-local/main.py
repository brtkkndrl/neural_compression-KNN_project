import os

os.environ["QT_QPA_PLATFORM"] = "wayland"

import matplotlib.pyplot as plt
import lightning.pytorch as pl
import matplotlib

matplotlib.use("Qt5Agg")

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, Subset

torch.set_float32_matmul_precision("medium")

DATASET_ROOT = Path("datasets/imagenet-mini")

BATCH_SIZE = 8
IMAGE_SIZE = 256
NUM_WORKERS = 2
TRAIN_LIMIT = None
VAL_LIMIT = None


class DatasetFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.image_folder = ImageFolder(root=str(self.root))
        self.samples = list(self.image_folder.samples)
        self.classes = list(self.image_folder.classes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, _ = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


def subset_dataset(dataset, limit):
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def count_images(directory):
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in image_suffixes
    )


class ImageNetSubsetDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir,
            batch_size=16,
            num_workers=2,
            image_size=256,
            train_limit=None,
            val_limit=None,
            pin_memory=True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_limit = train_limit
        self.val_limit = val_limit
        self.pin_memory = pin_memory
        self.train_dataset = None
        self.val_dataset = None
        self.class_names = []

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

    def prepare_data(self):
        if not self.train_dir.exists():
            raise FileNotFoundError(f"Missing training directory: {self.train_dir}")
        if not self.val_dir.exists():
            raise FileNotFoundError(f"Missing validation directory: {self.val_dir}")

    def setup(self, stage=None):
        train_dataset = DatasetFolder(self.train_dir, transform=self.transform)
        val_dataset = DatasetFolder(self.val_dir, transform=self.transform)
        self.class_names = train_dataset.classes
        self.train_dataset = subset_dataset(train_dataset, self.train_limit)
        self.val_dataset = subset_dataset(val_dataset, self.val_limit)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def describe(self):
        return {
            "data_dir": str(self.data_dir),
            "train_dir": str(self.train_dir),
            "val_dir": str(self.val_dir),
            "num_classes": len(self.class_names),
            "train_images_on_disk": count_images(self.train_dir),
            "val_images_on_disk": count_images(self.val_dir),
            "train_images_loaded": len(self.train_dataset),
            "val_images_loaded": len(self.val_dataset),
            "image_size": self.image_size,
            "batch_size": self.batch_size,
        }


def show_tensor_batch(batch, max_images=8, title="Sample images"):
    if isinstance(batch, (tuple, list)):
        batch = batch[0]

    batch = batch[:max_images].detach().cpu()
    num_images = batch.shape[0]
    figure, axes = plt.subplots(1, num_images, figsize=(3 * num_images, 3))
    if num_images == 1:
        axes = [axes]

    for axis, image_tensor in zip(axes, batch):
        image = image_tensor.permute(1, 2, 0).clamp(0, 1).numpy()
        axis.imshow(image)
        axis.axis("off")

    figure.suptitle(title)
    figure.tight_layout()
    plt.show()


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            32, 64, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            16, 3, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x


class LitAutoencoder(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


if __name__ == "__main__":
    datamodule = ImageNetSubsetDataModule(
        data_dir=DATASET_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE,
        train_limit=TRAIN_LIMIT,
        val_limit=VAL_LIMIT,
    )

    datamodule.prepare_data()
    datamodule.setup("fit")
    summary = datamodule.describe()
    print("Dataset Summary:", summary)

    train_loader = datamodule.train_dataloader()
    sample_batch = next(iter(train_loader))
    print(f"Batch shape: {tuple(sample_batch.shape)}")
    show_tensor_batch(
        sample_batch, max_images=6, title="ImageNet subset preview (Originals)"
    )

    model = LitAutoencoder(learning_rate=1e-3)

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="auto",
        devices=1,
    )

    print("Starting Training...")
    trainer.fit(model, datamodule)

    print("Training finished! Showing reconstructions...")
    model.eval()
    with torch.no_grad():
        reconstructed_batch = model(sample_batch.to(model.device))

    show_tensor_batch(reconstructed_batch, max_images=6, title="Reconstructed Images")
