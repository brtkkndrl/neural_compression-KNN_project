import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
import os

from data import ClassImagesDataModule, DF2KDataModule

from models import get_model

torch.set_float32_matmul_precision("medium")

datamodule_default_imagenet10k = ClassImagesDataModule(
    data_dir="datasets/imagenet_10K/imagenet_subtrain",
    batch_size=8,
    random_crop=True
)

datamodule_df2k = DF2KDataModule(
    train_dir="datasets/DF2K/train",
    test_dir="datasets/DF2K/test",
    batch_size=8,
    random_crop=True
)


def experiment1():
    """
        Train a basic AE on ImageNet.
    """
    EXPERIMENT_NAME = "basic_imagenet10k"
    MODEL_NAME = "basic"
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    
    model = get_model(MODEL_NAME, learning_rate=LEARNING_RATE)

    checkpoint_filename = f"{EXPERIMENT_NAME}-{MODEL_NAME}-best"

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=checkpoint_filename,
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        callbacks=[checkpoint_callback],
    )

    print("="*30)
    print(f"Started experiment: {EXPERIMENT_NAME}")

    print(f"Starting training for {MODEL_NAME}...")
    trainer.fit(model, datamodule_default_imagenet10k)
    print(f"Training complete. Best model saved to checkpoints/{os.path.basename(checkpoint_filename)}.ckpt")

    print(f"Finished experiment: {EXPERIMENT_NAME}")
    print("="*30)

def experiment2():
    """
        Train a basic DCAL 2018 on DF2K..
    """
    EXPERIMENT_NAME = "dcal_df2k"
    MODEL_NAME = "DCAL_2018"
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    
    model = get_model(MODEL_NAME, learning_rate=LEARNING_RATE)

    checkpoint_filename = f"{EXPERIMENT_NAME}-{MODEL_NAME}-best"

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=checkpoint_filename,
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        callbacks=[checkpoint_callback],
    )

    print("="*30)
    print(f"Started experiment: {EXPERIMENT_NAME}")

    print(f"Starting training for {MODEL_NAME}...")
    trainer.fit(model, datamodule_df2k)
    print(f"Training complete. Best model saved to checkpoints/{os.path.basename(checkpoint_filename)}")

    print(f"Finished experiment: {EXPERIMENT_NAME}")
    print("="*30)

def main():
   # pass
    experiment1()
    #experiment2()

if __name__ == "__main__":
    main()
