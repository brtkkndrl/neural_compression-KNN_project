import os

import torch
from torchvision.utils import save_image

from data import ImageNetSubsetDataModule
from models import get_model

MODEL = "basic"
CKPT = "checkpoints/model.ckpt"
DATA_DIR = "datasets/imagenet-mini"
IMAGE_SIZE = 256
NUM_IMAGES = 8


def main():
    os.makedirs("outputs", exist_ok=True)

    datamodule = ImageNetSubsetDataModule(
        data_dir=DATA_DIR,
        batch_size=NUM_IMAGES,
        image_size=IMAGE_SIZE,
        val_limit=100
    )
    datamodule.setup()

    val_loader = datamodule.val_dataloader()
    originals = next(iter(val_loader))

    print(f"Loading {MODEL} from {CKPT}...")

    model_class = get_model(MODEL).__class__
    model = model_class.load_from_checkpoint(CKPT)

    model.eval()
    model.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    originals = originals.to(device)

    print("Generating reconstructions...")
    with torch.no_grad():
        reconstructions = model(originals)

    comparison = torch.cat([originals, reconstructions])

    save_path = f"outputs/{MODEL}_comparison.png"

    save_image(comparison, save_path, nrow=NUM_IMAGES)
    print(f"Image saved to {save_path}")


if __name__ == "__main__":
    main()
