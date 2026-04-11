import os

import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import save_image
from PIL import Image, ImageOps

from data import ImageNetSubsetDataModule, ImageNet10KDataModule
from models import get_model

MODEL = "basic"
CKPT = "checkpoints/basic-best.ckpt"
DATA_DIR = "../datasets/imagenet_subtrain"
NUM_IMAGES = 8
OUTPUT_DIR = "outputs"

def non_overlaping_patches(img):
    """
        Creates non-overlapping patches of the image.
        Returns:
            list: list of tuples ((x,y), image_data)
    """
    img = Image.open("../datasets/misc/div2k_greece.png").convert("RGB")

    patch_size = 256

    # pad to create non-overlaping patches
    pad_w = (patch_size - img.size[0] % patch_size) % patch_size
    pad_h = (patch_size - img.size[1] % patch_size) % patch_size

    img = ImageOps.expand(img, (0, 0, pad_w, pad_h))

    patches = []

    for y in range(0, img.size[1], patch_size):
        for x in range(0, img.size[0], patch_size):
            patch_img = img.crop((x, y, x + patch_size, y + patch_size))
            patch = ((x,y), patch_img)
            patches.append(patch)

    return patches

def eval_compression():
    img = Image.open("../datasets/misc/div2k_greece.png").convert("RGB")
    patches =  non_overlaping_patches(img)

    print(f"Loading {MODEL} from {CKPT}...")
    model_class = get_model(MODEL).__class__
    model = model_class.load_from_checkpoint(CKPT)

    model.eval()
    model.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    from torchvision import transforms

    transform = transforms.ToTensor()
    batch = torch.stack([transform(patch) for _, patch in patches]).to(device)

    with torch.no_grad():
        reconstructions = model(batch)

    reconstructed = Image.new("RGB", img.size)
    for i, (x, y) in enumerate(pos for pos, _ in patches):
        patch = transforms.ToPILImage()(reconstructions[i].cpu())
        reconstructed.paste(patch, (x, y))

    reconstructed.save(f"{OUTPUT_DIR}/reconstructed.png")


def eval_patches():
    os.makedirs("outputs", exist_ok=True)

    datamodule = ImageNet10KDataModule(
        data_dir=DATA_DIR,
        batch_size=NUM_IMAGES
    )
    datamodule.setup()
    val_loader = datamodule.val_dataloader()

    print(f"Loading {MODEL} from {CKPT}...")
    model_class = get_model(MODEL).__class__
    model = model_class.load_from_checkpoint(CKPT)

    model.eval()
    model.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_mse = 0.0
    num_batches = 0

    first_batch_originals = torch.empty(0)
    first_batch_reconstructions = torch.empty(0)

    print("Evaluating model over the validation set...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            originals = batch.to(device)
            reconstructions = model(originals)

            total_mse += F.mse_loss(reconstructions, originals).item()
            psnr_metric.update(reconstructions, originals)
            ssim_metric.update(reconstructions, originals)
            num_batches += 1

            if i == 0:
                first_batch_originals = originals
                first_batch_reconstructions = reconstructions

    avg_mse = total_mse / num_batches
    avg_psnr = psnr_metric.compute()
    avg_ssim = ssim_metric.compute()

    print("\n" + "=" * 30)
    print(f"MSE:  {avg_mse:.6f}")
    print(f"PSNR: {avg_psnr:.2f} dB")
    print(f"SSIM: {avg_ssim:.4f}")
    print("=" * 30 + "\n")

    comparison = torch.cat([first_batch_originals, first_batch_reconstructions])
    save_path = f"outputs/{MODEL}_comparison.png"
    save_image(comparison, save_path, nrow=NUM_IMAGES)
    print(f"Image saved to {save_path}")
 
def main():
    eval_compression()
    return
    eval_patches()

if __name__ == "__main__":
    main()
