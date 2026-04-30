from penne._modules import SpaghettiGenerator
import torch
from torchvision.transforms import v2
from torch.nn import functional as F
import os

def init_spaghetti(model_path: str) -> torch.nn.Module:
    '''
    Initialize the SPAGHETTI model for image translation
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = SpaghettiGenerator(3, 9)
    generator.to(device)
    ckpt = torch.load(model_path, map_location=device)["state_dict"]
    # get only G_AB weights
    ckpt = {k[5:]: v for k, v in ckpt.items() if ("G_AB" in k)}
    generator.load_state_dict(ckpt)
    generator.eval()
    return generator

def pre_processing_phikon(model=None):
    class ResizeShortestEdge:
        def __init__(self, size, interpolation=v2.InterpolationMode.BICUBIC):
            self.size = size
            self.interpolation = interpolation

        def __call__(self, img: torch.Tensor):
            # img is (C, H, W)
            if img.dim() != 3:
                raise ValueError(f"Expected (C, H, W) input, but got {img.shape}")

            c, h, w = img.shape
            if h < w:
                new_h = self.size
                new_w = int(w * self.size / h)
            else:
                new_w = self.size
                new_h = int(h * self.size / w)

            img = img.unsqueeze(0)  # add batch dimension: (1, C, H, W)
            img = F.interpolate(
                img,
                size=(new_h, new_w),
                mode=self.interpolation.value.lower(),
                align_corners=False if self.interpolation in [v2.InterpolationMode.BILINEAR, v2.InterpolationMode.BICUBIC] else None
            )
            return img.squeeze(0)  # back to (C, H, W)
    if model: #if a model name is provided, use the corresponding image processor from HuggingFace
        from transformers import AutoImageProcessor
        image_processor = AutoImageProcessor.from_pretrained(model, use_fast=True)
        return lambda x: image_processor(x, return_tensors="pt", do_rescale=False)["pixel_values"]
    else: #implement the image processor for Phikon-v2 as described in the paper
        IMAGE_MEAN = [0.485, 0.456, 0.406]
        IMAGE_STD = [0.229, 0.224, 0.225]
        RESCALE_FACTOR = 0.00392156862745098  # = 1/255
        TARGET_SIZE = 224  # both resize shortest edge and crop
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            ResizeShortestEdge(TARGET_SIZE),                         # Resize shortest edge to 224
            v2.CenterCrop(TARGET_SIZE),                               # Center crop to 224x224
            v2.Lambda(lambda x: x * RESCALE_FACTOR),          # Rescale (1/255)
            v2.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),     # Normalize
        ])
        return lambda batch: torch.stack([transform(x) for x in batch])


def download_asset_from_github(asset_name: str, save_path: str) -> None:
    '''Download an asset from the penne GitHub repository.

    Args:
        asset_name (str): The name of the asset to download (e.g., "spaghetti.ckpt").
        save_path (str): The local path to save the downloaded asset.
    '''
    import requests
    save_dir = os.path.expanduser(save_path)
    os.makedirs(save_dir, exist_ok=True)
    f_save_path = os.path.join(save_dir, asset_name)
    url = f"https://raw.githubusercontent.com/schwartzlab-methods/penne/main/penne/assets/{asset_name}"
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_written = 0
        expected_length = response.headers.get("Content-Length")
        with open(f_save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                total_written += len(chunk)
        if expected_length is not None and total_written != int(expected_length):
            # Incomplete download; remove partial file and raise an error.
            try:
                os.remove(f_save_path)
            except OSError:
                pass
            raise Exception(
                f"Failed to download complete asset {asset_name} from GitHub. "
                f"Expected {expected_length} bytes, got {total_written} bytes."
            )
        print(f"Downloaded {asset_name} to {f_save_path}")
    except requests.RequestException as e:
        # Clean up any partially written file on network or HTTP error.
        if os.path.exists(f_save_path):
            try:
                os.remove(f_save_path)
            except OSError:
                pass
        raise Exception(f"Failed to download {asset_name} from GitHub: {e}")