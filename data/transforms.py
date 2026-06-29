"""
Augmentation pipelines for train and val/test splits.

Design principles:
- Training augmentations are clinically plausible (no extreme distortions).
- Validation / test use only resize + normalize (deterministic).
- ImageNet normalization stats used since BioMedCLIP ViT expects them.
"""

from torchvision import transforms


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """
    Return a torchvision Compose pipeline for the given split.

    Args:
        split: One of 'train', 'val', 'test'.
        image_size: Target square image size (default 224 for ViT).
    """
    normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    if split == "train":
        return transforms.Compose([
            # Upsample slightly then random crop — reduces border artifacts.
            transforms.Resize(int(image_size * 1.15)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                fill=0,
            ),
            transforms.ToTensor(),
            normalize,
        ])

    # val / test — deterministic
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


def denormalize(tensor):
    """
    Reverse ImageNet normalization for visualization (e.g., GradCAM overlay).

    Args:
        tensor: (C, H, W) float tensor.

    Returns:
        (C, H, W) float tensor with values in [0, 1].
    """
    import torch
    mean = torch.tensor(_IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device)
    std  = torch.tensor(_IMAGENET_STD,  dtype=tensor.dtype, device=tensor.device)
    return (tensor * std[:, None, None] + mean[:, None, None]).clamp(0.0, 1.0)
