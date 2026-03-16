"""
Author: Custom Script (based on MARS project structure)
Project: MARS - Mask Attention Refinement with Sequential Quadtree Nodes
Description: Custom PyTorch Dataset for loading the CarDD dataset in COCO format.
             Parses COCO JSON annotations and converts segmentation polygons to
             binary masks, returning data in the format expected by torchvision
             detection models (Mask R-CNN).
License: MIT License
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_utils
import torchvision.transforms.functional as F


class CarDDDataset(Dataset):
    """
    PyTorch Dataset for the CarDD (Car Damage Detection) dataset in COCO format.

    Each sample returns:
        image: (3, H, W) float tensor (normalized to [0, 1])
        target: dict with keys:
            - boxes:    (N, 4) float tensor, [x_min, y_min, x_max, y_max]
            - labels:   (N,) int64 tensor, category IDs
            - masks:    (N, H, W) uint8 tensor, binary instance masks
            - image_id: (1,) int64 tensor
            - area:     (N,) float tensor
            - iscrowd:  (N,) int64 tensor

    Args:
        root_dir (str): Path to the image directory (e.g., 'CarDD_COCO/train2017').
        annotation_file (str): Path to the COCO JSON annotation file.
        transforms (callable, optional): Transforms to apply to (image, target).
        input_size (int, optional): If set, resize images to (input_size, input_size).
    """

    def __init__(self, root_dir, annotation_file, transforms=None, input_size=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.input_size = input_size

        # Load COCO annotations
        self.coco = COCO(annotation_file)
        self.image_ids = list(sorted(self.coco.imgs.keys()))

        # Filter out images with no annotations
        self.image_ids = [
            img_id for img_id in self.image_ids
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]

        # Get category information
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.cat_names = [cat['name'] for cat in self.categories]

        print(f"CarDDDataset loaded: {len(self.image_ids)} images, "
              f"{len(self.categories)} categories: {self.cat_names}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Load image info
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])

        # Load image
        image = Image.open(img_path).convert("RGB")
        original_w, original_h = image.size

        # Load annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        # Parse annotations
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for ann in annotations:
            # Bounding box: COCO format is [x, y, width, height] -> convert to [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue  # Skip degenerate boxes

            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))

            # Convert segmentation to binary mask
            mask = self._ann_to_mask(ann, original_h, original_w)
            masks.append(mask)

        if len(boxes) == 0:
            # Handle edge case: no valid annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, original_h, original_w), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        # Resize if needed
        if self.input_size is not None:
            image, boxes, masks = self._resize(
                image, boxes, masks, original_w, original_h
            )

        # Convert image to tensor
        image = F.to_tensor(image)  # (3, H, W), float [0, 1]

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _ann_to_mask(self, ann, height, width):
        """Convert a COCO annotation's segmentation to a binary mask."""
        segmentation = ann['segmentation']

        if isinstance(segmentation, list):
            # Polygon format - convert to RLE then to mask
            rles = coco_mask_utils.frPyObjects(segmentation, height, width)
            rle = coco_mask_utils.merge(rles)
        elif isinstance(segmentation, dict):
            # Already in RLE format
            rle = segmentation
        else:
            raise ValueError(f"Unknown segmentation format: {type(segmentation)}")

        mask = coco_mask_utils.decode(rle)
        return mask

    def _resize(self, image, boxes, masks, orig_w, orig_h):
        """Resize image, boxes, and masks to self.input_size."""
        new_size = self.input_size
        image = image.resize((new_size, new_size), Image.BILINEAR)

        if len(boxes) > 0:
            # Scale bounding boxes
            scale_x = new_size / orig_w
            scale_y = new_size / orig_h
            boxes[:, 0] *= scale_x  # x1
            boxes[:, 1] *= scale_y  # y1
            boxes[:, 2] *= scale_x  # x2
            boxes[:, 3] *= scale_y  # y2

            # Resize masks
            masks_pil = []
            for i in range(masks.shape[0]):
                m = Image.fromarray(masks[i].numpy())
                m = m.resize((new_size, new_size), Image.NEAREST)
                masks_pil.append(torch.as_tensor(np.array(m), dtype=torch.uint8))
            masks = torch.stack(masks_pil)

        return image, boxes, masks

    def get_category_names(self):
        """Return the list of category names."""
        return self.cat_names

    def get_num_classes(self):
        """Return number of classes including background (class 0)."""
        return len(self.categories) + 1  # +1 for background


def cardd_collate_fn(batch):
    """
    Custom collate function for Mask R-CNN style models.

    Mask R-CNN expects a list of images and a list of target dicts,
    NOT stacked tensors. This collate function handles that.

    Args:
        batch: list of (image, target) tuples from CarDDDataset

    Returns:
        images: list of (3, H, W) tensors
        targets: list of target dicts
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
