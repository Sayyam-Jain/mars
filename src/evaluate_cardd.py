"""
Author: Custom Script (based on MARS project structure)
Project: MARS - Mask Attention Refinement with Sequential Quadtree Nodes
Description: Evaluation script for the MARS model trained on CarDD.
             Runs inference on val/test split and computes COCO evaluation
             metrics (AP, AP50, AP75, AR) using pycocotools.
License: MIT License
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from models import MARSModel
from cardd_dataset import CarDDDataset, cardd_collate_fn

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask_utils


def load_config(config_path='cardd_config.yaml'):
    """Load configuration from YAML file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, config_path)
    if not os.path.exists(full_path):
        full_path = config_path
    with open(full_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_model(config, device, checkpoint_path):
    """
    Build model and load checkpoint.

    Returns:
        model: MARSModel with loaded weights
    """
    model = MARSModel(
        num_classes=config['num_classes'],
        backbone_name=config['backbone_name'],
        pretrained=False,  # Don't load pretrained, we'll load checkpoint
        use_quadtree_attention=config.get('use_quadtree_attention', True),
        use_mask_transfiner=config.get('use_mask_transfiner', False),
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        print(f"  Best loss at save time: {checkpoint.get('best_loss', 'N/A')}")
    else:
        # Direct state_dict save
        model.load_state_dict(checkpoint)
        print("Loaded model state dict directly")

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, data_loader, device, score_threshold=0.05):
    """
    Run inference on dataset and collect predictions in COCO format.

    Args:
        model: trained MARSModel with maskrcnn backbone
        data_loader: DataLoader for evaluation dataset
        device: torch device
        score_threshold: minimum confidence score to keep predictions

    Returns:
        results: list of dicts in COCO results format
    """
    model.eval()

    # Access the detection model directly for inference
    if hasattr(model, 'detection_model'):
        det_model = model.detection_model
    else:
        det_model = model

    det_model.eval()
    results = []

    print("Running inference...")
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]

        # Mask R-CNN in eval mode returns list of dicts with
        # keys: boxes, labels, scores, masks
        outputs = det_model(images)

        for i, output in enumerate(outputs):
            image_id = targets[i]['image_id'].item()
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            masks = output['masks'].cpu().numpy()  # (N, 1, H, W)

            for j in range(len(scores)):
                if scores[j] < score_threshold:
                    continue

                # Convert mask to RLE format for COCO evaluation
                # Mask R-CNN outputs soft masks — threshold at 0.5
                binary_mask = (masks[j, 0] > 0.5).astype(np.uint8)
                rle = coco_mask_utils.encode(
                    np.asfortranarray(binary_mask)
                )
                rle['counts'] = rle['counts'].decode('utf-8')

                # Convert box from [x1, y1, x2, y2] to [x, y, w, h]
                x1, y1, x2, y2 = boxes[j]
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

                result = {
                    'image_id': int(image_id),
                    'category_id': int(labels[j]),
                    'bbox': bbox,
                    'score': float(scores[j]),
                    'segmentation': rle,
                }
                results.append(result)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(data_loader)} batches")

    print(f"Total predictions: {len(results)}")
    return results


def evaluate_coco(annotation_file, results, iou_type='segm'):
    """
    Run COCO evaluation.

    Args:
        annotation_file: path to ground truth COCO annotation JSON
        results: list of prediction dicts in COCO results format
        iou_type: 'segm' for mask evaluation, 'bbox' for box evaluation

    Returns:
        stats: COCO evaluation statistics
    """
    coco_gt = COCO(annotation_file)

    if len(results) == 0:
        print("WARNING: No predictions to evaluate!")
        return None

    # Load results into COCO format
    coco_dt = coco_gt.loadRes(results)

    # Run evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()

    print(f"\n{'='*60}")
    print(f"COCO Evaluation Results ({iou_type})")
    print(f"{'='*60}")
    coco_eval.summarize()

    # Per-category evaluation
    print(f"\n{'='*60}")
    print(f"Per-Category AP ({iou_type})")
    print(f"{'='*60}")

    categories = coco_gt.loadCats(coco_gt.getCatIds())
    for cat in categories:
        cat_id = cat['id']
        cat_name = cat['name']

        # Filter to this category
        coco_eval_cat = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval_cat.params.catIds = [cat_id]
        coco_eval_cat.evaluate()
        coco_eval_cat.accumulate()

        # Extract AP (index 0 = AP@[0.5:0.95])
        ap = coco_eval_cat.stats[0]
        ap50 = coco_eval_cat.stats[1]
        print(f"  {cat_name:20s}  AP: {ap:.4f}  AP50: {ap50:.4f}")

    return coco_eval.stats


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate MARS model on CarDD dataset (COCO metrics)"
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--split', type=str, default='val', choices=['val', 'test'],
        help='Dataset split to evaluate on (default: val)'
    )
    parser.add_argument(
        '--config', type=str, default='cardd_config.yaml',
        help='Path to config file (default: cardd_config.yaml)'
    )
    parser.add_argument(
        '--score-threshold', type=float, default=0.05,
        help='Minimum detection score threshold (default: 0.05)'
    )
    parser.add_argument(
        '--save-results', type=str, default=None,
        help='Path to save prediction results as JSON'
    )
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset_cfg = config['dataset']

    # Select split
    if args.split == 'val':
        images_dir = dataset_cfg['val_images']
        ann_file = dataset_cfg['val_annotations']
    else:
        images_dir = dataset_cfg['test_images']
        ann_file = dataset_cfg['test_annotations']

    print(f"Evaluating on {args.split} split")
    print(f"  Images: {images_dir}")
    print(f"  Annotations: {ann_file}")

    # ----------------------------------------------------------------
    # Dataset & DataLoader
    # ----------------------------------------------------------------
    eval_dataset = CarDDDataset(
        root_dir=images_dir,
        annotation_file=ann_file,
        input_size=None,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,  # Use batch_size=1 for evaluation to handle variable sizes
        shuffle=False,
        num_workers=2,
        collate_fn=cardd_collate_fn,
    )

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = build_model(config, device, args.checkpoint)

    # ----------------------------------------------------------------
    # Inference + Evaluation
    # ----------------------------------------------------------------
    results = run_inference(
        model, eval_loader, device,
        score_threshold=args.score_threshold,
    )

    # Save results if requested
    if args.save_results:
        os.makedirs(os.path.dirname(args.save_results) or '.', exist_ok=True)
        with open(args.save_results, 'w') as f:
            json.dump(results, f)
        print(f"Results saved to {args.save_results}")

    # Run COCO evaluation for both segmentation and bounding boxes
    print("\n--- Segmentation (Mask) Evaluation ---")
    segm_stats = evaluate_coco(ann_file, results, iou_type='segm')

    print("\n--- Bounding Box Evaluation ---")
    bbox_stats = evaluate_coco(ann_file, results, iou_type='bbox')

    # Summary
    if segm_stats is not None:
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"  Mask  AP@[.5:.95]: {segm_stats[0]:.4f}")
        print(f"  Mask  AP@0.50:     {segm_stats[1]:.4f}")
        print(f"  Mask  AP@0.75:     {segm_stats[2]:.4f}")
        if bbox_stats is not None:
            print(f"  BBox  AP@[.5:.95]: {bbox_stats[0]:.4f}")
            print(f"  BBox  AP@0.50:     {bbox_stats[1]:.4f}")
            print(f"  BBox  AP@0.75:     {bbox_stats[2]:.4f}")


if __name__ == '__main__':
    main()
