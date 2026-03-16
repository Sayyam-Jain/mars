"""
Author: Custom Script (based on MARS project structure)
Project: MARS - Mask Attention Refinement with Sequential Quadtree Nodes
Description: Training script for the MARS model on the CarDD dataset using
             COCO-format instance segmentation annotations. Uses Mask R-CNN
             backbone from MARSModel with built-in loss computation.

             Includes a live web dashboard (http://localhost:8085/dashboard.html)
             that shows loss curves and GT vs prediction comparisons.
License: MIT License
"""

import os
import sys
import time
import json
import random
import yaml
import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
from functools import partial
from threading import Thread
from http.server import HTTPServer, SimpleHTTPRequestHandler

from models import MARSModel
from cardd_dataset import CarDDDataset, cardd_collate_fn


# ====================================================================
# Category colors for visualization (one per CarDD damage category)
# ====================================================================
CATEGORY_COLORS = {
    1: ('#FF6B6B', 'dent'),
    2: ('#4ECDC4', 'scratch'),
    3: ('#FFE66D', 'crack'),
    4: ('#A78BFA', 'glass shatter'),
    5: ('#F97316', 'lamp broken'),
    6: ('#22D3EE', 'tire flat'),
}

DASHBOARD_DIR = 'dashboard_output'
DASHBOARD_PORT = 8085


# ====================================================================
# Dashboard & Visualization Functions
# ====================================================================

def start_dashboard_server(directory, port=DASHBOARD_PORT):
    """Start a simple HTTP server in a background thread to serve the dashboard."""
    abs_dir = os.path.abspath(directory)

    handler = partial(SimpleHTTPRequestHandler, directory=abs_dir)
    try:
        server = HTTPServer(('0.0.0.0', port), handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"\n{'='*60}")
        print(f"  📊 Dashboard: http://localhost:{port}/dashboard.html")
        print(f"{'='*60}\n")
        return server
    except OSError as e:
        print(f"  ⚠️  Could not start dashboard server on port {port}: {e}")
        print(f"  Dashboard images will still be saved to {abs_dir}/")
        return None


def save_metrics(metrics_history, total_epochs, output_dir):
    """Save training metrics to JSON for the dashboard to read."""
    data = {
        'total_epochs': total_epochs,
        'epochs': metrics_history,
    }
    path = os.path.join(output_dir, 'metrics.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def draw_annotations_on_axes(ax, image_np, boxes, labels, masks, scores=None,
                              title='', score_threshold=0.3):
    """
    Draw bounding boxes and semi-transparent masks on a matplotlib axes.

    Args:
        ax: matplotlib Axes object
        image_np: (H, W, 3) numpy array, float [0, 1] or uint8 [0, 255]
        boxes: (N, 4) numpy array [x1, y1, x2, y2]
        labels: (N,) numpy array of category IDs
        masks: (N, H, W) numpy array of binary masks
        scores: (N,) numpy array of confidence scores (None for GT)
        title: title for the axes
        score_threshold: minimum score to display (only for predictions)
    """
    if image_np.dtype != np.float32 and image_np.dtype != np.float64:
        image_np = image_np.astype(np.float32) / 255.0

    # Create overlay for masks
    overlay = image_np.copy()

    for i in range(len(boxes)):
        # Skip low-confidence predictions
        if scores is not None and scores[i] < score_threshold:
            continue

        cat_id = int(labels[i])
        hex_color, cat_name = CATEGORY_COLORS.get(cat_id, ('#FFFFFF', f'cls_{cat_id}'))

        # Convert hex to RGB float
        r = int(hex_color[1:3], 16) / 255.0
        g = int(hex_color[3:5], 16) / 255.0
        b = int(hex_color[5:7], 16) / 255.0
        rgb = (r, g, b)

        # Draw mask (semi-transparent overlay)
        if i < len(masks):
            mask = masks[i]
            if mask.ndim == 3:
                mask = mask[0]  # Remove channel dim if present
            mask_bool = mask > 0.5
            for c_idx, c_val in enumerate(rgb):
                overlay[:, :, c_idx] = np.where(
                    mask_bool,
                    overlay[:, :, c_idx] * 0.5 + c_val * 0.5,
                    overlay[:, :, c_idx]
                )

        # Draw bounding box
        x1, y1, x2, y2 = boxes[i]
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=rgb, facecolor='none'
        )
        ax.add_patch(rect)

        # Label text
        label_text = cat_name
        if scores is not None:
            label_text += f' {scores[i]:.2f}'

        ax.text(
            x1, max(y1 - 4, 0), label_text,
            fontsize=8, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=rgb, alpha=0.8, edgecolor='none')
        )

    ax.imshow(overlay)
    ax.set_title(title, fontsize=11, fontweight='bold', color='white', pad=8)
    ax.axis('off')


@torch.no_grad()
def generate_comparisons(model, val_dataset, device, epoch, output_dir,
                          num_samples=10, score_threshold=0.3):
    """
    Generate side-by-side GT vs Prediction comparison images.

    For each sample:
      - Left panel: original image with ground truth masks/boxes
      - Right panel: same image with predicted masks/boxes/scores
    """
    model.eval()

    # Get detection model for inference
    if hasattr(model, 'detection_model'):
        det_model = model.detection_model
    else:
        det_model = model
    det_model.eval()

    # Pick random samples
    indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))

    comp_dir = os.path.join(output_dir, 'comparisons')
    os.makedirs(comp_dir, exist_ok=True)

    for img_idx, dataset_idx in enumerate(indices):
        image_tensor, target = val_dataset[dataset_idx]
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

        # Ground truth
        gt_boxes = target['boxes'].cpu().numpy()
        gt_labels = target['labels'].cpu().numpy()
        gt_masks = target['masks'].cpu().numpy()

        # Run inference
        outputs = det_model([image_tensor.to(device)])
        pred = outputs[0]
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        pred_masks = pred['masks'].cpu().numpy()  # (N, 1, H, W)
        pred_masks = pred_masks.squeeze(1)  # (N, H, W)

        # Create side-by-side figure
        fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(20, 8))
        fig.patch.set_facecolor('#1a1d27')

        draw_annotations_on_axes(
            ax_gt, image_np.copy(), gt_boxes, gt_labels, gt_masks,
            scores=None, title='Ground Truth'
        )
        draw_annotations_on_axes(
            ax_pred, image_np.copy(), pred_boxes, pred_labels, pred_masks,
            scores=pred_scores, title='Predictions',
            score_threshold=score_threshold
        )

        # Add legend
        legend_patches = []
        for cat_id, (hex_color, cat_name) in CATEGORY_COLORS.items():
            r = int(hex_color[1:3], 16) / 255.0
            g = int(hex_color[3:5], 16) / 255.0
            b = int(hex_color[5:7], 16) / 255.0
            legend_patches.append(mpatches.Patch(color=(r, g, b), label=cat_name))

        fig.legend(
            handles=legend_patches, loc='lower center', ncol=6,
            fontsize=9, frameon=True, facecolor='#2a2d3a', edgecolor='#3a3d4a',
            labelcolor='white'
        )

        plt.tight_layout(rect=[0, 0.06, 1, 1])

        save_path = os.path.join(comp_dir, f'epoch_{epoch}_img_{img_idx}.png')
        fig.savefig(save_path, dpi=120, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)

    print(f"  📸 Saved {len(indices)} comparison images to {comp_dir}/")


# ====================================================================
# Model & Training Functions
# ====================================================================

def load_config(config_path='cardd_config.yaml'):
    """Load configuration from YAML file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, config_path)
    if not os.path.exists(full_path):
        full_path = config_path
    with open(full_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_model(config, device):
    """Build the MARSModel with Mask R-CNN backbone."""
    model = MARSModel(
        num_classes=config['num_classes'],
        backbone_name=config['backbone_name'],
        pretrained=config.get('pretrained', True),
        use_quadtree_attention=config.get('use_quadtree_attention', True),
        use_mask_transfiner=config.get('use_mask_transfiner', False),
    )
    model = model.to(device)
    return model


def build_optimizer(model, config):
    """Build optimizer from config."""
    opt_config = config.get('optimizer', {})
    opt_type = opt_config.get('type', 'SGD')
    params = [p for p in model.parameters() if p.requires_grad]

    if opt_type == 'SGD':
        optimizer = optim.SGD(
            params, lr=config['learning_rate'],
            momentum=opt_config.get('momentum', 0.9),
            weight_decay=opt_config.get('weight_decay', 0.0005),
        )
    elif opt_type == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['learning_rate'],
            weight_decay=opt_config.get('weight_decay', 0.0001),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_type}")
    return optimizer


def build_scheduler(optimizer, config):
    """Build learning rate scheduler from config."""
    sched_config = config.get('scheduler', {})
    sched_type = sched_config.get('type', 'StepLR')

    if sched_type == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config.get('step_size', 10),
            gamma=sched_config.get('gamma', 0.1),
        )
    elif sched_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'],
        )
    else:
        print(f"Warning: Unknown scheduler '{sched_type}', using StepLR")
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return scheduler


def get_detection_model(model):
    """Extract the underlying torchvision detection model from MARSModel."""
    if hasattr(model, 'detection_model'):
        return model.detection_model
    else:
        raise AttributeError(
            "MARSModel does not have a 'detection_model' attribute. "
            "Make sure backbone_name='maskrcnn' is used in the config."
        )


def train_one_epoch(model, data_loader, optimizer, device, epoch, config):
    """
    Train for one epoch. Returns (avg_loss, component_losses_dict).

    Mask R-CNN computes losses internally: loss_classifier, loss_box_reg,
    loss_mask, loss_objectness, loss_rpn_box_reg.
    """
    model.train()
    det_model = get_detection_model(model)
    total_loss = 0.0
    num_batches = 0
    log_interval = config.get('log_interval', 10)

    # Accumulate component losses for dashboard
    component_sums = {}

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = det_model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        num_batches += 1

        # Accumulate component losses
        for k, v in loss_dict.items():
            component_sums[k] = component_sums.get(k, 0.0) + v.item()

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            print(f"  Epoch [{epoch+1}] Batch [{batch_idx+1}/{len(data_loader)}] "
                  f"Loss: {losses.item():.4f} (avg: {avg_loss:.4f})")
            loss_str = " | ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(f"    Components: {loss_str}")

    avg_loss = total_loss / max(num_batches, 1)

    # Average component losses
    component_avgs = {k: v / max(num_batches, 1) for k, v in component_sums.items()}

    return avg_loss, component_avgs


@torch.no_grad()
def validate(model, data_loader, device):
    """Run validation and compute average loss."""
    model.train()  # Mask R-CNN only returns losses in training mode
    det_model = get_detection_model(model)
    total_loss = 0.0
    num_batches = 0
    component_sums = {}

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = det_model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        total_loss += losses.item()
        num_batches += 1

        for k, v in loss_dict.items():
            component_sums[k] = component_sums.get(k, 0.0) + v.item()

    avg_loss = total_loss / max(num_batches, 1)
    component_avgs = {k: v / max(num_batches, 1) for k, v in component_sums.items()}

    return avg_loss, component_avgs


# ====================================================================
# Main Training Loop
# ====================================================================

def main():
    # ----------------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------------
    config = load_config()
    print("=" * 60)
    print("MARS - Training on CarDD Dataset")
    print("=" * 60)
    print(f"Config: num_classes={config['num_classes']}, "
          f"backbone={config['backbone_name']}, "
          f"batch_size={config['batch_size']}, "
          f"lr={config['learning_rate']}, "
          f"epochs={config['epochs']}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ----------------------------------------------------------------
    # Dashboard setup
    # ----------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_dir = os.path.join(script_dir, DASHBOARD_DIR)
    os.makedirs(os.path.join(dashboard_dir, 'comparisons'), exist_ok=True)
    server = start_dashboard_server(dashboard_dir, DASHBOARD_PORT)

    # ----------------------------------------------------------------
    # Datasets & DataLoaders
    # ----------------------------------------------------------------
    dataset_cfg = config['dataset']

    print("Loading training dataset...")
    train_dataset = CarDDDataset(
        root_dir=dataset_cfg['train_images'],
        annotation_file=dataset_cfg['train_annotations'],
        input_size=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=cardd_collate_fn,
        pin_memory=True if device.type == 'cuda' else False,
    )

    # Validation dataset
    val_loader = None
    val_dataset = None
    if 'val_images' in dataset_cfg and 'val_annotations' in dataset_cfg:
        if os.path.exists(dataset_cfg['val_annotations']):
            print("Loading validation dataset...")
            val_dataset = CarDDDataset(
                root_dir=dataset_cfg['val_images'],
                annotation_file=dataset_cfg['val_annotations'],
                input_size=None,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=4,
                collate_fn=cardd_collate_fn,
                pin_memory=True if device.type == 'cuda' else False,
            )

    # ----------------------------------------------------------------
    # Model, Optimizer, Scheduler
    # ----------------------------------------------------------------
    print("\nBuilding model...")
    model = build_model(config, device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # ----------------------------------------------------------------
    # Training Loop
    # ----------------------------------------------------------------
    checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_name = config.get('checkpoint_name', 'mars_cardd_best_model.pth')
    best_val_loss = float('inf')
    save_every = config.get('save_every_epoch', 5)
    metrics_history = []

    print(f"\nStarting training for {config['epochs']} epochs...")
    print("-" * 60)

    for epoch in range(config['epochs']):
        epoch_start = time.time()

        # ---- Train ----
        train_loss, train_components = train_one_epoch(
            model, train_loader, optimizer, device, epoch, config
        )
        epoch_time = time.time() - epoch_start

        # ---- Validate ----
        val_loss = None
        val_components = {}
        if val_loader is not None:
            val_loss, val_components = validate(model, val_loader, device)

        # ---- Scheduler step ----
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ---- Logging ----
        log_msg = (f"Epoch [{epoch+1}/{config['epochs']}] "
                   f"Train Loss: {train_loss:.4f}")
        if val_loss is not None:
            log_msg += f" | Val Loss: {val_loss:.4f}"
        log_msg += f" | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
        print(log_msg)

        # ---- Save metrics for dashboard ----
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr,
            'time': epoch_time,
            'component_losses': train_components,
            'val_component_losses': val_components,
        }
        metrics_history.append(epoch_metrics)
        save_metrics(metrics_history, config['epochs'], dashboard_dir)

        # ---- Generate GT vs Prediction comparisons ----
        if val_dataset is not None:
            print(f"  Generating comparison images...")
            generate_comparisons(
                model, val_dataset, device,
                epoch=epoch + 1,
                output_dir=dashboard_dir,
                num_samples=5,
                score_threshold=0.3,
            )

        # ---- Save best model ----
        compare_loss = val_loss if val_loss is not None else train_loss
        if compare_loss < best_val_loss:
            best_val_loss = compare_loss
            best_path = os.path.join(checkpoint_dir, checkpoint_name)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_val_loss,
                'config': config,
            }, best_path)
            print(f"  -> Saved best model (loss={best_val_loss:.4f}) to {best_path}")

        # ---- Periodic checkpoint ----
        if (epoch + 1) % save_every == 0:
            periodic_path = os.path.join(
                checkpoint_dir, f"mars_cardd_epoch{epoch+1}.pth"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
            }, periodic_path)
            print(f"  -> Periodic checkpoint saved to {periodic_path}")

    print("=" * 60)
    print(f"Training complete! Best loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {os.path.join(checkpoint_dir, checkpoint_name)}")
    print(f"Dashboard: http://localhost:{DASHBOARD_PORT}/dashboard.html")


if __name__ == '__main__':
    main()
