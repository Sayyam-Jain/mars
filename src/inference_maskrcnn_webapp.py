"""
Project: MARS - Mask Attention Refinement with Sequential Quadtree Nodes
Description: Flask web application for car damage inference using Mask R-CNN.
             Users upload car images and receive annotated output with
             instance segmentation masks and bounding boxes.
License: MIT License
"""

import os
import io
import sys
import base64
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms as T
from flask import Flask, request, jsonify, render_template
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Add src directory to path so we can import models
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from models import MARSModel

# ====================================================================
# Configuration
# ====================================================================

PORT = 8085
SCORE_THRESHOLD = 0.3
# CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, 'checkpoints', 'mars_cardd_best_model.pth')
CHECKPOINT_PATH = '/app/checkpoints/mars_cardd_epoch40.pth'
# CHECKPOINT_PATH = '/app/checkpoints/mars_cardd_best_model.pth'
# CarDD damage categories (matching train_cardd.py)
CATEGORY_COLORS = {
    1: ('#FF6B6B', 'dent'),
    2: ('#4ECDC4', 'scratch'),
    3: ('#FFE66D', 'crack'),
    4: ('#A78BFA', 'glass shatter'),
    5: ('#F97316', 'lamp broken'),
    6: ('#22D3EE', 'tire flat'),
}

# COCO fallback categories (subset shown when using pretrained COCO weights)
COCO_CATEGORY_NAMES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush',
}

# ====================================================================
# Model Loading
# ====================================================================

def load_model(device):
    """Load the MARSModel with Mask R-CNN backbone."""
    using_cardd = False

    if os.path.exists(CHECKPOINT_PATH):
        print(f"  ✅ Found CarDD checkpoint: {CHECKPOINT_PATH}")
        # Load checkpoint to get config
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        config = checkpoint.get('config', {})
        num_classes = config.get('num_classes', 7)

        model = MARSModel(
            num_classes=num_classes,
            backbone_name='maskrcnn',
            pretrained=False,
            use_quadtree_attention=config.get('use_quadtree_attention', True),
            use_mask_transfiner=config.get('use_mask_transfiner', False),
            # Note: With maskrcnn backbone, inference runs through det_model
            # (the raw Mask R-CNN), not MARSModel.forward(), so the transfiner
            # module is not executed. We still load it to match the checkpoint
            # state_dict keys from training.
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        using_cardd = True
        print(f"  ✅ Loaded CarDD model with {num_classes} classes")
    # else:
    #     print(f"  ⚠️  No checkpoint found at {CHECKPOINT_PATH}")
    #     print(f"  ℹ️  Using COCO-pretrained Mask R-CNN (91 classes)")
    #     # Use pretrained COCO Mask R-CNN directly
    #     from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    #     model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    #     using_cardd = False

    model.to(device)
    model.eval()
    return model, using_cardd


def get_detection_model(model, using_cardd):
    """Get the underlying detection model for inference."""
    if using_cardd:
        if hasattr(model, 'detection_model'):
            return model.detection_model
        else:
            return model
    else:
        return model


# ====================================================================
# Inference & Visualization
# ====================================================================

def run_inference(model, det_model, image_pil, device, score_threshold=SCORE_THRESHOLD):
    """Run Mask R-CNN inference on a PIL image, return predictions dict."""
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image_pil).to(device)

    with torch.no_grad():
        outputs = det_model([image_tensor])

    pred = outputs[0]
    # Filter by score threshold
    keep = pred['scores'] >= score_threshold
    result = {
        'boxes': pred['boxes'][keep].cpu().numpy(),
        'labels': pred['labels'][keep].cpu().numpy(),
        'scores': pred['scores'][keep].cpu().numpy(),
        'masks': pred['masks'][keep].cpu().numpy(),  # (N, 1, H, W)
    }
    return result


def create_annotated_image(image_pil, predictions, using_cardd):
    """Create an annotated image with masks and bounding boxes."""
    image_np = np.array(image_pil).astype(np.float32) / 255.0
    H, W = image_np.shape[:2]

    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']
    masks = predictions['masks']

    if len(masks.shape) == 4:
        masks = masks.squeeze(1)  # (N, H, W)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(W / 80, H / 80), dpi=120)
    fig.patch.set_facecolor('#1a1d27')

    overlay = image_np.copy()

    # Generate colors for non-CarDD mode
    np.random.seed(42)
    random_colors = {}

    for i in range(len(boxes)):
        cat_id = int(labels[i])
        score = float(scores[i])

        if using_cardd:
            hex_color, cat_name = CATEGORY_COLORS.get(cat_id, ('#FFFFFF', f'class_{cat_id}'))
            r = int(hex_color[1:3], 16) / 255.0
            g = int(hex_color[3:5], 16) / 255.0
            b = int(hex_color[5:7], 16) / 255.0
        else:
            cat_name = COCO_CATEGORY_NAMES.get(cat_id, f'class_{cat_id}')
            if cat_id not in random_colors:
                random_colors[cat_id] = (
                    np.random.uniform(0.3, 1.0),
                    np.random.uniform(0.3, 1.0),
                    np.random.uniform(0.3, 1.0),
                )
            r, g, b = random_colors[cat_id]

        rgb = (r, g, b)

        # Draw mask
        if i < len(masks):
            mask = masks[i]
            mask_bool = mask > 0.5
            for c_idx, c_val in enumerate(rgb):
                overlay[:, :, c_idx] = np.where(
                    mask_bool,
                    overlay[:, :, c_idx] * 0.5 + c_val * 0.5,
                    overlay[:, :, c_idx],
                )

        # Draw bounding box
        x1, y1, x2, y2 = boxes[i]
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=rgb, facecolor='none',
        )
        ax.add_patch(rect)

        # Label text
        label_text = f'{cat_name} {score:.2f}'
        fontsize = max(7, min(12, int(W / 80)))
        ax.text(
            x1, max(y1 - 4, 0), label_text,
            fontsize=fontsize, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=rgb, alpha=0.85, edgecolor='none'),
        )

    ax.imshow(overlay)
    ax.axis('off')

    # Add legend
    if using_cardd:
        legend_patches = []
        for cat_id, (hex_color, cat_name) in CATEGORY_COLORS.items():
            r = int(hex_color[1:3], 16) / 255.0
            g = int(hex_color[3:5], 16) / 255.0
            b = int(hex_color[5:7], 16) / 255.0
            legend_patches.append(mpatches.Patch(color=(r, g, b), label=cat_name))
        if legend_patches:
            ax.legend(
                handles=legend_patches, loc='upper right', fontsize=max(7, int(W / 100)),
                frameon=True, facecolor='#2a2d3a', edgecolor='#3a3d4a',
                labelcolor='white', framealpha=0.9,
            )

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)

    return buf


# ====================================================================
# Flask App
# ====================================================================

TEMPLATE_DIR = os.path.join(SCRIPT_DIR, 'templates')
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Global model references (loaded at startup)
_model = None
_det_model = None
_device = None
_using_cardd = False


@app.route('/')
def index():
    """Serve the upload page."""
    global _using_cardd
    num_classes = 7 if _using_cardd else 91
    return render_template(
        'index.html',
        using_cardd=_using_cardd,
        num_classes=num_classes,
    )


@app.route('/predict', methods=['POST'])
def predict():
    """Run inference on uploaded image and return annotated result."""
    global _model, _det_model, _device, _using_cardd

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Read image
        image_bytes = file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Run inference
        predictions = run_inference(_model, _det_model, image_pil, _device)

        # Create annotated image
        result_buf = create_annotated_image(image_pil, predictions, _using_cardd)
        result_b64 = base64.b64encode(result_buf.read()).decode('utf-8')

        # Build category summary
        categories = {}
        for label_id in predictions['labels']:
            if _using_cardd:
                _, cat_name = CATEGORY_COLORS.get(int(label_id), ('', f'class_{label_id}'))
            else:
                cat_name = COCO_CATEGORY_NAMES.get(int(label_id), f'class_{label_id}')
            categories[cat_name] = categories.get(cat_name, 0) + 1

        avg_conf = float(np.mean(predictions['scores'])) if len(predictions['scores']) > 0 else 0.0

        return jsonify({
            'image': result_b64,
            'num_detections': int(len(predictions['boxes'])),
            'categories': categories,
            'avg_confidence': avg_conf,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def main():
    global _model, _det_model, _device, _using_cardd

    print("=" * 60)
    print("  MARS — Mask R-CNN Car Damage Inference Web App")
    print("=" * 60)

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {_device}")

    print("  Loading model...")
    _model, _using_cardd = load_model(_device)
    _det_model = get_detection_model(_model, _using_cardd)
    _det_model.eval()

    print(f"\n  🚀 Starting server on http://0.0.0.0:{PORT}")
    print(f"  Open http://localhost:{PORT} in your browser\n")

    app.run(host='0.0.0.0', port=PORT, debug=False)


if __name__ == '__main__':
    main()
