#!/usr/bin/env python3
"""
Save visualization outputs from Curia model.
Creates attention map overlays and saves as images.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom
from curia_tester import CuriaTester, CuriaConfig, MedicalImageLoader

# Class labels for anatomy heads (from TotalSegmentator)
ANATOMY_MRI_LABELS = {
    0: "Spleen", 1: "Kidney (right)", 2: "Kidney (left)", 3: "Gallbladder",
    4: "Liver", 5: "Stomach", 6: "Pancreas", 7: "Adrenal gland (right)",
    8: "Adrenal gland (left)", 9: "Lung (left)", 10: "Lung (right)",
    11: "Esophagus", 12: "Small bowel", 13: "Duodenum", 14: "Colon",
    15: "Urinary bladder", 16: "Prostate", 17: "Sacrum", 18: "Vertebrae",
    19: "Intervertebral discs", 20: "Spinal cord", 21: "Heart", 22: "Aorta",
    23: "Inferior vena cava", 24: "Portal/splenic vein", 25: "Iliac artery (left)",
    26: "Iliac artery (right)", 27: "Iliac vena (left)", 28: "Iliac vena (right)",
    29: "Humerus (left)", 30: "Humerus (right)", 31: "Fibula", 32: "Tibia",
    33: "Femur (left)", 34: "Femur (right)", 35: "Hip (left)", 36: "Hip (right)",
    37: "Gluteus maximus (left)", 38: "Gluteus maximus (right)",
    39: "Gluteus medius (left)", 40: "Gluteus medius (right)",
    41: "Gluteus minimus (left)", 42: "Gluteus minimus (right)",
    43: "Autochthon (left)", 44: "Autochthon (right)",
    45: "Iliopsoas (left)", 46: "Iliopsoas (right)",
    47: "Quadriceps femoris (left)", 48: "Quadriceps femoris (right)",
    49: "Thigh medial (left)", 50: "Thigh medial (right)",
    51: "Thigh posterior (left)", 52: "Thigh posterior (right)",
    53: "Sartorius (left)", 54: "Sartorius (right)", 55: "Brain"
}

ANATOMY_CT_LABELS = {
    0: "Spleen", 1: "Kidney (right)", 2: "Kidney (left)", 3: "Gallbladder",
    4: "Liver", 5: "Stomach", 6: "Pancreas", 7: "Adrenal gland (right)",
    8: "Adrenal gland (left)", 9: "Lung upper (left)", 10: "Lung lower (left)",
    11: "Lung upper (right)", 12: "Lung middle (right)", 13: "Lung lower (right)",
    14: "Esophagus", 15: "Trachea", 16: "Thyroid", 17: "Small bowel",
    18: "Duodenum", 19: "Colon", 20: "Urinary bladder", 21: "Prostate",
    22: "Kidney cyst (left)", 23: "Kidney cyst (right)", 24: "Sacrum",
    25: "Vertebrae", 26: "Intervertebral discs", 27: "Spinal cord",
    28: "Heart", 29: "Aorta", 30: "Pulmonary vein", 31: "Brachiocephalic trunk",
    32: "Subclavian artery (left)", 33: "Subclavian artery (right)",
    34: "Common carotid (left)", 35: "Common carotid (right)",
    36: "Brachiocephalic vein (left)", 37: "Brachiocephalic vein (right)",
    38: "Atrial appendage (left)", 39: "Superior vena cava", 40: "Inferior vena cava",
    41: "Portal/splenic vein", 42: "Iliac artery (left)", 43: "Iliac artery (right)",
    44: "Iliac vena (left)", 45: "Iliac vena (right)", 46: "Humerus (left)",
    47: "Humerus (right)", 48: "Scapula (left)", 49: "Scapula (right)",
    50: "Clavicula (left)", 51: "Clavicula (right)", 52: "Femur (left)",
    53: "Femur (right)", 54: "Hip (left)", 55: "Hip (right)"
}

# OASIS brain MRI labels (Binary dementia classification)
OASIS_LABELS = {
    0: "Non-demented",
    1: "Demented"
}

# ICH (Intracranial Hemorrhage) labels - Binary detection
ICH_LABELS = {
    0: "No hemorrhage",
    1: "Hemorrhage present"
}

# IXI brain tissue labels (single class output)
IXI_LABELS = {
    0: "Brain tissue"
}

# ATLAS stroke detection labels
ATLAS_STROKE_LABELS = {
    0: "No stroke",
    1: "Stroke lesion present"
}

# Spine stenosis labels (3-class severity)
SPINE_STENOSIS_LABELS = {
    0: "Normal",
    1: "Moderate",
    2: "Severe"
}

# KneeMRI labels (3-class)
KNEE_MRI_LABELS = {
    0: "Normal",
    1: "Partial tear",
    2: "Complete tear"
}

# COVID-CT labels (3-class)
COVID_CT_LABELS = {
    0: "Normal/Healthy",
    1: "COVID-19 pneumonia",
    2: "Non-COVID pneumonia"
}

# KITS kidney tumor labels (binary)
KITS_LABELS = {
    0: "No tumor",
    1: "Tumor present"
}

# Luna16 lung nodule labels (binary)
LUNA16_LABELS = {
    0: "No nodule",
    1: "Nodule present"
}

# Deep lesion site labels (8 anatomical regions)
DEEP_LESION_LABELS = {
    0: "Bone", 1: "Abdomen", 2: "Mediastinum", 3: "Liver",
    4: "Lung", 5: "Kidney", 6: "Soft tissue", 7: "Pelvis"
}

def get_label(head: str, class_idx: int) -> str:
    """Get human-readable label for class index."""
    label_maps = {
        "anatomy-mri": ANATOMY_MRI_LABELS,
        "anatomy-ct": ANATOMY_CT_LABELS,
        "oasis": OASIS_LABELS,
        "ich": ICH_LABELS,
        "ixi": IXI_LABELS,
        "atlas-stroke": ATLAS_STROKE_LABELS,
        "neural_foraminal_narrowing": SPINE_STENOSIS_LABELS,
        "spinal_canal_stenosis": SPINE_STENOSIS_LABELS,
        "subarticular_stenosis": SPINE_STENOSIS_LABELS,
        "kneeMRI": KNEE_MRI_LABELS,
        "covidx-ct": COVID_CT_LABELS,
        "kits": KITS_LABELS,
        "luna16-3D": LUNA16_LABELS,
        "deep-lesion-site": DEEP_LESION_LABELS,
    }
    label_map = label_maps.get(head, {})
    return label_map.get(class_idx, f"Class {class_idx}")


def save_attention_overlay(
    image_path: str,
    output_dir: str = "outputs",
    head: str = "anatomy-mri",
    slice_idx: int = None
):
    """
    Load image, run model, and save attention map overlay.

    Args:
        image_path: Path to NIfTI/DICOM file
        output_dir: Directory to save outputs
        head: Classification head to use
        slice_idx: Slice index for 3D volumes (None = middle slice)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load image
    print(f"Loading image: {image_path}")
    loader = MedicalImageLoader()
    image, metadata = loader.load(image_path)

    # Select slice if 3D
    if image.ndim == 3:
        if slice_idx is None:
            slice_idx = image.shape[-1] // 2
        img_slice = image[:, :, slice_idx]
        print(f"Using slice {slice_idx}/{image.shape[-1]}")
    else:
        img_slice = image
        slice_idx = 0

    # Load model
    print(f"Loading model with head: {head}")
    config = CuriaConfig(subfolder=head)
    tester = CuriaTester(config)
    tester.load_model(load_classifier=True)

    # Get classification
    print("Running classification...")
    result = tester.classify(img_slice)
    pred_class = result['prediction']
    probs = result['probabilities'][0]
    top_prob = probs[pred_class]
    pred_label = get_label(head, pred_class)

    print(f"Prediction: {pred_label} (prob: {top_prob:.4f})")

    # Get attention map
    print("Generating attention map...")
    attn_map = None
    try:
        attn_map = tester.get_attention_map(img_slice)
        # Resize attention map to match image size
        if attn_map.shape != img_slice.shape:
            zoom_factors = (img_slice.shape[0] / attn_map.shape[0],
                          img_slice.shape[1] / attn_map.shape[1])
            attn_map = zoom(attn_map, zoom_factors, order=1)
    except Exception as e:
        print(f"Could not get attention map: {e}")

    # Create visualization
    fig, axes = plt.subplots(1, 3 if attn_map is not None else 2, figsize=(15, 5))

    # Original image
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Classification result - show top 5 with labels
    top_indices = np.argsort(probs)[::-1][:10]
    top_probs = probs[top_indices]
    top_labels = [get_label(head, i)[:15] for i in top_indices]  # Truncate labels

    bars = axes[1].barh(range(len(top_indices)), top_probs[::-1])
    axes[1].set_yticks(range(len(top_indices)))
    axes[1].set_yticklabels(top_labels[::-1], fontsize=8)
    axes[1].set_xlabel('Probability')
    axes[1].set_title(f'Prediction: {pred_label}\n({top_prob:.1%})')

    # Highlight the prediction
    bars[-1].set_color('red')

    # Attention overlay
    if attn_map is not None:
        axes[2].imshow(img_slice, cmap='gray')
        # Normalize attention for overlay
        attn_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        axes[2].imshow(attn_norm, cmap='hot', alpha=0.5)
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')

    plt.tight_layout()

    # Save figure
    output_path = output_dir / f"visualization_{Path(image_path).stem}_slice{slice_idx}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Also save attention map as numpy
    if attn_map is not None:
        attn_path = output_dir / f"attention_{Path(image_path).stem}_slice{slice_idx}.npy"
        np.save(attn_path, attn_map)
        print(f"Saved: {attn_path}")

    plt.close()

    return {
        'prediction': pred_class,
        'label': pred_label,
        'probability': top_prob,
        'output_path': str(output_path)
    }


def process_volume(
    image_path: str,
    output_dir: str = "outputs",
    head: str = "anatomy-mri",
    num_slices: int = 5
):
    """
    Process multiple slices from a 3D volume.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load image
    loader = MedicalImageLoader()
    image, _ = loader.load(image_path)

    if image.ndim != 3:
        print("Image is 2D, processing single slice")
        return [save_attention_overlay(image_path, output_dir, head)]

    # Select slices uniformly
    total_slices = image.shape[-1]
    indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)

    print(f"Processing {num_slices} slices: {indices.tolist()}")

    results = []
    for idx in indices:
        result = save_attention_overlay(image_path, output_dir, head, slice_idx=idx)
        results.append(result)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Save Curia visualization outputs")
    parser.add_argument("image", help="Path to medical image (NIfTI/DICOM)")
    parser.add_argument("--head", default="anatomy-mri", help="Classification head")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--slice", type=int, default=None, help="Slice index (3D volumes)")
    parser.add_argument("--multi", action="store_true", help="Process multiple slices")
    parser.add_argument("--num-slices", type=int, default=5, help="Number of slices for --multi")

    args = parser.parse_args()

    if args.multi:
        results = process_volume(args.image, args.output, args.head, args.num_slices)
    else:
        results = save_attention_overlay(args.image, args.output, args.head, args.slice)

    print("\nDone!")
