"""
Example usage of CURIA Tester
=============================
This script demonstrates various ways to use the CURIA testing interface.
"""

import os
import numpy as np

# Set HF token if needed (uncomment and set your token)
# os.environ["HF_TOKEN"] = "your_token_here"


def example_basic_feature_extraction():
    """Example: Basic feature extraction from an image."""
    from curia_tester import CuriaTester, CuriaConfig

    print("\n" + "="*60)
    print("Example 1: Basic Feature Extraction")
    print("="*60)

    # Create tester with default config
    config = CuriaConfig()
    tester = CuriaTester(config)

    # Load the backbone model (no classification head)
    tester.load_model(load_classifier=False)

    # Create a dummy image (in practice, load from DICOM/NIfTI)
    dummy_image = np.random.rand(256, 256).astype(np.float32) * 2000 - 1000  # CT-like range

    # Extract features
    features = tester.extract_features(dummy_image)

    print(f"CLS token shape: {features['last_hidden_state'][:, 0].shape}")
    print(f"Patch tokens shape: {features['last_hidden_state'][:, 1:].shape}")


def example_classification():
    """Example: Image classification with a pretrained head."""
    from curia_tester import CuriaTester, CuriaConfig

    print("\n" + "="*60)
    print("Example 2: Classification with Pretrained Head")
    print("="*60)

    # Configure with a classification head
    config = CuriaConfig(
        subfolder="luna16-3D"  # Lung nodule detection head
    )
    tester = CuriaTester(config)

    # Load model with classifier
    tester.load_model(load_classifier=True)

    # Create a dummy CT slice
    dummy_ct = np.random.rand(512, 512).astype(np.float32) * 2000 - 1000

    # Classify
    result = tester.classify(dummy_ct)

    print(f"Prediction: {result['prediction']}")
    print(f"Probabilities: {result['probabilities']}")


def example_mask_prompting():
    """Example: Feature extraction with mask prompting."""
    from curia_tester import CuriaTester, CuriaConfig, MaskPrompt

    print("\n" + "="*60)
    print("Example 3: Mask-Guided Feature Extraction")
    print("="*60)

    config = CuriaConfig()
    tester = CuriaTester(config)
    tester.load_model(load_classifier=False)

    # Create image
    image = np.random.rand(256, 256).astype(np.float32)

    # Create mask using point prompts
    mask_prompt = MaskPrompt()
    mask = mask_prompt.create_point_mask(
        image_shape=(256, 256),
        points=[(128, 128), (100, 100)],
        radius=20
    )

    print(f"Mask shape: {mask.shape}")
    print(f"Mask coverage: {mask.sum() / mask.size * 100:.2f}%")

    # Extract features with mask
    features = tester.extract_features(image, mask=mask)

    if "mask_features" in features:
        print(f"Mask-guided features shape: {features['mask_features'].shape}")


def example_box_prompting():
    """Example: Feature extraction with bounding box prompts."""
    from curia_tester import CuriaTester, CuriaConfig, MaskPrompt

    print("\n" + "="*60)
    print("Example 4: Box Prompt Feature Extraction")
    print("="*60)

    config = CuriaConfig()
    tester = CuriaTester(config)
    tester.load_model(load_classifier=False)

    # Create image
    image = np.random.rand(512, 512).astype(np.float32)

    # Create mask using box prompts
    mask_prompt = MaskPrompt()
    mask = mask_prompt.create_box_mask(
        image_shape=(512, 512),
        boxes=[(100, 100, 200, 200), (300, 300, 400, 400)]  # Two boxes
    )

    print(f"Box mask shape: {mask.shape}")
    print(f"Box mask coverage: {mask.sum() / mask.size * 100:.2f}%")

    # Extract features with mask
    features = tester.extract_features(image, mask=mask)
    print(f"Features extracted successfully")


def example_3d_volume():
    """Example: Process a 3D volume."""
    from curia_tester import CuriaTester, CuriaConfig

    print("\n" + "="*60)
    print("Example 5: 3D Volume Processing")
    print("="*60)

    config = CuriaConfig(
        subfolder="anatomy-ct"  # Anatomy classification head
    )
    tester = CuriaTester(config)
    tester.load_model(load_classifier=True)

    # Create a dummy 3D volume (Z, H, W)
    volume = np.random.rand(50, 256, 256).astype(np.float32) * 2000 - 1000

    # Process volume (processes specified slices)
    results = tester.process_volume(
        volume,
        slice_indices=[20, 25, 30]  # Process specific slices
    )

    print(f"Processed slices: {results['slice_indices']}")
    print(f"Per-slice features count: {len(results['per_slice_features'])}")
    print(f"Aggregated features shape: {results['aggregated_features'].shape}")

    if results['per_slice_predictions']:
        print("\nPer-slice predictions:")
        for i, pred in enumerate(results['per_slice_predictions']):
            print(f"  Slice {results['slice_indices'][i]}: class {pred['prediction']}")


def example_attention_visualization():
    """Example: Get attention maps for visualization."""
    from curia_tester import CuriaTester, CuriaConfig

    print("\n" + "="*60)
    print("Example 6: Attention Map Extraction")
    print("="*60)

    config = CuriaConfig()
    tester = CuriaTester(config)
    tester.load_model(load_classifier=False)

    # Create image
    image = np.random.rand(256, 256).astype(np.float32)

    try:
        # Get attention map
        attn_map = tester.get_attention_map(image, layer_idx=-1)

        print(f"Attention map shape: {attn_map.shape}")
        print(f"Attention range: [{attn_map.min():.4f}, {attn_map.max():.4f}]")

        # The attention map can be used as a pseudo-segmentation
        # by thresholding regions of high attention
        threshold = np.percentile(attn_map, 90)
        pseudo_seg = (attn_map > threshold).astype(np.float32)
        print(f"Pseudo-segmentation coverage: {pseudo_seg.sum() / pseudo_seg.size * 100:.2f}%")

    except Exception as e:
        print(f"Note: Attention extraction may not be available: {e}")


def example_load_medical_images():
    """Example: Loading different medical image formats."""
    from curia_tester import MedicalImageLoader, ImageOrientation

    print("\n" + "="*60)
    print("Example 7: Medical Image Loading")
    print("="*60)

    loader = MedicalImageLoader(orientation=ImageOrientation.AXIAL)

    # Check available libraries
    print("Available libraries:")
    print(f"  pydicom: {loader.has_pydicom}")
    print(f"  nibabel: {loader.has_nibabel}")
    print(f"  SimpleITK: {loader.has_simpleitk}")

    # Example paths (these won't exist, just for demonstration)
    print("\nSupported file formats:")
    print("  - DICOM: .dcm, directories containing DICOM files")
    print("  - NIfTI: .nii, .nii.gz")
    print("  - NumPy: .npy, .npz")
    print("  - Images: .png, .jpg (converted to grayscale)")


def example_classification_heads():
    """Example: List and use different classification heads."""
    from curia_tester import CuriaConfig

    print("\n" + "="*60)
    print("Example 8: Available Classification Heads")
    print("="*60)

    config = CuriaConfig()

    print("Available pretrained classification heads:")
    for head in config.AVAILABLE_HEADS:
        print(f"  - {head}")

    print("\nHead descriptions:")
    head_descriptions = {
        "anatomy-ct": "CT anatomical structure classification",
        "anatomy-mri": "MRI anatomical structure classification",
        "atlas-stroke": "Stroke lesion detection",
        "covidx-ct": "COVID-19 detection from CT",
        "deep-lesion-site": "Lesion site classification",
        "ich": "Intracranial hemorrhage detection",
        "kits": "Kidney tumor segmentation related",
        "luna16-3D": "Lung nodule classification (3D)",
        "oasis": "Brain MRI analysis",
        "spinal_canal_stenosis": "Spinal canal stenosis grading",
    }

    for head, desc in head_descriptions.items():
        print(f"  {head}: {desc}")


def example_command_line():
    """Example: Command-line usage patterns."""
    print("\n" + "="*60)
    print("Example 9: Command-Line Usage")
    print("="*60)

    print("""
Command-line examples:

# Interactive mode
python curia_tester.py --interactive

# Extract features from a DICOM file
python curia_tester.py --input scan.dcm --mode features --output features.npz

# Classify with lung nodule head
python curia_tester.py --input ct_scan.nii.gz --mode classify --head luna16-3D

# Use mask prompting
python curia_tester.py --input scan.dcm --mask tumor_mask.nii.gz --mode features

# Point prompting (specify center of region of interest)
python curia_tester.py --input scan.dcm --points "256,256;300,300" --mode features

# Box prompting
python curia_tester.py --input scan.dcm --boxes "100,100,200,200" --mode classify --head anatomy-ct

# Get attention map for visualization
python curia_tester.py --input scan.dcm --mode attention --output attention.npy

# Specify device
python curia_tester.py --input scan.dcm --mode features --device cuda

# Start GUI
python curia_gui.py
""")


def main():
    """Run all examples."""
    print("="*60)
    print("CURIA Tester Examples")
    print("="*60)
    print("\nNote: Some examples require the model to be downloaded.")
    print("Make sure you have set HF_TOKEN if the model is gated.\n")

    # Examples that don't require model loading
    example_load_medical_images()
    example_classification_heads()
    example_command_line()

    # Ask before running model examples
    response = input("\nRun examples that require model loading? (y/n): ")
    if response.lower() == 'y':
        try:
            example_basic_feature_extraction()
            example_mask_prompting()
            example_box_prompting()
            example_classification()
            example_3d_volume()
            example_attention_visualization()
        except Exception as e:
            print(f"\nError running model examples: {e}")
            print("Make sure you have the required dependencies installed:")
            print("  pip install -r requirements.txt")

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
