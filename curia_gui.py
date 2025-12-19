"""
CURIA GUI - Gradio-based interface for medical image testing
=============================================================
A web-based interface for testing CURIA vision foundation model.

Usage:
    python curia_gui.py

Then open http://localhost:7860 in your browser.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from huggingface_hub import HfFolder

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    print("Gradio not installed. Install with: pip install gradio")

from curia_tester import CuriaTester, CuriaConfig, MedicalImageLoader, MaskPrompt
from preprocessing import Preprocessor, Normalizer, SliceSelector, CT_WINDOWS


class CuriaGUI:
    """Gradio-based GUI for CURIA testing."""

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
        8: "Adrenal gland (left)", 9: "Lung upper lobe (left)", 10: "Lung lower lobe (left)",
        11: "Lung upper lobe (right)", 12: "Lung middle lobe (right)", 13: "Lung lower lobe (right)",
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

    # Head metadata: modality compatibility and number of classes (verified from model)
    HEAD_INFO = {
        "None": {"modality": "any", "num_classes": None, "description": "Feature extraction only"},
        "abdominal-trauma": {"modality": "CT", "num_classes": 2, "description": "Abdominal trauma detection (binary)"},
        "anatomy-ct": {"modality": "CT", "num_classes": 54, "description": "Organ/anatomy identification (54 CT classes)"},
        "anatomy-mri": {"modality": "MRI", "num_classes": 56, "description": "Organ/anatomy identification (56 MRI classes)"},
        "atlas-stroke": {"modality": "both", "num_classes": 2, "description": "Stroke detection (binary: no-stroke/stroke)"},
        "covidx-ct": {"modality": "CT", "num_classes": 3, "description": "COVID-19 detection (normal/COVID/non-COVID pneumonia)"},
        "deep-lesion-site": {"modality": "CT", "num_classes": 8, "description": "Lesion anatomical site (8 regions)"},
        "emidec-classification-mask": {"modality": "both", "num_classes": 2, "description": "Lesion classification with mask (binary)"},
        "ich": {"modality": "CT", "num_classes": 2, "description": "Intracranial hemorrhage detection (binary)"},
        "ixi": {"modality": "MRI", "num_classes": 1, "description": "Brain tissue analysis (single output)"},
        "kits": {"modality": "CT", "num_classes": 2, "description": "Kidney tumor detection (binary)"},
        "kneeMRI": {"modality": "MRI", "num_classes": 3, "description": "Knee pathology (normal/partial/complete tear)"},
        "luna16-3D": {"modality": "CT", "num_classes": 2, "description": "Lung nodule detection (binary)"},
        "neural_foraminal_narrowing": {"modality": "MRI", "num_classes": 3, "description": "Neural foraminal stenosis (normal/moderate/severe)"},
        "oasis": {"modality": "MRI", "num_classes": 2, "description": "Dementia classification (binary: non-demented/demented)"},
        "spinal_canal_stenosis": {"modality": "MRI", "num_classes": 3, "description": "Spinal canal stenosis (normal/moderate/severe)"},
        "subarticular_stenosis": {"modality": "MRI", "num_classes": 3, "description": "Subarticular stenosis (normal/moderate/severe)"},
    }

    def __init__(self):
        self.tester = None
        self.current_image = None
        self.current_volume = None  # Store full volume
        self.current_mask = None
        self.current_metadata = None
        self.config = CuriaConfig()
        self.preprocessor = Preprocessor()
        self.current_slice_idx = 0
        self.apply_normalize = True  # Track normalization state
        self.current_head = "None"
        self.detected_modality = None
        self.current_image_path = None  # Track image path for export naming
        self.last_feature_results = None  # Store last feature extraction results

        # Check for stored HF token
        self.stored_token = HfFolder.get_token()
        self.has_stored_token = self.stored_token is not None

    def get_head_info(self, head_name: str) -> str:
        """Get information about a classification head."""
        head_info = self.HEAD_INFO.get(head_name, {})
        modality = head_info.get("modality", "unknown")
        num_classes = head_info.get("num_classes", "N/A")
        description = head_info.get("description", "")

        info = f"📋 {head_name}\n"
        info += f"Modality: {modality} | Classes: {num_classes}\n"
        info += f"{description}"
        return info

    def load_model(self, head_name: str, device: str, hf_token: str) -> str:
        """Load CURIA model with specified configuration."""
        try:
            self.config.device = device
            # Use provided token, or fall back to stored token
            self.config.hf_token = hf_token if hf_token else self.stored_token
            self.config.subfolder = head_name if head_name != "None" else None
            self.current_head = head_name

            self.tester = CuriaTester(self.config)
            self.tester.load_model(load_classifier=(head_name != "None"))

            # Get head info
            head_info = self.HEAD_INFO.get(head_name, {})
            modality = head_info.get("modality", "unknown")
            num_classes = head_info.get("num_classes", "N/A")
            description = head_info.get("description", "")

            status = f"Model loaded on {self.tester.device}\n"
            status += f"Head: {head_name}\n"
            status += f"Modality: {modality} | Classes: {num_classes}\n"
            status += f"{description}"
            return status
        except Exception as e:
            return f"Error loading model: {str(e)}"

    def update_preprocessing(self, modality: str, ct_window: str, normalize: bool) -> str:
        """Update preprocessing settings."""
        try:
            self.apply_normalize = normalize  # Store normalization state
            self.preprocessor = Preprocessor(
                modality=modality.lower(),
                ct_window=ct_window.lower().replace(" ", "_"),
                target_size=(512, 512)
            )
            return f"Preprocessing: {modality}, window={ct_window}, normalize={normalize}"
        except Exception as e:
            return f"Error updating preprocessing: {str(e)}"

    def load_image(self, file) -> Tuple[Optional[np.ndarray], str, int]:
        """Load medical image from uploaded file."""
        if file is None:
            return None, "No file uploaded", 0

        try:
            loader = MedicalImageLoader()
            self.current_image, self.current_metadata = loader.load(file.name)
            self.current_image_path = file.name  # Store for export naming
            self.last_feature_results = None  # Reset features on new image

            # Store volume if 3D - format is (H, W, Z), slice along last axis
            if self.current_image.ndim == 3:
                self.current_volume = self.current_image.copy()
                num_slices = self.current_image.shape[-1]  # Last axis is slices
                self.current_slice_idx = num_slices // 2
                display_img = self.current_image[:, :, self.current_slice_idx]
            else:
                self.current_volume = None
                display_img = self.current_image.copy()
                num_slices = 1

            # Normalize for display
            display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min() + 1e-8)
            display_img = (display_img * 255).astype(np.uint8)

            info = f"Shape: {self.current_image.shape}\n"
            info += f"Range: [{self.current_image.min():.2f}, {self.current_image.max():.2f}]\n"
            if self.current_metadata:
                for k, v in list(self.current_metadata.items())[:5]:
                    info += f"{k}: {v}\n"

            # Auto-detect modality and store it
            detected = self.preprocessor.normalizer.detect_modality(self.current_image)
            self.detected_modality = detected.value.upper()
            info += f"Detected modality: {self.detected_modality}\n"

            # Check modality compatibility with current head
            if self.current_head != "None":
                head_info = self.HEAD_INFO.get(self.current_head, {})
                head_modality = head_info.get("modality", "any")
                if head_modality not in ["any", "both", self.detected_modality]:
                    info += f"\n⚠️ WARNING: Head '{self.current_head}' expects {head_modality}, but image appears to be {self.detected_modality}\n"

            return display_img, info, num_slices

        except Exception as e:
            return None, f"Error loading image: {str(e)}", 0

    def change_slice(self, slice_idx: int, apply_preprocessing: bool, ct_window: str) -> Tuple[Optional[np.ndarray], str]:
        """Change displayed slice for 3D volumes."""
        if self.current_volume is None:
            return None, "No 3D volume loaded"

        try:
            slice_idx = int(slice_idx)
            # Handle (H, W, Z) format - slice along last axis
            num_slices = self.current_volume.shape[-1]
            self.current_slice_idx = max(0, min(slice_idx, num_slices - 1))
            display_img = self.current_volume[:, :, self.current_slice_idx].copy()

            # Apply preprocessing if requested
            if apply_preprocessing:
                window_key = ct_window.lower().replace(" ", "_")
                if window_key in CT_WINDOWS:
                    self.preprocessor.normalizer.window = CT_WINDOWS[window_key]
                # Use the stored normalize state
                display_img = self.preprocessor.process_slice(
                    display_img,
                    normalize=self.apply_normalize,
                    resize=False
                )
                display_img = (display_img * 255).astype(np.uint8)
            else:
                display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min() + 1e-8)
                display_img = (display_img * 255).astype(np.uint8)

            return display_img, f"Slice {self.current_slice_idx}/{num_slices-1}"

        except Exception as e:
            return None, f"Error: {str(e)}"

    def auto_select_slices(self, method: str, num_slices: int) -> str:
        """Auto-select informative slices."""
        if self.current_volume is None:
            return "No 3D volume loaded"

        try:
            selector = SliceSelector(method=method.lower(), num_slices=int(num_slices))
            # Volume is (H, W, Z), slice along last axis
            indices = selector.select(self.current_volume, axis=-1)
            return f"Selected slices ({method}): {indices}"
        except Exception as e:
            return f"Error: {str(e)}"

    def load_mask(self, file) -> Tuple[Optional[np.ndarray], str]:
        """Load mask from uploaded file."""
        if file is None:
            self.current_mask = None
            return None, "Mask cleared"

        try:
            prompt = MaskPrompt()
            self.current_mask = prompt.load_mask(file.name)

            # Normalize for display - for 3D (H, W, Z), use middle slice on last axis
            display_mask = self.current_mask.copy()
            if display_mask.ndim == 3:
                display_mask = display_mask[:, :, display_mask.shape[-1] // 2]

            display_mask = (display_mask * 255).astype(np.uint8)

            return display_mask, f"Mask shape: {self.current_mask.shape}"

        except Exception as e:
            return None, f"Error loading mask: {str(e)}"

    def create_point_mask(self, image: np.ndarray, evt: gr.SelectData) -> Tuple[np.ndarray, str]:
        """Create mask from clicked points."""
        if self.current_image is None:
            return image, "Load an image first"

        try:
            x, y = evt.index
            prompt = MaskPrompt()
            # For 3D (H, W, Z), use shape[:2] to get (H, W)
            img_shape = self.current_image.shape[:2]

            if self.current_mask is None:
                self.current_mask = np.zeros(img_shape, dtype=np.float32)

            point_mask = prompt.create_point_mask(img_shape, [(x, y)], radius=15)
            self.current_mask = np.maximum(self.current_mask, point_mask)

            # Overlay mask on image for display
            display_img = image.copy()
            if display_img.ndim == 2:
                display_img = np.stack([display_img] * 3, axis=-1)

            mask_overlay = (self.current_mask * 255).astype(np.uint8)
            # Red overlay
            display_img[:, :, 0] = np.where(mask_overlay > 0, 255, display_img[:, :, 0])

            return display_img, f"Added point at ({x}, {y})"

        except Exception as e:
            return image, f"Error: {str(e)}"

    def extract_features(self) -> str:
        """Extract features from current image."""
        if self.tester is None:
            return "Load model first"
        if self.current_image is None:
            return "Load image first"

        try:
            if self.current_image.ndim == 3:
                self.last_feature_results = self.tester.process_volume(self.current_image, mask=self.current_mask)
                results = self.last_feature_results
                output = f"Processed {results['num_slices']} slices\n"
                output += f"Feature dimension: {results['feature_dim']}\n"
                output += f"Slice features shape: {results['slice_features'].shape}\n"
                output += f"Aggregation: mean, max, std available\n"
                output += f"\nReady to export as JSON or CSV"
                return output
            else:
                features = self.tester.extract_features(self.current_image, mask=self.current_mask)
                # Store for single slice
                self.last_feature_results = {
                    "num_slices": 1,
                    "feature_dim": 768,
                    "slice_indices": [self.current_slice_idx],
                    "slice_features": features["last_hidden_state"][:, 0].cpu().numpy(),
                    "aggregated_features": features["last_hidden_state"][:, 0].cpu().squeeze(),
                }
                output = f"Feature shape: {features['last_hidden_state'].shape}"
                if "mask_features" in features:
                    output += f"\nMask feature shape: {features['mask_features'].shape}"
                output += f"\nReady to export as JSON or CSV"
                return output

        except Exception as e:
            return f"Error: {str(e)}"

    def export_features_json(self) -> Tuple[Optional[str], str]:
        """Export features to JSON file."""
        if not hasattr(self, 'last_feature_results') or self.last_feature_results is None:
            return None, "No features to export. Run 'Extract Features' first."

        try:
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)

            timestamp = Path(self.current_image_path).stem if hasattr(self, 'current_image_path') else "features"
            output_path = output_dir / f"{timestamp}_features.json"

            self.tester.export_features_to_json(self.last_feature_results, str(output_path))
            return str(output_path), f"Exported to: {output_path}"
        except Exception as e:
            return None, f"Error: {str(e)}"

    def export_features_csv(self) -> Tuple[Optional[str], str]:
        """Export features to CSV file."""
        if not hasattr(self, 'last_feature_results') or self.last_feature_results is None:
            return None, "No features to export. Run 'Extract Features' first."

        try:
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)

            timestamp = Path(self.current_image_path).stem if hasattr(self, 'current_image_path') else "features"
            output_path = output_dir / f"{timestamp}_features.csv"

            self.tester.export_features_to_csv(self.last_feature_results, str(output_path))
            return str(output_path), f"Exported to: {output_path}"
        except Exception as e:
            return None, f"Error: {str(e)}"

    def get_class_label(self, class_idx: int) -> str:
        """Get human-readable label for a class index."""
        label_maps = {
            "anatomy-mri": self.ANATOMY_MRI_LABELS,
            "anatomy-ct": self.ANATOMY_CT_LABELS,
            "oasis": self.OASIS_LABELS,
            "ich": self.ICH_LABELS,
            "ixi": self.IXI_LABELS,
            "atlas-stroke": self.ATLAS_STROKE_LABELS,
            "neural_foraminal_narrowing": self.SPINE_STENOSIS_LABELS,
            "spinal_canal_stenosis": self.SPINE_STENOSIS_LABELS,
            "subarticular_stenosis": self.SPINE_STENOSIS_LABELS,
            "kneeMRI": self.KNEE_MRI_LABELS,
            "covidx-ct": self.COVID_CT_LABELS,
            "kits": self.KITS_LABELS,
            "luna16-3D": self.LUNA16_LABELS,
            "deep-lesion-site": self.DEEP_LESION_LABELS,
        }
        label_map = label_maps.get(self.current_head, {})
        return label_map.get(class_idx, f"Class {class_idx}")

    def classify(self) -> Tuple[str, Optional[np.ndarray], str]:
        """Classify current image and generate attention map (uses currently displayed slice for 3D)."""
        if self.tester is None:
            return "Load model first", None, ""
        if self.tester.classifier is None:
            return "No classification head loaded. Select a head and reload model.", None, ""
        if self.current_image is None:
            return "Load image first", None, ""

        try:
            img = self.current_image
            mask = self.current_mask

            slice_info = ""
            if img.ndim == 3:
                # Use currently displayed slice, not middle slice
                slice_idx = self.current_slice_idx
                img = img[:, :, slice_idx]
                if mask is not None and mask.ndim == 3:
                    mask = mask[:, :, slice_idx]
                slice_info = f" [Slice {slice_idx}/{self.current_image.shape[-1]-1}]"

            result = self.tester.classify(img, mask=mask)

            # Get head info
            head_info = self.HEAD_INFO.get(self.current_head, {})
            num_classes = head_info.get("num_classes", "N/A")

            # Get prediction label
            pred_class = result['prediction']
            pred_label = self.get_class_label(pred_class)

            output = f"Head: {self.current_head} ({num_classes} classes){slice_info}\n"
            output += f"Prediction: {pred_label}\n"
            output += f"\nTop 5 Predictions:\n"

            # Sort by probability and show top predictions with labels
            probs = result['probabilities'][0]
            sorted_indices = np.argsort(probs)[::-1]
            for idx in sorted_indices[:5]:  # Show top 5
                label = self.get_class_label(idx)
                bar = "█" * int(probs[idx] * 20)
                output += f"  {probs[idx]:.2%} {bar} {label}\n"

            # Generate attention map
            attn_overlay, attn_info = self._generate_attention_overlay(img)

            return output, attn_overlay, attn_info

        except Exception as e:
            return f"Error: {str(e)}", None, ""

    def _generate_attention_overlay(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        """Generate attention map overlay for a 2D image slice."""
        try:
            from scipy.ndimage import zoom

            attn_map = self.tester.get_attention_map(img)

            # Resize attention map to match image size
            if attn_map.shape != img.shape:
                zoom_factors = (img.shape[0] / attn_map.shape[0], img.shape[1] / attn_map.shape[1])
                attn_map = zoom(attn_map, zoom_factors, order=1)

            # Create overlay visualization
            # Normalize image for display
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

            # Normalize attention map
            attn_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

            # Create RGB overlay (image in grayscale, attention in red/yellow)
            overlay = np.zeros((*img.shape, 3), dtype=np.uint8)
            overlay[:, :, 0] = (img_norm * 200 + attn_norm * 55).clip(0, 255).astype(np.uint8)  # R
            overlay[:, :, 1] = (img_norm * 200 * (1 - attn_norm * 0.5)).clip(0, 255).astype(np.uint8)  # G
            overlay[:, :, 2] = (img_norm * 200 * (1 - attn_norm * 0.7)).clip(0, 255).astype(np.uint8)  # B

            return overlay, f"Attention map - size: {attn_map.shape}"

        except Exception as e:
            return None, f"Attention error: {str(e)}"

    def clear_mask(self) -> str:
        """Clear current mask."""
        self.current_mask = None
        return "Mask cleared"

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface."""
        head_choices = ["None"] + self.config.AVAILABLE_HEADS
        ct_window_choices = list(CT_WINDOWS.keys())

        with gr.Blocks(title="CURIA Medical Image Tester") as interface:
            gr.Markdown("# CURIA Medical Image Tester")
            gr.Markdown(
                "Test CURIA vision foundation model with DICOM and NIfTI images. "
                "Supports mask-based prompting for region-specific analysis."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Model Configuration")

                    head_dropdown = gr.Dropdown(
                        choices=head_choices,
                        value="None",
                        label="Classification Head"
                    )

                    head_info_text = gr.Textbox(
                        label="Head Info",
                        value="Select a head to see details",
                        interactive=False,
                        lines=3
                    )

                    device_dropdown = gr.Dropdown(
                        choices=["auto", "cuda", "mps", "cpu"],
                        value="auto",
                        label="Device"
                    )

                    token_label = "HuggingFace Token (using stored credentials ✓)" if self.has_stored_token else "HuggingFace Token (required)"
                    hf_token = gr.Textbox(
                        label=token_label,
                        type="password",
                        placeholder="Leave empty to use stored token" if self.has_stored_token else "Enter your HF token"
                    )

                    load_model_btn = gr.Button("Load Model", variant="primary")
                    model_status = gr.Textbox(label="Model Status", lines=4, interactive=False)

                    gr.Markdown("---")
                    gr.Markdown("## Preprocessing")

                    modality_dropdown = gr.Dropdown(
                        choices=["Auto", "CT", "MRI"],
                        value="Auto",
                        label="Modality"
                    )

                    ct_window_dropdown = gr.Dropdown(
                        choices=ct_window_choices,
                        value="soft_tissue",
                        label="CT Window"
                    )

                    apply_preprocessing = gr.Checkbox(
                        label="Apply Preprocessing",
                        value=True
                    )

                    preprocess_status = gr.Textbox(label="Preprocessing Status", interactive=False)

                with gr.Column(scale=2):
                    gr.Markdown("## Image Input")

                    with gr.Row():
                        image_upload = gr.File(
                            label="Upload Medical Image (DICOM/NIfTI/NPY)",
                            file_types=[".dcm", ".nii", ".gz", ".npy", ".npz"]
                        )
                        mask_upload = gr.File(
                            label="Upload Mask (optional)",
                            file_types=[".nii", ".gz", ".npy", ".npz", ".png"]
                        )

                    with gr.Row():
                        image_display = gr.Image(label="Image Preview", interactive=True)
                        mask_display = gr.Image(label="Mask Preview")

                    image_info = gr.Textbox(label="Image Info", interactive=False, lines=6)

                    gr.Markdown("### Slice Navigation (for 3D volumes)")
                    with gr.Row():
                        slice_slider = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=50,
                            step=1,
                            label="Slice Index"
                        )
                        slice_info = gr.Textbox(label="Slice Info", interactive=False)

                    with gr.Row():
                        slice_method = gr.Dropdown(
                            choices=["Uniform", "Content", "Center"],
                            value="Uniform",
                            label="Slice Selection Method"
                        )
                        num_slices = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=10,
                            step=1,
                            label="Number of Slices"
                        )
                        auto_select_btn = gr.Button("Auto-Select Slices")
                    selected_slices_info = gr.Textbox(label="Selected Slices", interactive=False)

            with gr.Row():
                gr.Markdown("## Analysis")

            with gr.Row():
                with gr.Column():
                    extract_btn = gr.Button("Extract Features", variant="primary")
                    features_output = gr.Textbox(label="Features", interactive=False, lines=5)
                    with gr.Row():
                        export_json_btn = gr.Button("Export JSON")
                        export_csv_btn = gr.Button("Export CSV")
                    export_status = gr.Textbox(label="Export Status", interactive=False, lines=1)
                    export_file = gr.File(label="Download", interactive=False)

                with gr.Column():
                    classify_btn = gr.Button("Classify", variant="primary")
                    classify_output = gr.Textbox(label="Classification", interactive=False, lines=8)
                    attention_display = gr.Image(label="Attention Map")
                    attention_info = gr.Textbox(label="Attention Info", interactive=False)

            with gr.Row():
                clear_mask_btn = gr.Button("Clear Mask")
                mask_status = gr.Textbox(label="Mask Status", interactive=False)

            gr.Markdown("---")
            gr.Markdown(
                "**Tips:**\n"
                "- **Classify** generates both prediction and attention map for current slice\n"
                "- Click on the image to add point prompts (creates a circular mask)\n"
                "- Use the slice slider to navigate 3D volumes, then click Classify\n"
                "- Select CT window preset for different anatomical regions (brain, lung, etc.)"
            )

            # Event handlers
            head_dropdown.change(
                fn=self.get_head_info,
                inputs=head_dropdown,
                outputs=head_info_text
            )

            load_model_btn.click(
                fn=self.load_model,
                inputs=[head_dropdown, device_dropdown, hf_token],
                outputs=model_status
            )

            # Preprocessing settings change
            modality_dropdown.change(
                fn=self.update_preprocessing,
                inputs=[modality_dropdown, ct_window_dropdown, apply_preprocessing],
                outputs=preprocess_status
            )

            ct_window_dropdown.change(
                fn=self.update_preprocessing,
                inputs=[modality_dropdown, ct_window_dropdown, apply_preprocessing],
                outputs=preprocess_status
            )

            # Image loading - now also updates slice slider
            def load_and_update_slider(file):
                img, info, num = self.load_image(file)
                return img, info, gr.update(maximum=max(1, num-1), value=num//2)

            image_upload.change(
                fn=load_and_update_slider,
                inputs=image_upload,
                outputs=[image_display, image_info, slice_slider]
            )

            # Slice navigation
            slice_slider.change(
                fn=self.change_slice,
                inputs=[slice_slider, apply_preprocessing, ct_window_dropdown],
                outputs=[image_display, slice_info]
            )

            # Auto-select slices
            auto_select_btn.click(
                fn=self.auto_select_slices,
                inputs=[slice_method, num_slices],
                outputs=selected_slices_info
            )

            mask_upload.change(
                fn=self.load_mask,
                inputs=mask_upload,
                outputs=[mask_display, mask_status]
            )

            image_display.select(
                fn=self.create_point_mask,
                inputs=image_display,
                outputs=[image_display, mask_status]
            )

            extract_btn.click(
                fn=self.extract_features,
                outputs=features_output
            )

            export_json_btn.click(
                fn=self.export_features_json,
                outputs=[export_file, export_status]
            )

            export_csv_btn.click(
                fn=self.export_features_csv,
                outputs=[export_file, export_status]
            )

            classify_btn.click(
                fn=self.classify,
                outputs=[classify_output, attention_display, attention_info]
            )

            clear_mask_btn.click(
                fn=self.clear_mask,
                outputs=mask_status
            )

        return interface


def main():
    if not HAS_GRADIO:
        print("Please install Gradio: pip install gradio")
        return

    gui = CuriaGUI()
    interface = gui.create_interface()

    print("Starting CURIA GUI...")
    print("Open http://localhost:7860 in your browser")

    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
