"""
CURIA Medical Image Tester
==========================
A comprehensive interface for testing CURIA vision foundation model
with DICOM and NIfTI medical images.

CURIA is a DINOv2-based foundation model for radiology that supports:
- Feature extraction from medical images
- Classification with various pretrained heads
- Mask-guided region analysis (prompting)

Note: CURIA is primarily a classification/feature extraction model.
For segmentation-like behavior, we use attention maps and mask prompting.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class ImageOrientation(Enum):
    """Medical image orientation types expected by CURIA."""
    AXIAL = "PL"      # Posterior-Left (axial slices)
    CORONAL = "IL"    # Inferior-Left (coronal slices)
    SAGITTAL = "IP"   # Inferior-Posterior (sagittal slices)


@dataclass
class CuriaConfig:
    """Configuration for CURIA model testing."""
    model_name: str = "raidium/curia"
    subfolder: Optional[str] = None  # e.g., "luna16-3D", "anatomy-ct"
    crop_size: int = 512
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    hf_token: Optional[str] = None

    # Available heads for classification
    AVAILABLE_HEADS: List[str] = field(default_factory=lambda: [
        "anatomy-ct", "anatomy-mri", "atlas-stroke", "covidx-ct",
        "deep-lesion-site", "emidec-classification-mask", "ich", "ixi",
        "kits", "kneeMRI", "luna16-3D", "neural_foraminal_narrowing",
        "oasis", "spinal_canal_stenosis", "subarticular_stenosis"
    ])


class MedicalImageLoader:
    """Loader for DICOM and NIfTI medical images."""

    def __init__(self, orientation: ImageOrientation = ImageOrientation.AXIAL):
        self.orientation = orientation
        self._check_dependencies()

    def _check_dependencies(self):
        """Check for required medical imaging libraries."""
        self.has_pydicom = False
        self.has_nibabel = False
        self.has_simpleitk = False

        try:
            import pydicom
            self.has_pydicom = True
        except ImportError:
            pass

        try:
            import nibabel
            self.has_nibabel = True
        except ImportError:
            pass

        try:
            import SimpleITK
            self.has_simpleitk = True
        except ImportError:
            pass

    def load_dicom(self, path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load a DICOM file or directory of DICOM files.

        Args:
            path: Path to DICOM file or directory

        Returns:
            Tuple of (image array, metadata dict)
        """
        path = Path(path)

        if self.has_simpleitk:
            return self._load_dicom_sitk(path)
        elif self.has_pydicom:
            return self._load_dicom_pydicom(path)
        else:
            raise ImportError(
                "No DICOM library available. Install with:\n"
                "  pip install pydicom\n"
                "  or pip install SimpleITK"
            )

    def _load_dicom_sitk(self, path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load DICOM using SimpleITK."""
        import SimpleITK as sitk

        if path.is_dir():
            reader = sitk.ImageSeriesReader()
            dicom_files = reader.GetGDCMSeriesFileNames(str(path))
            reader.SetFileNames(dicom_files)
            image = reader.Execute()
        else:
            image = sitk.ReadImage(str(path))

        array = sitk.GetArrayFromImage(image)  # (Z, Y, X) or (Y, X)

        # Transpose to (H, W, Z) for consistent format
        if array.ndim == 3:
            array = array.transpose(1, 2, 0)  # (Z,Y,X) -> (Y,X,Z) = (H,W,Z)

        metadata = {
            "spacing": image.GetSpacing(),
            "origin": image.GetOrigin(),
            "direction": image.GetDirection(),
            "size": image.GetSize(),
            "slice_axis": 2 if array.ndim == 3 else None,
        }

        return self._reorient_array(array), metadata

    def _load_dicom_pydicom(self, path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load DICOM using pydicom."""
        import pydicom

        if path.is_dir():
            # Load all DICOM files in directory
            slices = []
            for f in sorted(path.glob("*.dcm")) or sorted(path.glob("*")):
                try:
                    ds = pydicom.dcmread(str(f))
                    slices.append(ds)
                except:
                    continue

            if not slices:
                raise ValueError(f"No valid DICOM files found in {path}")

            # Sort by InstanceNumber or SliceLocation
            slices.sort(key=lambda x: float(getattr(x, 'SliceLocation', 0)))
            array = np.stack([s.pixel_array for s in slices])
            ds = slices[0]
        else:
            ds = pydicom.dcmread(str(path))
            array = ds.pixel_array

        # Apply rescale if available
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            array = array * ds.RescaleSlope + ds.RescaleIntercept

        # Transpose to (H, W, Z) for consistent format
        # pydicom stacks as (Z, H, W) for multi-slice
        if array.ndim == 3:
            array = array.transpose(1, 2, 0)  # (Z,H,W) -> (H,W,Z)

        metadata = {
            "modality": getattr(ds, 'Modality', 'Unknown'),
            "patient_id": getattr(ds, 'PatientID', 'Unknown'),
            "study_description": getattr(ds, 'StudyDescription', ''),
            "pixel_spacing": getattr(ds, 'PixelSpacing', None),
            "slice_axis": 2 if array.ndim == 3 else None,
        }

        return self._reorient_array(array.astype(np.float32)), metadata

    def load_nifti(self, path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load a NIfTI file.

        Args:
            path: Path to NIfTI file (.nii or .nii.gz)

        Returns:
            Tuple of (image array, metadata dict)
        """
        if self.has_simpleitk:
            return self._load_nifti_sitk(path)
        elif self.has_nibabel:
            return self._load_nifti_nibabel(path)
        else:
            raise ImportError(
                "No NIfTI library available. Install with:\n"
                "  pip install nibabel\n"
                "  or pip install SimpleITK"
            )

    def _load_nifti_sitk(self, path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load NIfTI using SimpleITK."""
        import SimpleITK as sitk

        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image)  # Returns (Z, Y, X)

        # Transpose to (H, W, Z) = (Y, X, Z) for consistent format
        # Processor expects slice dimension last
        if array.ndim == 3:
            array = array.transpose(1, 2, 0)  # (Z,Y,X) -> (Y,X,Z) = (H,W,Z)

        metadata = {
            "spacing": image.GetSpacing(),
            "origin": image.GetOrigin(),
            "direction": image.GetDirection(),
            "size": image.GetSize(),
            "slice_axis": 2,  # Slice axis is last
        }

        return self._reorient_array(array.astype(np.float32)), metadata

    def _load_nifti_nibabel(self, path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load NIfTI using nibabel."""
        import nibabel as nib

        img = nib.load(str(path))
        array = img.get_fdata()

        # Keep as (H, W, Z) - the processor expects slice dim last
        # NIfTI native format is (X, Y, Z) which we treat as (H, W, Z)
        # Do NOT transpose here - processor iterates img[:, :, i]

        metadata = {
            "affine": img.affine,
            "header": dict(img.header),
            "shape": img.shape,
            "slice_axis": 2,  # Mark which axis is slices
        }

        return self._reorient_array(array.astype(np.float32)), metadata

    def _reorient_array(self, array: np.ndarray) -> np.ndarray:
        """Reorient array based on target orientation."""
        # For now, assume input is in standard orientation
        # In production, would use orientation info from metadata
        return array

    def load(self, path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Auto-detect and load medical image.

        Args:
            path: Path to medical image file or directory

        Returns:
            Tuple of (image array, metadata dict)
        """
        path = Path(path)

        if path.is_dir():
            return self.load_dicom(path)

        suffix = path.suffix.lower()
        if suffix in ['.dcm', '.dicom'] or not suffix:
            return self.load_dicom(path)
        elif suffix in ['.nii', '.gz']:
            return self.load_nifti(path)
        elif suffix in ['.npy', '.npz']:
            array = np.load(str(path))
            if isinstance(array, np.lib.npyio.NpzFile):
                array = array[list(array.keys())[0]]
            return array.astype(np.float32), {}
        else:
            # Try loading as image
            img = Image.open(path).convert('L')
            return np.array(img, dtype=np.float32), {}


class MaskPrompt:
    """Handle mask-based prompting for CURIA."""

    def __init__(self, crop_size: int = 512):
        self.crop_size = crop_size

    def create_point_mask(
        self,
        image_shape: Tuple[int, int],
        points: List[Tuple[int, int]],
        radius: int = 10
    ) -> np.ndarray:
        """
        Create a mask from point prompts.

        Args:
            image_shape: (H, W) shape of the image
            points: List of (x, y) coordinates
            radius: Radius around each point

        Returns:
            Binary mask array
        """
        mask = np.zeros(image_shape, dtype=np.float32)

        for x, y in points:
            yy, xx = np.ogrid[:image_shape[0], :image_shape[1]]
            circle = (xx - x)**2 + (yy - y)**2 <= radius**2
            mask[circle] = 1.0

        return mask

    def create_box_mask(
        self,
        image_shape: Tuple[int, int],
        boxes: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """
        Create a mask from bounding box prompts.

        Args:
            image_shape: (H, W) shape of the image
            boxes: List of (x1, y1, x2, y2) bounding boxes

        Returns:
            Binary mask array
        """
        mask = np.zeros(image_shape, dtype=np.float32)

        for x1, y1, x2, y2 in boxes:
            mask[y1:y2, x1:x2] = 1.0

        return mask

    def load_mask(self, path: Union[str, Path]) -> np.ndarray:
        """Load mask from file."""
        path = Path(path)

        if path.suffix in ['.npy', '.npz']:
            mask = np.load(str(path))
            if isinstance(mask, np.lib.npyio.NpzFile):
                mask = mask[list(mask.keys())[0]]
        elif path.suffix in ['.nii', '.gz']:
            loader = MedicalImageLoader()
            mask, _ = loader.load_nifti(path)
        else:
            # Load as image
            img = Image.open(path).convert('L')
            mask = np.array(img, dtype=np.float32) / 255.0

        return (mask > 0.5).astype(np.float32)

    def transform_mask(self, mask: np.ndarray) -> torch.Tensor:
        """Transform mask for CURIA input."""
        # Convert to tensor
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # Add channel dim

        # Resize to crop_size
        mask = F.interpolate(
            mask.unsqueeze(0).float(),
            size=(self.crop_size, self.crop_size),
            mode='bilinear',
            align_corners=False
        )

        # Threshold to binary
        mask = (mask > 0.5).float()

        return mask.squeeze(0)


class CuriaTester:
    """Main class for testing CURIA model."""

    def __init__(self, config: Optional[CuriaConfig] = None):
        self.config = config or CuriaConfig()
        self.device = self._get_device()
        self.loader = MedicalImageLoader()
        self.mask_prompt = MaskPrompt(self.config.crop_size)

        self.processor = None
        self.backbone = None
        self.classifier = None

    def _get_device(self) -> torch.device:
        """Determine the best available device."""
        if self.config.device != "auto":
            return torch.device(self.config.device)

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def load_model(self, load_classifier: bool = False):
        """Load CURIA model and processor."""
        from transformers import AutoModel, AutoImageProcessor, AutoModelForImageClassification
        from huggingface_hub import HfFolder

        # Try to get token from: 1) config, 2) environment, 3) stored HF credentials
        token = self.config.hf_token or os.environ.get("HF_TOKEN") or HfFolder.get_token()

        print(f"Loading CURIA processor from {self.config.model_name}...")
        self.processor = AutoImageProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            token=token,
        )

        print(f"Loading CURIA backbone to {self.device}...")
        self.backbone = AutoModel.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            token=token,
            attn_implementation="eager",  # Required for attention map output
        ).to(self.device)
        self.backbone.eval()

        if load_classifier and self.config.subfolder:
            print(f"Loading classifier head: {self.config.subfolder}...")
            self.classifier = AutoModelForImageClassification.from_pretrained(
                self.config.model_name,
                subfolder=self.config.subfolder,
                trust_remote_code=True,
                token=token,
            ).to(self.device)
            self.classifier.eval()

        print("Model loaded successfully!")

    def preprocess_image(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        """Preprocess image for CURIA."""
        if self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Handle 3D volumes
        if image.ndim == 3:
            # Process each slice
            processed = self.processor(image, return_tensors="pt")
        else:
            processed = self.processor(image, return_tensors="pt")

        # Move to device
        return {k: v.to(self.device) for k, v in processed.items()}

    def extract_features(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        output_hidden_states: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from an image.

        Args:
            image: Input image array (H, W) or (Z, H, W)
            mask: Optional binary mask for region-specific features
            output_hidden_states: Whether to return all hidden states

        Returns:
            Dictionary containing features and attention maps
        """
        if self.backbone is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        processed = self.preprocess_image(image)

        with torch.no_grad():
            outputs = self.backbone(
                **processed,
                output_hidden_states=output_hidden_states,
                output_attentions=True
            )

        result = {
            "last_hidden_state": outputs.last_hidden_state,
            "pooler_output": outputs.pooler_output if hasattr(outputs, 'pooler_output') else None,
        }

        if output_hidden_states and hasattr(outputs, 'hidden_states'):
            result["hidden_states"] = outputs.hidden_states

        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            result["attentions"] = outputs.attentions

        # Extract mask-guided features if mask provided
        if mask is not None:
            result["mask_features"] = self._extract_mask_features(
                outputs.last_hidden_state,
                mask
            )

        return result

    def _extract_mask_features(
        self,
        patch_tokens: torch.Tensor,
        mask: np.ndarray
    ) -> torch.Tensor:
        """Extract features from masked region."""
        # Transform mask
        mask_tensor = self.mask_prompt.transform_mask(mask)
        mask_tensor = mask_tensor.to(self.device)

        # Get spatial dimensions
        spatial_dim = int(np.sqrt(patch_tokens.shape[1] - 1))  # Exclude CLS token

        # Reshape patch tokens to spatial
        patch_tokens = patch_tokens[:, 1:, :]  # Remove CLS token
        patch_tokens = patch_tokens.view(
            patch_tokens.shape[0], spatial_dim, spatial_dim, -1
        )

        # Resize mask to match patch spatial dims
        mask_resized = F.interpolate(
            mask_tensor.unsqueeze(0),
            size=(spatial_dim, spatial_dim),
            mode='bilinear'
        )
        mask_resized = (mask_resized > 0.5).float()
        mask_resized = mask_resized.permute(0, 2, 3, 1)

        # Extract masked features
        mask_features = (patch_tokens * mask_resized).sum(dim=(1, 2)) / (mask_resized.sum() + 1e-6)

        return mask_features

    def classify(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Classify an image using a pretrained head.

        Args:
            image: Input image array
            mask: Optional mask for region-specific classification

        Returns:
            Dictionary with logits, predictions, and probabilities
        """
        if self.classifier is None:
            raise RuntimeError(
                "Classifier not loaded. Call load_model(load_classifier=True) "
                "with a valid subfolder specified in config."
            )

        processed = self.preprocess_image(image)

        # Add mask if provided
        if mask is not None:
            if mask.ndim == 3:
                # For 3D volumes, use middle slice mask or aggregate
                # The classifier processes 2D images, so we need a 2D mask
                mid_idx = mask.shape[-1] // 2  # Assuming (H, W, Z) format
                mask_2d = mask[:, :, mid_idx]
                mask_tensor = self.mask_prompt.transform_mask(mask_2d)
            else:
                mask_tensor = self.mask_prompt.transform_mask(mask)
            processed["mask"] = mask_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.classifier(**processed)

        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax(-1)

        return {
            "logits": logits.cpu().numpy(),
            "probabilities": probs.cpu().numpy(),
            "prediction": pred.cpu().item(),
        }

    def get_attention_map(
        self,
        image: np.ndarray,
        layer_idx: int = -1
    ) -> np.ndarray:
        """
        Get attention map for visualization.

        This can be used for pseudo-segmentation by visualizing
        where the model attends to.

        Args:
            image: Input image array
            layer_idx: Which transformer layer to use (-1 for last)

        Returns:
            Attention map as numpy array
        """
        features = self.extract_features(image, output_hidden_states=True)

        if "attentions" not in features or features["attentions"] is None:
            raise RuntimeError("Model did not return attention maps")

        # Get attention from specified layer
        attn = features["attentions"][layer_idx]  # (batch, heads, seq, seq)

        # Average over heads
        attn = attn.mean(dim=1)  # (batch, seq, seq)

        # Get attention from CLS token to patches
        cls_attn = attn[0, 0, 1:]  # Exclude CLS-to-CLS

        # Reshape to spatial
        spatial_dim = int(np.sqrt(len(cls_attn)))
        attn_map = cls_attn.view(spatial_dim, spatial_dim)

        # Upsample to original size
        attn_map = F.interpolate(
            attn_map.unsqueeze(0).unsqueeze(0),
            size=(self.config.crop_size, self.config.crop_size),
            mode='bilinear'
        )

        return attn_map.squeeze().cpu().numpy()

    def process_volume(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None,
        slice_indices: Optional[List[int]] = None,
        slice_axis: int = -1
    ) -> Dict[str, Any]:
        """
        Process a 3D volume.

        Args:
            volume: 3D image array (H, W, Z) - slice axis last by default
            mask: Optional mask - can be 3D (H, W, Z) or 2D (H, W) broadcast to all
            slice_indices: Optional list of slice indices to process
            slice_axis: Axis along which to slice (default -1, i.e. last)

        Returns:
            Dictionary with per-slice and aggregated results
        """
        # Determine number of slices based on axis
        num_slices = volume.shape[slice_axis]

        if slice_indices is None:
            slice_indices = list(range(num_slices))

        results = {
            "per_slice_features": [],
            "per_slice_predictions": [],
            "slice_indices": slice_indices,
        }

        # Handle mask dimensions
        mask_is_2d = mask is not None and mask.ndim == 2

        for idx in slice_indices:
            # Extract slice along the correct axis
            if slice_axis == -1 or slice_axis == 2:
                slice_img = volume[:, :, idx]
                slice_mask = mask if mask_is_2d else (mask[:, :, idx] if mask is not None else None)
            elif slice_axis == 0:
                slice_img = volume[idx, :, :]
                slice_mask = mask if mask_is_2d else (mask[idx, :, :] if mask is not None else None)
            else:
                slice_img = volume[:, idx, :]
                slice_mask = mask if mask_is_2d else (mask[:, idx, :] if mask is not None else None)

            # Extract features
            features = self.extract_features(slice_img, mask=slice_mask)
            results["per_slice_features"].append(
                features["last_hidden_state"][:, 0].cpu()  # CLS token
            )

            # Classify if classifier available
            if self.classifier is not None:
                pred = self.classify(slice_img, mask=slice_mask)
                results["per_slice_predictions"].append(pred)

        # Stack and store all slice features
        if results["per_slice_features"]:
            all_features = torch.stack(results["per_slice_features"])  # [num_slices, 1, 768]
            all_features = all_features.squeeze(1)  # [num_slices, 768]

            # Store as numpy for easy export
            results["slice_features"] = all_features.numpy()  # [num_slices, 768]

            # Aggregation methods
            results["aggregated_features"] = all_features.mean(dim=0)  # [768] - mean
            results["aggregated_features_max"] = all_features.max(dim=0).values  # [768] - max pooling
            results["aggregated_features_std"] = all_features.std(dim=0)  # [768] - std deviation

            # Summary stats
            results["num_slices"] = len(slice_indices)
            results["feature_dim"] = all_features.shape[1]

        return results

    def export_features_to_csv(
        self,
        results: Dict[str, Any],
        output_path: str,
        include_header: bool = True
    ) -> str:
        """
        Export slice-wise features to CSV.

        Args:
            results: Output from process_volume()
            output_path: Path to save CSV
            include_header: Whether to include column headers

        Returns:
            Path to saved file
        """
        import csv

        if "slice_features" not in results:
            raise ValueError("No slice features found. Run process_volume() first.")

        slice_features = results["slice_features"]
        slice_indices = results["slice_indices"]
        feature_dim = slice_features.shape[1]

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            if include_header:
                header = ["slice_idx"] + [f"feat_{i}" for i in range(feature_dim)]
                writer.writerow(header)

            for i, idx in enumerate(slice_indices):
                row = [idx] + slice_features[i].tolist()
                writer.writerow(row)

        return output_path

    def export_features_to_json(
        self,
        results: Dict[str, Any],
        output_path: str,
        include_predictions: bool = True
    ) -> str:
        """
        Export features and predictions to JSON.

        Args:
            results: Output from process_volume()
            output_path: Path to save JSON
            include_predictions: Whether to include per-slice predictions

        Returns:
            Path to saved file
        """
        import json

        export_data = {
            "num_slices": results.get("num_slices", 0),
            "feature_dim": results.get("feature_dim", 768),
            "slice_indices": results.get("slice_indices", []),
            "aggregated_features": {
                "mean": results.get("aggregated_features", torch.zeros(768)).tolist()
                    if hasattr(results.get("aggregated_features", None), "tolist")
                    else results.get("aggregated_features", []),
                "max": results.get("aggregated_features_max", torch.zeros(768)).tolist()
                    if hasattr(results.get("aggregated_features_max", None), "tolist")
                    else results.get("aggregated_features_max", []),
                "std": results.get("aggregated_features_std", torch.zeros(768)).tolist()
                    if hasattr(results.get("aggregated_features_std", None), "tolist")
                    else results.get("aggregated_features_std", []),
            },
            "slice_features": results.get("slice_features", np.array([])).tolist(),
        }

        if include_predictions and results.get("per_slice_predictions"):
            export_data["per_slice_predictions"] = []
            for pred in results["per_slice_predictions"]:
                export_data["per_slice_predictions"].append({
                    "prediction": int(pred["prediction"]),
                    "probabilities": pred["probabilities"].tolist() if hasattr(pred["probabilities"], "tolist") else pred["probabilities"],
                })

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        return output_path


def create_interactive_session(tester: CuriaTester):
    """Create an interactive testing session."""
    print("\n" + "="*60)
    print("CURIA Interactive Testing Session")
    print("="*60)
    print("\nAvailable commands:")
    print("  load <path>        - Load a medical image (DICOM/NIfTI)")
    print("  mask <path>        - Load a mask file")
    print("  point <x> <y>      - Add a point prompt")
    print("  box <x1> <y1> <x2> <y2> - Add a box prompt")
    print("  clear_prompts      - Clear all prompts")
    print("  extract            - Extract features from current image")
    print("  classify           - Classify current image")
    print("  attention          - Get attention map")
    print("  info               - Show current image info")
    print("  heads              - List available classification heads")
    print("  set_head <name>    - Set classification head")
    print("  help               - Show this help message")
    print("  quit               - Exit session")
    print("="*60 + "\n")

    current_image = None
    current_mask = None
    current_metadata = None
    point_prompts = []
    box_prompts = []

    while True:
        try:
            cmd = input("curia> ").strip()
            if not cmd:
                continue

            parts = cmd.split()
            action = parts[0].lower()

            if action == "quit" or action == "exit":
                print("Goodbye!")
                break

            elif action == "help":
                # Print help without recursion
                print("\nAvailable commands:")
                print("  load <path>        - Load a medical image (DICOM/NIfTI)")
                print("  mask <path>        - Load a mask file")
                print("  point <x> <y>      - Add a point prompt")
                print("  box <x1> <y1> <x2> <y2> - Add a box prompt")
                print("  clear_prompts      - Clear all prompts")
                print("  extract            - Extract features from current image")
                print("  classify           - Classify current image")
                print("  attention          - Get attention map")
                print("  info               - Show current image info")
                print("  heads              - List available classification heads")
                print("  set_head <name>    - Set classification head")
                print("  help               - Show this help message")
                print("  quit               - Exit session")

            elif action == "heads":
                print("\nAvailable classification heads:")
                for head in tester.config.AVAILABLE_HEADS:
                    marker = " *" if head == tester.config.subfolder else ""
                    print(f"  - {head}{marker}")
                print()

            elif action == "set_head":
                if len(parts) < 2:
                    print("Usage: set_head <head_name>")
                    continue
                head_name = parts[1]
                if head_name in tester.config.AVAILABLE_HEADS:
                    tester.config.subfolder = head_name
                    tester.load_model(load_classifier=True)
                else:
                    print(f"Unknown head: {head_name}")

            elif action == "load":
                if len(parts) < 2:
                    print("Usage: load <path>")
                    continue
                path = " ".join(parts[1:])
                try:
                    current_image, current_metadata = tester.loader.load(path)
                    point_prompts = []
                    box_prompts = []
                    current_mask = None
                    print(f"Loaded image with shape: {current_image.shape}")
                    print(f"Value range: [{current_image.min():.2f}, {current_image.max():.2f}]")
                except Exception as e:
                    print(f"Error loading image: {e}")

            elif action == "mask":
                if len(parts) < 2:
                    print("Usage: mask <path>")
                    continue
                path = " ".join(parts[1:])
                try:
                    current_mask = tester.mask_prompt.load_mask(path)
                    print(f"Loaded mask with shape: {current_mask.shape}")
                except Exception as e:
                    print(f"Error loading mask: {e}")

            elif action == "point":
                if len(parts) < 3:
                    print("Usage: point <x> <y>")
                    continue
                x, y = int(parts[1]), int(parts[2])
                point_prompts.append((x, y))
                print(f"Added point prompt at ({x}, {y})")
                print(f"Total point prompts: {len(point_prompts)}")

            elif action == "box":
                if len(parts) < 5:
                    print("Usage: box <x1> <y1> <x2> <y2>")
                    continue
                box = tuple(int(p) for p in parts[1:5])
                box_prompts.append(box)
                print(f"Added box prompt: {box}")
                print(f"Total box prompts: {len(box_prompts)}")

            elif action == "clear_prompts":
                point_prompts = []
                box_prompts = []
                current_mask = None
                print("Cleared all prompts")

            elif action == "info":
                if current_image is None:
                    print("No image loaded")
                else:
                    print(f"\nCurrent image:")
                    print(f"  Shape: {current_image.shape}")
                    print(f"  Dtype: {current_image.dtype}")
                    print(f"  Range: [{current_image.min():.2f}, {current_image.max():.2f}]")
                    if current_metadata:
                        print(f"  Metadata: {current_metadata}")
                    print(f"  Point prompts: {len(point_prompts)}")
                    print(f"  Box prompts: {len(box_prompts)}")
                    print(f"  Mask loaded: {current_mask is not None}")
                print()

            elif action == "extract":
                if current_image is None:
                    print("No image loaded. Use 'load <path>' first.")
                    continue

                # Create mask from prompts if no explicit mask
                # For 3D (H, W, Z), use shape[:2] to get (H, W)
                mask_to_use = current_mask
                if mask_to_use is None and (point_prompts or box_prompts):
                    img_shape = current_image.shape[:2]  # (H, W)
                    if point_prompts:
                        mask_to_use = tester.mask_prompt.create_point_mask(img_shape, point_prompts)
                    if box_prompts:
                        box_mask = tester.mask_prompt.create_box_mask(img_shape, box_prompts)
                        if mask_to_use is None:
                            mask_to_use = box_mask
                        else:
                            mask_to_use = np.maximum(mask_to_use, box_mask)

                try:
                    if current_image.ndim == 3:
                        print("Processing 3D volume...")
                        results = tester.process_volume(current_image, mask=mask_to_use)
                        print(f"Processed {len(results['slice_indices'])} slices")
                        print(f"Aggregated feature shape: {results['aggregated_features'].shape}")
                    else:
                        print("Extracting features...")
                        features = tester.extract_features(current_image, mask=mask_to_use)
                        print(f"Feature shape: {features['last_hidden_state'].shape}")
                        if "mask_features" in features:
                            print(f"Mask feature shape: {features['mask_features'].shape}")
                except Exception as e:
                    print(f"Error extracting features: {e}")

            elif action == "classify":
                if current_image is None:
                    print("No image loaded. Use 'load <path>' first.")
                    continue
                if tester.classifier is None:
                    print("No classifier loaded. Use 'set_head <name>' first.")
                    continue

                # For 3D (H, W, Z), use shape[:2] to get (H, W)
                mask_to_use = current_mask
                if mask_to_use is None and (point_prompts or box_prompts):
                    img_shape = current_image.shape[:2]  # (H, W)
                    if point_prompts:
                        mask_to_use = tester.mask_prompt.create_point_mask(img_shape, point_prompts)
                    if box_prompts:
                        box_mask = tester.mask_prompt.create_box_mask(img_shape, box_prompts)
                        if mask_to_use is None:
                            mask_to_use = box_mask
                        else:
                            mask_to_use = np.maximum(mask_to_use, box_mask)

                try:
                    # For 3D, use middle slice (format is H, W, Z)
                    if current_image.ndim == 3:
                        slice_idx = current_image.shape[-1] // 2
                        img = current_image[:, :, slice_idx]
                        mask = mask_to_use[:, :, slice_idx] if mask_to_use is not None and mask_to_use.ndim == 3 else mask_to_use
                    else:
                        img = current_image
                        mask = mask_to_use

                    result = tester.classify(img, mask=mask)
                    print(f"\nClassification result:")
                    print(f"  Prediction: {result['prediction']}")
                    print(f"  Probabilities: {result['probabilities']}")
                    print()
                except Exception as e:
                    print(f"Error classifying: {e}")

            elif action == "attention":
                if current_image is None:
                    print("No image loaded. Use 'load <path>' first.")
                    continue

                try:
                    # For 3D, use middle slice (format is H, W, Z)
                    if current_image.ndim == 3:
                        img = current_image[:, :, current_image.shape[-1] // 2]
                    else:
                        img = current_image

                    attn_map = tester.get_attention_map(img)
                    print(f"Attention map shape: {attn_map.shape}")
                    print(f"Attention range: [{attn_map.min():.4f}, {attn_map.max():.4f}]")

                    # Save attention map
                    save_path = Path("attention_map.npy")
                    np.save(save_path, attn_map)
                    print(f"Saved attention map to {save_path}")
                except Exception as e:
                    print(f"Error getting attention: {e}")

            else:
                print(f"Unknown command: {action}. Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="CURIA Medical Image Testing Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python curia_tester.py --interactive

  # Extract features from a DICOM file
  python curia_tester.py --input scan.dcm --mode features

  # Classify with a specific head
  python curia_tester.py --input scan.nii.gz --mode classify --head luna16-3D

  # Use mask prompting
  python curia_tester.py --input scan.dcm --mask lesion_mask.nii.gz --mode features

  # Point prompting
  python curia_tester.py --input scan.dcm --points "100,100;200,150" --mode features
        """
    )

    parser.add_argument("--input", "-i", type=str, help="Input image path (DICOM, NIfTI, or numpy)")
    parser.add_argument("--mask", "-m", type=str, help="Mask file path for prompting")
    parser.add_argument("--points", type=str, help="Point prompts as 'x1,y1;x2,y2;...'")
    parser.add_argument("--boxes", type=str, help="Box prompts as 'x1,y1,x2,y2;...'")
    parser.add_argument("--mode", choices=["features", "classify", "attention"], default="features",
                        help="Processing mode")
    parser.add_argument("--head", type=str, help="Classification head to use")
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    parser.add_argument("--interactive", action="store_true", help="Start interactive session")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                        help="Device to use")
    parser.add_argument("--hf-token", type=str, help="HuggingFace token")

    args = parser.parse_args()

    # Create config
    config = CuriaConfig(
        device=args.device,
        hf_token=args.hf_token,
        subfolder=args.head,
    )

    # Create tester
    tester = CuriaTester(config)

    # Load model
    print(f"Using device: {tester.device}")
    tester.load_model(load_classifier=(args.head is not None))

    if args.interactive:
        create_interactive_session(tester)
        return

    if not args.input:
        parser.print_help()
        print("\nError: --input is required unless --interactive is specified")
        return

    # Load image
    print(f"Loading image from {args.input}...")
    image, metadata = tester.loader.load(args.input)
    print(f"Image shape: {image.shape}")

    # Load or create mask
    mask = None
    if args.mask:
        mask = tester.mask_prompt.load_mask(args.mask)
        print(f"Loaded mask with shape: {mask.shape}")

    # For 3D (H, W, Z), use shape[:2] to get (H, W)
    if args.points:
        points = [tuple(map(int, p.split(','))) for p in args.points.split(';')]
        img_shape = image.shape[:2]  # (H, W)
        point_mask = tester.mask_prompt.create_point_mask(img_shape, points)
        mask = point_mask if mask is None else np.maximum(mask, point_mask)
        print(f"Created point mask from {len(points)} points")

    if args.boxes:
        boxes = [tuple(map(int, b.split(','))) for b in args.boxes.split(';')]
        img_shape = image.shape[:2]  # (H, W)
        box_mask = tester.mask_prompt.create_box_mask(img_shape, boxes)
        mask = box_mask if mask is None else np.maximum(mask, box_mask)
        print(f"Created box mask from {len(boxes)} boxes")

    # Process based on mode
    if args.mode == "features":
        print("Extracting features...")
        if image.ndim == 3:
            results = tester.process_volume(image, mask=mask)
            output_data = {
                "aggregated_features": results["aggregated_features"].numpy(),
                "slice_indices": results["slice_indices"],
            }
        else:
            features = tester.extract_features(image, mask=mask)
            output_data = {
                "features": features["last_hidden_state"].cpu().numpy(),
            }
            if "mask_features" in features:
                output_data["mask_features"] = features["mask_features"].cpu().numpy()

        output_path = args.output or "features.npz"
        np.savez(output_path, **output_data)
        print(f"Saved features to {output_path}")

    elif args.mode == "classify":
        if tester.classifier is None:
            print("Error: No classification head specified. Use --head option.")
            return

        print("Classifying...")
        # For 3D (H, W, Z), use middle slice on last axis
        if image.ndim == 3:
            slice_idx = image.shape[-1] // 2
            image = image[:, :, slice_idx]
            if mask is not None and mask.ndim == 3:
                mask = mask[:, :, slice_idx]

        result = tester.classify(image, mask=mask)
        print(f"\nClassification result:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Probabilities: {result['probabilities']}")

        if args.output:
            np.savez(args.output, **result)
            print(f"Saved result to {args.output}")

    elif args.mode == "attention":
        print("Generating attention map...")
        # For 3D (H, W, Z), use middle slice on last axis
        if image.ndim == 3:
            image = image[:, :, image.shape[-1] // 2]

        attn_map = tester.get_attention_map(image)

        output_path = args.output or "attention_map.npy"
        np.save(output_path, attn_map)
        print(f"Saved attention map to {output_path}")
        print(f"Attention range: [{attn_map.min():.4f}, {attn_map.max():.4f}]")


if __name__ == "__main__":
    main()
