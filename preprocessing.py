"""
CURIA Preprocessing Utilities
=============================

Preprocessing functions for medical images before CURIA inference:
- Normalization (CT windowing, MRI intensity normalization)
- Slice selection (automatic and manual)
- Image orientation handling
- Resampling and resizing

Usage:
    from preprocessing import Preprocessor, SliceSelector, Normalizer

    # Create preprocessor
    preprocessor = Preprocessor(modality='ct')

    # Process volume
    processed = preprocessor.process(volume)

    # Or use individual components
    normalizer = Normalizer(modality='ct')
    selector = SliceSelector()
"""

import numpy as np
from typing import Optional, Tuple, List, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum


class Modality(Enum):
    """Supported imaging modalities."""
    CT = "ct"
    MRI = "mri"
    AUTO = "auto"


@dataclass
class WindowSettings:
    """CT window settings for different anatomical regions."""
    center: float  # Window center (level)
    width: float   # Window width
    name: str = ""

    def __post_init__(self):
        self.min_hu = self.center - self.width / 2
        self.max_hu = self.center + self.width / 2


# Predefined CT windows for common anatomical regions
CT_WINDOWS = {
    "brain": WindowSettings(center=40, width=80, name="Brain"),
    "brain_stroke": WindowSettings(center=40, width=40, name="Brain Stroke"),
    "subdural": WindowSettings(center=75, width=215, name="Subdural"),
    "bone": WindowSettings(center=400, width=2000, name="Bone"),
    "soft_tissue": WindowSettings(center=40, width=400, name="Soft Tissue"),
    "lung": WindowSettings(center=-600, width=1500, name="Lung"),
    "lung_nodule": WindowSettings(center=-400, width=1000, name="Lung Nodule"),
    "mediastinum": WindowSettings(center=40, width=400, name="Mediastinum"),
    "abdomen": WindowSettings(center=40, width=350, name="Abdomen"),
    "liver": WindowSettings(center=60, width=150, name="Liver"),
    "spine": WindowSettings(center=50, width=250, name="Spine"),
    "angio": WindowSettings(center=300, width=600, name="CT Angiography"),
}


class Normalizer:
    """
    Normalize medical images for CURIA inference.

    For CT: Apply Hounsfield Unit windowing, then normalize to [0, 1]
    For MRI: Apply percentile-based intensity normalization
    """

    def __init__(
        self,
        modality: Union[str, Modality] = "auto",
        window: Optional[Union[str, WindowSettings]] = None,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
    ):
        """
        Initialize normalizer.

        Args:
            modality: 'ct', 'mri', or 'auto' (detect from intensity range)
            window: CT window preset name or WindowSettings object
            percentile_low: Lower percentile for MRI normalization
            percentile_high: Upper percentile for MRI normalization
        """
        if isinstance(modality, str):
            modality = Modality(modality.lower())
        self.modality = modality

        # Set CT window
        if isinstance(window, str):
            if window in CT_WINDOWS:
                self.window = CT_WINDOWS[window]
            else:
                raise ValueError(f"Unknown CT window: {window}. Available: {list(CT_WINDOWS.keys())}")
        elif isinstance(window, WindowSettings):
            self.window = window
        else:
            self.window = CT_WINDOWS["soft_tissue"]  # Default

        self.percentile_low = percentile_low
        self.percentile_high = percentile_high

    def detect_modality(self, image: np.ndarray) -> Modality:
        """Auto-detect modality from intensity range."""
        min_val, max_val = image.min(), image.max()

        # CT typically has negative values (air = -1000 HU)
        if min_val < -100:
            return Modality.CT
        # CT range is typically -1024 to 3000+
        elif max_val > 2000:
            return Modality.CT
        else:
            return Modality.MRI

    def normalize_ct(
        self,
        image: np.ndarray,
        window: Optional[WindowSettings] = None,
        output_range: Tuple[float, float] = (0, 1)
    ) -> np.ndarray:
        """
        Apply CT windowing and normalize to output range.

        Args:
            image: Input CT image in Hounsfield Units
            window: Window settings (uses default if None)
            output_range: Output intensity range

        Returns:
            Normalized image in output_range
        """
        window = window or self.window

        # Clip to window range
        clipped = np.clip(image, window.min_hu, window.max_hu)

        # Normalize to output range
        normalized = (clipped - window.min_hu) / window.width
        normalized = normalized * (output_range[1] - output_range[0]) + output_range[0]

        return normalized.astype(np.float32)

    def normalize_mri(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        output_range: Tuple[float, float] = (0, 1)
    ) -> np.ndarray:
        """
        Normalize MRI using percentile-based intensity scaling.

        Args:
            image: Input MRI image
            mask: Optional mask for calculating percentiles (brain mask, etc.)
            output_range: Output intensity range

        Returns:
            Normalized image in output_range
        """
        if mask is not None:
            values = image[mask > 0]
        else:
            values = image.flatten()

        # Calculate percentiles
        p_low = np.percentile(values, self.percentile_low)
        p_high = np.percentile(values, self.percentile_high)

        # Clip and normalize
        clipped = np.clip(image, p_low, p_high)
        if p_high > p_low:
            normalized = (clipped - p_low) / (p_high - p_low)
        else:
            normalized = np.zeros_like(clipped)

        normalized = normalized * (output_range[1] - output_range[0]) + output_range[0]

        return normalized.astype(np.float32)

    def zscore_normalize(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Z-score normalization (mean=0, std=1).

        This is what CURIA's processor does internally.

        Args:
            image: Input image
            mask: Optional mask for calculating statistics

        Returns:
            Z-score normalized image
        """
        if mask is not None:
            values = image[mask > 0]
        else:
            values = image.flatten()

        mean = np.mean(values)
        std = np.std(values)

        if std > 0:
            normalized = (image - mean) / std
        else:
            normalized = image - mean

        return normalized.astype(np.float32)

    def normalize(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        output_range: Tuple[float, float] = (0, 1)
    ) -> np.ndarray:
        """
        Normalize image based on modality.

        Args:
            image: Input image
            mask: Optional mask for MRI normalization
            output_range: Output intensity range

        Returns:
            Normalized image
        """
        modality = self.modality
        if modality == Modality.AUTO:
            modality = self.detect_modality(image)

        if modality == Modality.CT:
            return self.normalize_ct(image, output_range=output_range)
        else:
            return self.normalize_mri(image, mask=mask, output_range=output_range)

    @staticmethod
    def list_ct_windows() -> Dict[str, WindowSettings]:
        """Get available CT window presets."""
        return CT_WINDOWS.copy()


class SliceSelector:
    """
    Select informative slices from 3D volumes.

    Methods:
    - Manual: Specify exact slice indices
    - Uniform: Select evenly spaced slices
    - Content-based: Select slices with most content
    - Center: Select slices around the center
    """

    def __init__(
        self,
        method: str = "uniform",
        num_slices: int = 10,
        content_threshold: float = 0.1,
    ):
        """
        Initialize slice selector.

        Args:
            method: Selection method ('uniform', 'content', 'center', 'all')
            num_slices: Number of slices to select
            content_threshold: Threshold for content-based selection
        """
        self.method = method
        self.num_slices = num_slices
        self.content_threshold = content_threshold

    def select_uniform(
        self,
        volume: np.ndarray,
        num_slices: Optional[int] = None,
        axis: int = -1
    ) -> List[int]:
        """
        Select uniformly spaced slices.

        Args:
            volume: 3D volume
            num_slices: Number of slices to select
            axis: Axis along which to select slices

        Returns:
            List of slice indices
        """
        num_slices = num_slices or self.num_slices
        n_total = volume.shape[axis]

        if num_slices >= n_total:
            return list(range(n_total))

        # Evenly space slices, avoiding edges
        margin = n_total // (2 * num_slices)
        indices = np.linspace(margin, n_total - margin - 1, num_slices)
        return [int(i) for i in indices]

    def select_content_based(
        self,
        volume: np.ndarray,
        num_slices: Optional[int] = None,
        axis: int = -1,
        threshold: Optional[float] = None
    ) -> List[int]:
        """
        Select slices with most content (non-background pixels).

        Args:
            volume: 3D volume
            num_slices: Number of slices to select
            axis: Axis along which to select slices
            threshold: Content threshold (fraction of max)

        Returns:
            List of slice indices sorted by content amount
        """
        num_slices = num_slices or self.num_slices
        threshold = threshold or self.content_threshold
        n_total = volume.shape[axis]

        # Calculate content score for each slice
        scores = []
        for i in range(n_total):
            if axis == -1 or axis == 2:
                slice_data = volume[:, :, i]
            elif axis == 0:
                slice_data = volume[i, :, :]
            else:
                slice_data = volume[:, i, :]

            # Content score: variance + non-zero fraction
            variance = np.var(slice_data)

            # For CT, consider non-air voxels; for MRI, non-zero
            if slice_data.min() < -500:  # Likely CT
                non_bg = np.mean(slice_data > -500)
            else:
                non_bg = np.mean(slice_data > threshold * slice_data.max())

            score = variance * non_bg
            scores.append((i, score))

        # Sort by score and take top slices
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [idx for idx, _ in scores[:num_slices]]

        # Return in sorted order (from inferior to superior)
        return sorted(selected)

    def select_center(
        self,
        volume: np.ndarray,
        num_slices: Optional[int] = None,
        axis: int = -1
    ) -> List[int]:
        """
        Select slices around the center of the volume.

        Args:
            volume: 3D volume
            num_slices: Number of slices to select
            axis: Axis along which to select slices

        Returns:
            List of slice indices around center
        """
        num_slices = num_slices or self.num_slices
        n_total = volume.shape[axis]

        if num_slices >= n_total:
            return list(range(n_total))

        center = n_total // 2
        half = num_slices // 2

        start = max(0, center - half)
        end = min(n_total, start + num_slices)

        return list(range(start, end))

    def select(
        self,
        volume: np.ndarray,
        method: Optional[str] = None,
        num_slices: Optional[int] = None,
        axis: int = -1,
        indices: Optional[List[int]] = None
    ) -> List[int]:
        """
        Select slices using specified method.

        Args:
            volume: 3D volume
            method: Selection method (overrides default)
            num_slices: Number of slices (overrides default)
            axis: Axis along which to select
            indices: Manual indices (overrides method)

        Returns:
            List of slice indices
        """
        if indices is not None:
            # Manual selection
            return [i for i in indices if 0 <= i < volume.shape[axis]]

        method = method or self.method
        num_slices = num_slices or self.num_slices

        if method == "uniform":
            return self.select_uniform(volume, num_slices, axis)
        elif method == "content":
            return self.select_content_based(volume, num_slices, axis)
        elif method == "center":
            return self.select_center(volume, num_slices, axis)
        elif method == "all":
            return list(range(volume.shape[axis]))
        else:
            raise ValueError(f"Unknown selection method: {method}")

    def get_slices(
        self,
        volume: np.ndarray,
        indices: List[int],
        axis: int = -1
    ) -> List[np.ndarray]:
        """
        Extract slices from volume.

        Args:
            volume: 3D volume
            indices: Slice indices
            axis: Axis along which to extract

        Returns:
            List of 2D slice arrays
        """
        slices = []
        for i in indices:
            if axis == -1 or axis == 2:
                slices.append(volume[:, :, i])
            elif axis == 0:
                slices.append(volume[i, :, :])
            else:
                slices.append(volume[:, i, :])
        return slices


class Resampler:
    """
    Resample and resize medical images.
    """

    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        """
        Initialize resampler.

        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size

    def resize(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        method: str = "bilinear"
    ) -> np.ndarray:
        """
        Resize 2D image.

        Args:
            image: Input 2D image
            target_size: Target size (height, width)
            method: Interpolation method ('nearest', 'bilinear', 'bicubic')

        Returns:
            Resized image
        """
        target_size = target_size or self.target_size

        if image.shape[:2] == target_size:
            return image

        try:
            from scipy import ndimage

            # Calculate zoom factors
            zoom_factors = [target_size[0] / image.shape[0],
                           target_size[1] / image.shape[1]]

            # Map method names to scipy order
            order_map = {"nearest": 0, "bilinear": 1, "bicubic": 3}
            order = order_map.get(method, 1)

            resized = ndimage.zoom(image, zoom_factors, order=order)
            return resized.astype(image.dtype)

        except ImportError:
            # Fallback using numpy (nearest neighbor only)
            import warnings
            warnings.warn("scipy not available, using simple resize")

            h, w = target_size
            h_old, w_old = image.shape[:2]

            # Create index arrays
            row_idx = (np.arange(h) * h_old / h).astype(int)
            col_idx = (np.arange(w) * w_old / w).astype(int)

            # Clip indices
            row_idx = np.clip(row_idx, 0, h_old - 1)
            col_idx = np.clip(col_idx, 0, w_old - 1)

            return image[row_idx][:, col_idx].astype(image.dtype)

    def resize_volume(
        self,
        volume: np.ndarray,
        target_spacing: Optional[Tuple[float, float, float]] = None,
        current_spacing: Optional[Tuple[float, float, float]] = None,
        method: str = "bilinear"
    ) -> np.ndarray:
        """
        Resample 3D volume to target spacing.

        Args:
            volume: Input 3D volume
            target_spacing: Target voxel spacing (x, y, z)
            current_spacing: Current voxel spacing (x, y, z)
            method: Interpolation method

        Returns:
            Resampled volume
        """
        from scipy import ndimage

        if target_spacing is None or current_spacing is None:
            return volume

        # Calculate zoom factors
        zoom_factors = [
            current_spacing[i] / target_spacing[i]
            for i in range(3)
        ]

        order_map = {"nearest": 0, "bilinear": 1, "bicubic": 3}
        order = order_map.get(method, 1)

        resampled = ndimage.zoom(volume, zoom_factors, order=order)
        return resampled.astype(volume.dtype)

    def center_crop(
        self,
        image: np.ndarray,
        crop_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Center crop image to specified size.

        Args:
            image: Input image
            crop_size: Target crop size (height, width)

        Returns:
            Cropped image
        """
        h, w = image.shape[:2]
        new_h, new_w = crop_size

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        return image[top:top+new_h, left:left+new_w]

    def pad_to_size(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        pad_value: float = 0
    ) -> np.ndarray:
        """
        Pad image to target size.

        Args:
            image: Input image
            target_size: Target size (height, width)
            pad_value: Value to use for padding

        Returns:
            Padded image
        """
        h, w = image.shape[:2]
        new_h, new_w = target_size

        if h >= new_h and w >= new_w:
            return image

        pad_h = max(0, new_h - h)
        pad_w = max(0, new_w - w)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padded = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=pad_value
        )

        return padded


class Preprocessor:
    """
    Complete preprocessing pipeline for CURIA.

    Combines normalization, slice selection, and resizing.
    """

    def __init__(
        self,
        modality: str = "auto",
        ct_window: str = "soft_tissue",
        target_size: Tuple[int, int] = (512, 512),
        slice_method: str = "uniform",
        num_slices: int = 10,
    ):
        """
        Initialize preprocessor.

        Args:
            modality: Image modality ('ct', 'mri', 'auto')
            ct_window: CT window preset name
            target_size: Target image size for CURIA
            slice_method: Slice selection method
            num_slices: Number of slices to select from volumes
        """
        self.normalizer = Normalizer(modality=modality, window=ct_window)
        self.selector = SliceSelector(method=slice_method, num_slices=num_slices)
        self.resampler = Resampler(target_size=target_size)
        self.target_size = target_size

    def process_slice(
        self,
        image: np.ndarray,
        normalize: bool = True,
        resize: bool = True,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Process a single 2D slice.

        Args:
            image: Input 2D image
            normalize: Whether to normalize
            resize: Whether to resize to target size
            mask: Optional mask for normalization

        Returns:
            Processed image
        """
        processed = image.astype(np.float32)

        if normalize:
            processed = self.normalizer.normalize(processed, mask=mask)

        if resize and processed.shape[:2] != self.target_size:
            processed = self.resampler.resize(processed, self.target_size)

        return processed

    def process_volume(
        self,
        volume: np.ndarray,
        slice_indices: Optional[List[int]] = None,
        normalize: bool = True,
        resize: bool = True,
        axis: int = -1
    ) -> Dict[str, Any]:
        """
        Process a 3D volume.

        Args:
            volume: Input 3D volume
            slice_indices: Manual slice indices (auto-select if None)
            normalize: Whether to normalize
            resize: Whether to resize to target size
            axis: Slice axis

        Returns:
            Dictionary with processed slices and metadata
        """
        # Select slices
        if slice_indices is None:
            slice_indices = self.selector.select(volume, axis=axis)

        # Extract and process slices
        raw_slices = self.selector.get_slices(volume, slice_indices, axis)
        processed_slices = []

        for slice_img in raw_slices:
            processed = self.process_slice(slice_img, normalize, resize)
            processed_slices.append(processed)

        return {
            "slices": processed_slices,
            "slice_indices": slice_indices,
            "volume_shape": volume.shape,
            "modality": self.normalizer.detect_modality(volume).value,
            "normalized": normalize,
            "resized": resize,
        }

    def batch_process(
        self,
        images: List[np.ndarray],
        normalize: bool = True,
        resize: bool = True
    ) -> np.ndarray:
        """
        Process batch of images.

        Args:
            images: List of 2D images
            normalize: Whether to normalize
            resize: Whether to resize

        Returns:
            Batch array of shape (N, H, W)
        """
        processed = []
        for img in images:
            proc_img = self.process_slice(img, normalize, resize)
            processed.append(proc_img)

        return np.stack(processed, axis=0)


def quick_preprocess(
    image: np.ndarray,
    modality: str = "auto",
    window: str = "soft_tissue"
) -> np.ndarray:
    """
    Quick preprocessing for a single image.

    Args:
        image: Input image
        modality: 'ct', 'mri', or 'auto'
        window: CT window preset (if CT)

    Returns:
        Preprocessed image
    """
    preprocessor = Preprocessor(modality=modality, ct_window=window)
    return preprocessor.process_slice(image)


# Example usage
if __name__ == "__main__":
    print("CURIA Preprocessing Utilities")
    print("=" * 50)

    # List available CT windows
    print("\nAvailable CT Windows:")
    for name, settings in CT_WINDOWS.items():
        print(f"  {name}: center={settings.center}, width={settings.width}")

    # Demo with synthetic data
    print("\n" + "=" * 50)
    print("Demo with synthetic data")
    print("=" * 50)

    # Create synthetic CT slice
    ct_slice = np.random.randn(256, 256) * 500 - 500  # CT-like range

    # Normalize with different windows
    normalizer = Normalizer(modality="ct")

    for window_name in ["brain", "lung", "soft_tissue"]:
        normalizer.window = CT_WINDOWS[window_name]
        normalized = normalizer.normalize_ct(ct_slice)
        print(f"\n{window_name} window: range [{normalized.min():.3f}, {normalized.max():.3f}]")

    # Create synthetic 3D volume
    print("\n" + "=" * 50)
    print("Slice Selection Demo")
    print("=" * 50)

    volume = np.random.randn(256, 256, 100)
    selector = SliceSelector()

    print(f"\nVolume shape: {volume.shape}")
    print(f"Uniform selection (10): {selector.select_uniform(volume)}")
    print(f"Center selection (10): {selector.select_center(volume)}")
    print(f"Content-based (10): {selector.select_content_based(volume)}")

    # Complete pipeline
    print("\n" + "=" * 50)
    print("Complete Preprocessing Pipeline")
    print("=" * 50)

    preprocessor = Preprocessor(
        modality="ct",
        ct_window="soft_tissue",
        target_size=(512, 512),
        slice_method="uniform",
        num_slices=5
    )

    result = preprocessor.process_volume(volume)
    print(f"\nProcessed {len(result['slices'])} slices")
    print(f"Slice indices: {result['slice_indices']}")
    print(f"Output shape: {result['slices'][0].shape}")
    print(f"Detected modality: {result['modality']}")
