"""
Create synthetic test data for CURIA testing.

This script generates:
1. Synthetic brain MRI NIfTI file
2. Synthetic CT NIfTI file
3. Sample lesion masks
4. Sample DICOM files (if pydicom available)

Usage:
    python create_test_data.py
"""

import os
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent / "test_data"


def create_synthetic_brain_mri(shape=(256, 256, 128), output_path=None):
    """
    Create a synthetic brain MRI volume.

    Simulates:
    - Skull (bright ring)
    - Gray matter (medium intensity)
    - White matter (higher intensity)
    - CSF/ventricles (low intensity)
    """
    print("Creating synthetic brain MRI...")

    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center = np.array(shape) // 2

    # Create base volume
    volume = np.zeros(shape, dtype=np.float32)

    # Distance from center
    dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)

    # Skull (outer bright ring)
    skull_outer = 100
    skull_inner = 90
    skull_mask = (dist < skull_outer) & (dist > skull_inner)
    volume[skull_mask] = 800 + np.random.randn(*volume[skull_mask].shape) * 50

    # Brain parenchyma (gray matter)
    brain_mask = dist < skull_inner
    volume[brain_mask] = 600 + np.random.randn(*volume[brain_mask].shape) * 80

    # White matter (central, higher intensity)
    wm_mask = dist < 60
    volume[wm_mask] = 750 + np.random.randn(*volume[wm_mask].shape) * 60

    # Ventricles (CSF, low intensity, elongated shape)
    vent_mask = (dist < 25) & (np.abs(y - center[1]) < 15)
    volume[vent_mask] = 200 + np.random.randn(*volume[vent_mask].shape) * 30

    # Add some texture/noise
    volume += np.random.randn(*shape) * 20

    # Clip to realistic MRI range
    volume = np.clip(volume, 0, 1000)

    if output_path:
        save_nifti(volume, output_path)
        print(f"  Saved: {output_path}")

    return volume


def create_synthetic_ct(shape=(512, 512, 100), output_path=None):
    """
    Create a synthetic CT volume (chest/abdomen).

    Simulates:
    - Air (outside body): -1000 HU
    - Lung: -500 to -900 HU
    - Soft tissue: 20-80 HU
    - Bone: 300-1000 HU
    """
    print("Creating synthetic CT...")

    # Shape is (H, W, D) for CT
    H, W, D = shape
    z, y, x = np.ogrid[:H, :W, :D]
    center_y, center_x = H // 2, W // 2

    # Start with air
    volume = np.full(shape, -1000, dtype=np.float32)

    # Body outline (elliptical) - broadcast across all slices
    body_mask = ((y - center_y)**2 / 150**2 + (x - center_x)**2 / 120**2) < 1
    body_mask = np.broadcast_to(body_mask, shape)
    volume[body_mask] = 40 + np.random.randn(body_mask.sum()) * 20  # Soft tissue

    # Lungs (two ellipses)
    left_lung = ((y - center_y - 50)**2 / 80**2 + (x - center_x)**2 / 50**2) < 1
    right_lung = ((y - center_y + 50)**2 / 80**2 + (x - center_x)**2 / 50**2) < 1
    lung_mask = np.broadcast_to((left_lung | right_lung), shape) & body_mask
    volume[lung_mask] = -700 + np.random.randn(lung_mask.sum()) * 100

    # Spine (central bone)
    spine_mask = ((y - center_y)**2 + (x - center_x + 100)**2) < 20**2
    spine_mask = np.broadcast_to(spine_mask, shape) & body_mask
    volume[spine_mask] = 500 + np.random.randn(spine_mask.sum()) * 100

    # Ribs (simplified as arcs)
    for rib_z in range(10, shape[2] - 10, 15):
        rib_mask_slice = np.zeros((shape[0], shape[1]), dtype=bool)
        for angle in np.linspace(-1.5, 1.5, 50):
            ry = int(center_y + 120 * np.sin(angle))
            rx = int(center_x + 100 * np.cos(angle))
            if 0 <= ry < shape[0] and 0 <= rx < shape[1]:
                rib_mask_slice[max(0, ry-3):min(shape[0], ry+3),
                              max(0, rx-3):min(shape[1], rx+3)] = True
        for dz in range(-2, 3):
            if 0 <= rib_z + dz < shape[2]:
                volume[rib_mask_slice, rib_z + dz] = 400 + np.random.randn() * 50

    # Add noise
    volume += np.random.randn(*shape) * 10

    # Clip to CT range
    volume = np.clip(volume, -1024, 3000)

    if output_path:
        save_nifti(volume, output_path, is_ct=True)
        print(f"  Saved: {output_path}")

    return volume


def create_lesion_mask(volume_shape, lesion_type='sphere', output_path=None):
    """
    Create a synthetic lesion mask.

    Args:
        volume_shape: Shape of the volume
        lesion_type: 'sphere', 'irregular', or 'multi'
    """
    print(f"Creating {lesion_type} lesion mask...")

    mask = np.zeros(volume_shape, dtype=np.uint8)
    center = np.array(volume_shape) // 2

    if lesion_type == 'sphere':
        # Single spherical lesion
        z, y, x = np.ogrid[:volume_shape[0], :volume_shape[1], :volume_shape[2]]
        lesion_center = center + np.array([10, -20, 15])  # Offset from center
        dist = np.sqrt((z - lesion_center[0])**2 +
                      (y - lesion_center[1])**2 +
                      (x - lesion_center[2])**2)
        mask[dist < 15] = 1

    elif lesion_type == 'irregular':
        # Irregular shaped lesion
        z, y, x = np.ogrid[:volume_shape[0], :volume_shape[1], :volume_shape[2]]
        lesion_center = center + np.array([-15, 25, -10])

        # Base ellipsoid
        dist = np.sqrt((z - lesion_center[0])**2 / 20**2 +
                      (y - lesion_center[1])**2 / 15**2 +
                      (x - lesion_center[2])**2 / 12**2)
        mask[dist < 1] = 1

        # Add some protrusions
        for _ in range(3):
            offset = np.random.randint(-10, 10, 3)
            dist2 = np.sqrt((z - lesion_center[0] - offset[0])**2 +
                           (y - lesion_center[1] - offset[1])**2 +
                           (x - lesion_center[2] - offset[2])**2)
            mask[dist2 < 8] = 1

    elif lesion_type == 'multi':
        # Multiple small lesions
        z, y, x = np.ogrid[:volume_shape[0], :volume_shape[1], :volume_shape[2]]

        lesion_centers = [
            center + np.array([20, -30, 10]),
            center + np.array([-10, 20, -15]),
            center + np.array([5, 10, 25]),
        ]

        for lc in lesion_centers:
            dist = np.sqrt((z - lc[0])**2 + (y - lc[1])**2 + (x - lc[2])**2)
            radius = np.random.randint(5, 12)
            mask[dist < radius] = 1

    if output_path:
        save_nifti(mask.astype(np.float32), output_path)
        print(f"  Saved: {output_path}")

    return mask


def create_2d_test_images(output_dir):
    """Create 2D test images (PNG/NPY)."""
    print("Creating 2D test images...")

    # Brain MRI slice
    mri_slice = create_synthetic_brain_mri(shape=(256, 256, 1))[:, :, 0]
    np.save(output_dir / "test_mri_slice.npy", mri_slice)
    print(f"  Saved: {output_dir / 'test_mri_slice.npy'}")

    # CT slice
    ct_slice = create_synthetic_ct(shape=(512, 512, 1))[:, :, 0]
    np.save(output_dir / "test_ct_slice.npy", ct_slice)
    print(f"  Saved: {output_dir / 'test_ct_slice.npy'}")

    # Circular mask
    mask = np.zeros((256, 256), dtype=np.float32)
    y, x = np.ogrid[:256, :256]
    mask[((y - 128)**2 + (x - 128)**2) < 40**2] = 1.0
    np.save(output_dir / "test_mask_circle.npy", mask)
    print(f"  Saved: {output_dir / 'test_mask_circle.npy'}")

    # Box mask
    mask_box = np.zeros((256, 256), dtype=np.float32)
    mask_box[80:180, 80:180] = 1.0
    np.save(output_dir / "test_mask_box.npy", mask_box)
    print(f"  Saved: {output_dir / 'test_mask_box.npy'}")


def save_nifti(data, output_path, is_ct=False):
    """Save numpy array as NIfTI file."""
    try:
        import nibabel as nib

        # Transpose to NIfTI convention (X, Y, Z)
        if data.ndim == 3:
            data = data.transpose(1, 2, 0)

        # Create affine matrix (identity with 1mm spacing)
        affine = np.eye(4)

        # Create NIfTI image
        nii = nib.Nifti1Image(data, affine)

        # Set header info
        nii.header.set_xyzt_units('mm', 'sec')

        if is_ct:
            nii.header['descrip'] = b'Synthetic CT scan'
        else:
            nii.header['descrip'] = b'Synthetic MRI scan'

        nib.save(nii, str(output_path))

    except ImportError:
        print("  Warning: nibabel not installed, saving as .npy instead")
        np.save(str(output_path).replace('.nii.gz', '.npy'), data)


def create_dicom_test(output_dir):
    """Create test DICOM file (if pydicom available)."""
    try:
        import pydicom
        from pydicom.dataset import Dataset, FileDataset
        from pydicom.uid import ExplicitVRLittleEndian
        import datetime

        print("Creating test DICOM file...")

        # Create a simple CT slice
        pixel_data = np.random.randint(0, 4096, (512, 512), dtype=np.uint16)

        # Create file meta
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        # Create dataset
        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Add required DICOM tags
        ds.PatientName = "Test^Patient"
        ds.PatientID = "TEST001"
        ds.Modality = "CT"
        ds.SeriesDescription = "Test CT Series"
        ds.StudyDescription = "Test Study"

        ds.Rows = 512
        ds.Columns = 512
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"

        ds.RescaleIntercept = -1024
        ds.RescaleSlope = 1

        ds.PixelData = pixel_data.tobytes()

        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

        # Save
        output_path = output_dir / "test_ct.dcm"
        ds.save_as(str(output_path))
        print(f"  Saved: {output_path}")

    except ImportError:
        print("  Skipping DICOM creation (pydicom not installed)")


def main():
    """Generate all test data."""
    print("=" * 60)
    print("CURIA Test Data Generator")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}\n")

    # Create 3D volumes
    create_synthetic_brain_mri(
        shape=(256, 256, 128),
        output_path=OUTPUT_DIR / "synthetic_brain_mri.nii.gz"
    )

    create_synthetic_ct(
        shape=(512, 512, 100),
        output_path=OUTPUT_DIR / "synthetic_chest_ct.nii.gz"
    )

    # Create lesion masks
    create_lesion_mask(
        (256, 256, 128),
        lesion_type='sphere',
        output_path=OUTPUT_DIR / "lesion_mask_sphere.nii.gz"
    )

    create_lesion_mask(
        (256, 256, 128),
        lesion_type='irregular',
        output_path=OUTPUT_DIR / "lesion_mask_irregular.nii.gz"
    )

    create_lesion_mask(
        (256, 256, 128),
        lesion_type='multi',
        output_path=OUTPUT_DIR / "lesion_mask_multi.nii.gz"
    )

    # Create 2D test images
    create_2d_test_images(OUTPUT_DIR)

    # Create DICOM test
    create_dicom_test(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Test data generation complete!")
    print("=" * 60)

    # List created files
    print("\nCreated files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size = f.stat().st_size / 1024
        if size > 1024:
            size_str = f"{size/1024:.1f} MB"
        else:
            size_str = f"{size:.1f} KB"
        print(f"  {f.name}: {size_str}")


if __name__ == "__main__":
    main()
