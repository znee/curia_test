# CURIA Testing Guide

A comprehensive guide for testing CURIA, a vision foundation model for radiology.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Overview](#overview)
3. [Installation](#installation)
4. [Model Architecture](#model-architecture)
5. [Available Classification Heads](#available-classification-heads)
6. [Input Formats](#input-formats)
7. [Preprocessing](#preprocessing)
8. [Test Data](#test-data)
9. [Prompting Methods](#prompting-methods)
10. [Usage Examples](#usage-examples)
11. [GUI / Web App](#gui--web-app)
12. [API Reference](#api-reference)
13. [Troubleshooting](#troubleshooting)
14. [Testing Workflows](#testing-workflows)

---

## Quick Start

Get up and running in 5 minutes:

### 1. Setup Environment

```bash
# Activate conda environment
conda activate curia

# Login to HuggingFace (required - CURIA is a gated model)
huggingface-cli login
```

### 2. Generate Test Data

```bash
python create_test_data.py
```

### 3. Launch GUI

```bash
python curia_gui.py
# Open http://localhost:7860 in your browser
```

### 4. Test Workflow (GUI)

1. Click **"Load Model"** (select `atlas-stroke` or `anatomy-mri` head)
2. Upload a test image from `test_data/` folder
3. Adjust slice slider if 3D volume
4. Click **"Classify"** or **"Extract Features"**

### 5. Test Workflow (Python)

```python
from curia_tester import CuriaTester, CuriaConfig

# Initialize
config = CuriaConfig(subfolder='atlas-stroke')
tester = CuriaTester(config)
tester.load_model(load_classifier=True)

# Load test data and classify
import numpy as np
image = np.load('test_data/test_mri_slice.npy')
result = tester.classify(image)
print(f"Prediction: {result['prediction']}, Confidence: {result['probabilities'].max():.2%}")
```

### Files Overview

| File | Description |
|------|-------------|
| `curia_gui.py` | Web-based GUI (Gradio) |
| `curia_tester.py` | Python API and CLI |
| `preprocessing.py` | Normalization and slice selection |
| `create_test_data.py` | Generate synthetic test data |
| `test_data/` | Generated test images |

---

## Overview

CURIA is a DINOv2-based foundation model trained on 150,000 radiology exams (130TB of data). It provides:

- **Feature extraction** from medical images (CT, MRI)
- **Classification** with pretrained task-specific heads
- **Mask-based prompting** for region-specific analysis

**Key Characteristics:**
- Input: 2D medical image slices (grayscale)
- Output: 768-dimensional feature vectors or classification logits
- Supported modalities: CT, MRI
- Image size: Resized to 512×512 internally

---

## Installation

### Environment Setup

```bash
# Create conda environment
conda create -n curia python=3.11 -y
conda activate curia

# Install PyTorch (choose based on your platform)
# For Apple Silicon (MPS):
pip install torch torchvision torchaudio

# For CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install transformers>=4.45 huggingface_hub>=0.34
pip install pydicom nibabel SimpleITK  # Medical imaging
pip install numpy Pillow scipy
```

### HuggingFace Authentication

CURIA is a gated model. You need to:

1. Request access at https://huggingface.co/raidium/curia
2. Login to HuggingFace:

```bash
huggingface-cli login
```

---

## Model Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CURIA Architecture                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input Image (H×W)                                      │
│       │                                                 │
│       ▼                                                 │
│  ┌─────────────────┐                                    │
│  │ Image Processor │  Resize to 512×512                 │
│  │ (Normalization) │  Z-score normalization             │
│  └────────┬────────┘                                    │
│           │                                             │
│           ▼                                             │
│  ┌─────────────────┐                                    │
│  │  DINOv2 ViT-B   │  Backbone (frozen)                 │
│  │   (Backbone)    │  Output: 1025 tokens × 768 dim     │
│  └────────┬────────┘  (1 CLS + 1024 patch tokens)       │
│           │                                             │
│           ▼                                             │
│  ┌─────────────────┐                                    │
│  │  Optional Mask  │  Extract features from ROI         │
│  │   Processing    │  (if mask provided)                │
│  └────────┬────────┘                                    │
│           │                                             │
│           ▼                                             │
│  ┌─────────────────┐                                    │
│  │ Classification  │  Task-specific head                │
│  │      Head       │  (e.g., stroke detection)          │
│  └────────┬────────┘                                    │
│           │                                             │
│           ▼                                             │
│      Predictions                                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Available Classification Heads

| Head Name | Task | Classes | Modality |
|-----------|------|---------|----------|
| `anatomy-ct` | CT anatomical structure | 54 | CT |
| `anatomy-mri` | MRI anatomical structure | 56 | MRI |
| `atlas-stroke` | Stroke detection | 2 | MRI |
| `covidx-ct` | COVID-19 detection | 2 | CT |
| `ich` | Intracranial hemorrhage | 2 | CT |
| `luna16-3D` | Lung nodule (3D) | 2 | CT |
| `kits` | Kidney tumor | 2 | CT |
| `kneeMRI` | Knee pathology | varies | MRI |
| `oasis` | Brain MRI analysis | varies | MRI |
| `spinal_canal_stenosis` | Spinal stenosis grading | varies | MRI |
| `neural_foraminal_narrowing` | Neural foramen narrowing | varies | MRI |
| `subarticular_stenosis` | Subarticular stenosis | varies | MRI |
| `deep-lesion-site` | Lesion site classification | varies | CT |
| `ixi` | Brain MRI | varies | MRI |
| `emidec-classification-mask` | Cardiac classification | varies | MRI |

---

## Input Formats

### Supported File Types

| Format | Extension | Library Used |
|--------|-----------|--------------|
| DICOM | `.dcm`, directory | pydicom, SimpleITK |
| NIfTI | `.nii`, `.nii.gz` | nibabel, SimpleITK |
| NumPy | `.npy`, `.npz` | numpy |
| Images | `.png`, `.jpg` | PIL |

### Image Orientation Requirements

| View | Orientation | Description |
|------|-------------|-------------|
| Axial | PL (Posterior-Left) | Standard axial CT/MRI |
| Coronal | IL (Inferior-Left) | Coronal view |
| Sagittal | IP (Inferior-Posterior) | Sagittal view |

### Intensity Values

- **CT**: Hounsfield Units (HU), raw values preferred
- **MRI**: Raw intensity values, no windowing
- **Normalization**: Handled automatically by the processor (z-score)

---

## Preprocessing

CURIA includes preprocessing utilities for normalization and slice selection.

### Normalization

#### CT Windowing

For CT images, apply Hounsfield Unit (HU) windowing:

```python
from preprocessing import Normalizer, CT_WINDOWS

# Available CT windows
print(CT_WINDOWS.keys())
# brain, brain_stroke, subdural, bone, soft_tissue, lung, lung_nodule,
# mediastinum, abdomen, liver, spine, angio

# Apply brain window
normalizer = Normalizer(modality='ct', window='brain')
normalized = normalizer.normalize_ct(ct_image)  # Output: 0-1 range

# Custom window
from preprocessing import WindowSettings
custom_window = WindowSettings(center=50, width=100, name="custom")
normalizer.window = custom_window
```

| Window | Center | Width | Use Case |
|--------|--------|-------|----------|
| brain | 40 | 80 | Brain parenchyma |
| brain_stroke | 40 | 40 | Acute stroke detection |
| subdural | 75 | 215 | Subdural hemorrhage |
| bone | 400 | 2000 | Bone structures |
| soft_tissue | 40 | 400 | General soft tissue |
| lung | -600 | 1500 | Lung parenchyma |
| abdomen | 40 | 350 | Abdominal organs |

#### MRI Normalization

For MRI, percentile-based intensity normalization:

```python
normalizer = Normalizer(
    modality='mri',
    percentile_low=1.0,
    percentile_high=99.0
)
normalized = normalizer.normalize_mri(mri_image)
```

### Slice Selection

Select informative slices from 3D volumes:

```python
from preprocessing import SliceSelector

selector = SliceSelector(method='uniform', num_slices=10)

# Methods: 'uniform', 'content', 'center', 'all'
indices = selector.select(volume)  # Returns list of slice indices

# Content-based: selects slices with most anatomical content
selector = SliceSelector(method='content', num_slices=10)
indices = selector.select_content_based(volume)
```

### Complete Pipeline

```python
from preprocessing import Preprocessor

# Configure preprocessing
preprocessor = Preprocessor(
    modality='ct',
    ct_window='brain',
    target_size=(512, 512),
    slice_method='uniform',
    num_slices=10
)

# Process single slice
processed = preprocessor.process_slice(image, normalize=True, resize=True)

# Process 3D volume
result = preprocessor.process_volume(volume)
print(result['slices'])         # List of processed 2D slices
print(result['slice_indices'])  # Which slices were selected
print(result['modality'])       # Detected modality
```

---

## Test Data

Generate synthetic test data for development and testing.

### Generate Test Data

```bash
python create_test_data.py
```

This creates the following files in `test_data/`:

| File | Description | Shape |
|------|-------------|-------|
| `synthetic_brain_mri.nii.gz` | Synthetic brain MRI volume | 256×256×128 |
| `synthetic_chest_ct.nii.gz` | Synthetic chest CT volume | 512×512×100 |
| `lesion_mask_sphere.nii.gz` | Spherical lesion mask | 256×256×128 |
| `lesion_mask_irregular.nii.gz` | Irregular lesion mask | 256×256×128 |
| `lesion_mask_multi.nii.gz` | Multiple lesion mask | 256×256×128 |
| `test_mri_slice.npy` | 2D MRI slice | 256×256 |
| `test_ct_slice.npy` | 2D CT slice | 512×512 |
| `test_mask_circle.npy` | Circular mask | 256×256 |
| `test_mask_box.npy` | Box mask | 256×256 |
| `test_ct.dcm` | Test DICOM file | 512×512 |

### Using Test Data

```python
import nibabel as nib
import numpy as np

# Load NIfTI test data
mri = nib.load('test_data/synthetic_brain_mri.nii.gz')
mri_data = mri.get_fdata()

# Load mask
mask = nib.load('test_data/lesion_mask_sphere.nii.gz')
mask_data = mask.get_fdata()

# Load 2D test data
ct_slice = np.load('test_data/test_ct_slice.npy')
```

---

## Prompting Methods

CURIA supports **mask-based prompting** to focus analysis on specific regions.

### 1. Point Prompting

Click on a point to create a circular region of interest:

```python
import numpy as np

def create_point_mask(image_shape, point, radius=20):
    """Create circular mask around a point."""
    mask = np.zeros(image_shape, dtype=np.float32)
    y, x = np.ogrid[:image_shape[0], :image_shape[1]]
    dist = (x - point[0])**2 + (y - point[1])**2
    mask[dist <= radius**2] = 1.0
    return mask

# Example: focus on point (150, 120)
mask = create_point_mask((256, 256), (150, 120), radius=30)
```

### 2. Box Prompting

Define a bounding box region:

```python
def create_box_mask(image_shape, box):
    """Create rectangular mask from bounding box (x1, y1, x2, y2)."""
    mask = np.zeros(image_shape, dtype=np.float32)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = 1.0
    return mask

# Example: focus on region from (50,50) to (200,200)
mask = create_box_mask((256, 256), (50, 50, 200, 200))
```

### 3. Loaded Mask

Use an existing segmentation mask:

```python
import nibabel as nib

# Load mask from NIfTI file
mask_nii = nib.load('lesion_mask.nii.gz')
mask = mask_nii.get_fdata()
mask = (mask > 0).astype(np.float32)  # Binarize
```

### 4. Using Masks with the Model

```python
import torch
import torch.nn.functional as F

# Prepare mask for model input
mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
mask_tensor = F.interpolate(mask_tensor, size=(512, 512), mode='bilinear')
mask_tensor = (mask_tensor > 0.5).float()

# Add to model inputs
inputs = processor(image, return_tensors='pt')
inputs['mask'] = mask_tensor.to(device)

# Run inference
outputs = model(**inputs)
```

---

## Usage Examples

### Example 1: Basic Feature Extraction

```python
import numpy as np
import torch
from transformers import AutoModel, AutoImageProcessor

# Load model
processor = AutoImageProcessor.from_pretrained('raidium/curia', trust_remote_code=True)
model = AutoModel.from_pretrained('raidium/curia', trust_remote_code=True)
model.eval()

# Load image (example: random data)
image = np.random.rand(256, 256).astype(np.float32)

# Process
inputs = processor(image, return_tensors='pt')

# Extract features
with torch.no_grad():
    outputs = model(**inputs)

# Get CLS token (global image representation)
cls_features = outputs.last_hidden_state[:, 0, :]  # Shape: (1, 768)

# Get patch tokens (spatial features)
patch_features = outputs.last_hidden_state[:, 1:, :]  # Shape: (1, 1024, 768)
```

### Example 2: Classification with Head

```python
from transformers import AutoModelForImageClassification

# Load with classification head
model = AutoModelForImageClassification.from_pretrained(
    'raidium/curia',
    subfolder='atlas-stroke',  # Stroke detection
    trust_remote_code=True
)
model.eval()

# Process image
inputs = processor(image, return_tensors='pt')

# Classify
with torch.no_grad():
    outputs = model(**inputs)

# Get predictions
logits = outputs['logits']
prediction = logits.argmax(-1).item()
probabilities = torch.softmax(logits, dim=-1)

print(f"Prediction: {prediction}")  # 0=no stroke, 1=stroke
print(f"Confidence: {probabilities.max().item():.2%}")
```

### Example 3: Classification with Mask Prompting

```python
import torch.nn.functional as F

# Create mask for region of interest
mask = np.zeros((256, 256), dtype=np.float32)
mask[100:200, 100:200] = 1.0  # Focus on center region

# Prepare mask
mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
mask_tensor = F.interpolate(mask_tensor, size=(512, 512), mode='bilinear')
mask_tensor = (mask_tensor > 0.5).float()

# Process with mask
inputs = processor(image, return_tensors='pt')
inputs['mask'] = mask_tensor

with torch.no_grad():
    outputs = model(**inputs)

print(f"ROI Prediction: {outputs['logits'].argmax(-1).item()}")
```

### Example 4: Processing DICOM Files

```python
import pydicom
import numpy as np

# Load DICOM
dcm = pydicom.dcmread('scan.dcm')
image = dcm.pixel_array.astype(np.float32)

# Apply rescale if available
if hasattr(dcm, 'RescaleSlope'):
    image = image * dcm.RescaleSlope + dcm.RescaleIntercept

# Process
inputs = processor(image, return_tensors='pt')
outputs = model(**inputs)
```

### Example 5: Processing NIfTI Volumes

```python
import nibabel as nib

# Load NIfTI
nii = nib.load('brain_mri.nii.gz')
volume = nii.get_fdata().astype(np.float32)

# Process middle slice
mid_slice = volume.shape[2] // 2
image = volume[:, :, mid_slice]

inputs = processor(image, return_tensors='pt')
outputs = model(**inputs)
```

### Example 6: Batch Processing

```python
# Process multiple slices
slices = [volume[:, :, i] for i in range(0, volume.shape[2], 5)]

# Batch process
results = []
for slice_img in slices:
    inputs = processor(slice_img, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    results.append({
        'prediction': outputs['logits'].argmax(-1).item(),
        'probability': torch.softmax(outputs['logits'], dim=-1).max().item()
    })
```

---

## GUI / Web App

CURIA includes a Gradio-based web interface for interactive testing.

### Starting the GUI

```bash
python curia_gui.py
```

Then open http://localhost:7860 in your browser.

### GUI Features

1. **Model Configuration**
   - Select classification head (atlas-stroke, anatomy-mri, etc.)
   - Choose device (auto, cuda, mps, cpu)
   - Enter HuggingFace token if needed

2. **Preprocessing Options**
   - Select modality (Auto, CT, MRI)
   - Choose CT window preset (brain, lung, soft_tissue, etc.)
   - Toggle preprocessing on/off

3. **Image Input**
   - Upload DICOM, NIfTI, or NPY files
   - Upload mask files for region-specific analysis
   - Click on image to create point prompts

4. **Slice Navigation** (for 3D volumes)
   - Slider to navigate through slices
   - Auto-select informative slices
   - Selection methods: Uniform, Content-based, Center

5. **Analysis**
   - Extract features (768-dim vectors)
   - Classify with selected head
   - Visualize attention maps

### GUI Screenshots

```
┌─────────────────────────────────────────────────────────┐
│  CURIA Medical Image Tester                             │
├─────────────────┬───────────────────────────────────────┤
│ Model Config    │  Image Input                          │
│ ─────────────── │  ─────────────────────────────────────│
│ Head: atlas-    │  [Upload Image]  [Upload Mask]        │
│       stroke    │                                       │
│ Device: auto    │  ┌─────────────┐ ┌─────────────┐      │
│ Token: ****     │  │   Image     │ │    Mask     │      │
│                 │  │   Preview   │ │   Preview   │      │
│ [Load Model]    │  └─────────────┘ └─────────────┘      │
│                 │                                       │
│ Preprocessing   │  Slice: [────────●──────] 64/128     │
│ ─────────────── │                                       │
│ Modality: Auto  │  Auto-Select: [Uniform ▼] [10 slices]│
│ Window: brain   │                                       │
│ ☑ Apply Preproc │                                       │
├─────────────────┴───────────────────────────────────────┤
│ Analysis                                                │
│ [Extract Features] [Classify] [Get Attention Map]       │
└─────────────────────────────────────────────────────────┘
```

---

## API Reference

### Command Line Interface

```bash
# Basic usage
python curia_tester.py --input <path> --mode <mode> [options]

# Options:
#   --input, -i      Input image path (DICOM, NIfTI, NPY)
#   --mask, -m       Mask file for prompting
#   --points         Point prompts as "x1,y1;x2,y2"
#   --boxes          Box prompts as "x1,y1,x2,y2;..."
#   --mode           Processing mode: features, classify, attention
#   --head           Classification head name
#   --output, -o     Output file path
#   --device         Device: auto, cuda, mps, cpu
#   --interactive    Start interactive session
```

### Examples

```bash
# Extract features
python curia_tester.py --input scan.dcm --mode features --output features.npz

# Classify with stroke detection
python curia_tester.py --input brain.nii.gz --mode classify --head atlas-stroke

# Classify with mask prompting
python curia_tester.py --input scan.dcm --mask lesion.nii.gz --mode classify --head anatomy-ct

# Point prompting
python curia_tester.py --input scan.dcm --points "100,100;200,150" --mode classify --head atlas-stroke

# Box prompting
python curia_tester.py --input scan.dcm --boxes "50,50,200,200" --mode features

# Interactive mode
python curia_tester.py --interactive

# Start GUI
python curia_gui.py
```

### Python API

```python
from curia_tester import CuriaTester, CuriaConfig

# Configuration
config = CuriaConfig(
    model_name="raidium/curia",
    subfolder="atlas-stroke",  # Classification head
    crop_size=512,
    device="auto"  # auto, cuda, mps, cpu
)

# Initialize tester
tester = CuriaTester(config)
tester.load_model(load_classifier=True)

# Load image
image, metadata = tester.loader.load("scan.dcm")

# Extract features
features = tester.extract_features(image, mask=None)

# Classify
result = tester.classify(image, mask=None)
print(f"Prediction: {result['prediction']}")
print(f"Probabilities: {result['probabilities']}")

# Get attention map
attention = tester.get_attention_map(image)
```

---

## Troubleshooting

### Common Issues

#### 1. Authentication Error (401)

```
GatedRepoError: 401 Client Error. Access to model raidium/curia is restricted.
```

**Solution:**
```bash
# Login to HuggingFace
huggingface-cli login

# Or set token in environment
export HF_TOKEN="your_token_here"
```

#### 2. Model Download Stuck

**Solution:**
```bash
# Clear incomplete downloads
rm -f ~/.cache/huggingface/hub/models--raidium--curia/blobs/*.incomplete

# Retry download
python -c "from huggingface_hub import snapshot_download; snapshot_download('raidium/curia')"
```

#### 3. CUDA Out of Memory

**Solution:**
- Reduce batch size
- Use CPU for preprocessing
- Process slices sequentially

```python
# Process on CPU, move to GPU only for inference
inputs = processor(image, return_tensors='pt')  # CPU
inputs = {k: v.to(device) for k, v in inputs.items()}  # GPU
```

#### 4. MPS (Apple Silicon) Issues

```python
# If MPS has issues, fall back to CPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# For specific MPS errors
import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
```

#### 5. Slow Inference

**Solutions:**
- Ensure model is on GPU: `model.to(device)`
- Use `torch.no_grad()` for inference
- Batch multiple images when possible

```python
# Verify GPU usage
print(f"Model device: {next(model.parameters()).device}")
print(f"Input device: {inputs['pixel_values'].device}")
```

### Performance Benchmarks

| Device | Image Size | Time per Image |
|--------|-----------|----------------|
| Apple M1 (MPS) | 512×512 | ~40ms |
| NVIDIA RTX 3090 | 512×512 | ~15ms |
| CPU (i9) | 512×512 | ~200ms |

---

## Testing Workflows

### Workflow 1: Stroke Detection on Brain MRI

```bash
# Step 1: Generate test data
python create_test_data.py

# Step 2: Test via CLI
python curia_tester.py \
    --input test_data/synthetic_brain_mri.nii.gz \
    --mode classify \
    --head atlas-stroke
```

Or via Python:

```python
from curia_tester import CuriaTester, CuriaConfig
import nibabel as nib

# Load model with stroke detection head
config = CuriaConfig(subfolder='atlas-stroke')
tester = CuriaTester(config)
tester.load_model(load_classifier=True)

# Load brain MRI
nii = nib.load('test_data/synthetic_brain_mri.nii.gz')
volume = nii.get_fdata()

# Process volume (analyzes multiple slices)
results = tester.process_volume(volume)
print(f"Analyzed {len(results['slice_indices'])} slices")
print(f"Features shape: {results['aggregated_features'].shape}")
```

### Workflow 2: Anatomy Classification with Region Prompting

```python
from curia_tester import CuriaTester, CuriaConfig, MaskPrompt
import numpy as np

# Load model with anatomy head
config = CuriaConfig(subfolder='anatomy-mri')
tester = CuriaTester(config)
tester.load_model(load_classifier=True)

# Load image
image = np.load('test_data/test_mri_slice.npy')

# Create mask for specific region (point prompt)
prompt = MaskPrompt()
mask = prompt.create_point_mask(image.shape, [(128, 128)], radius=30)

# Classify with mask
result = tester.classify(image, mask=mask)
print(f"Anatomy prediction: {result['prediction']}")
print(f"Top probabilities: {result['probabilities'][0][:5]}")
```

### Workflow 3: CT Lung Analysis

```python
from curia_tester import CuriaTester, CuriaConfig
from preprocessing import Preprocessor, CT_WINDOWS
import nibabel as nib

# Load CT volume
nii = nib.load('test_data/synthetic_chest_ct.nii.gz')
volume = nii.get_fdata()

# Preprocess with lung window
preprocessor = Preprocessor(modality='ct', ct_window='lung')
result = preprocessor.process_volume(volume, num_slices=10)

print(f"Selected slices: {result['slice_indices']}")
print(f"Processed {len(result['slices'])} slices")

# Extract features for each slice
config = CuriaConfig()
tester = CuriaTester(config)
tester.load_model()

features = []
for slice_img in result['slices']:
    feat = tester.extract_features(slice_img)
    features.append(feat['last_hidden_state'][:, 0, :])  # CLS token

print(f"Extracted features from {len(features)} slices")
```

### Workflow 4: Lesion Analysis with Mask

```python
import nibabel as nib
import numpy as np
from curia_tester import CuriaTester, CuriaConfig

# Load image and lesion mask
mri = nib.load('test_data/synthetic_brain_mri.nii.gz').get_fdata()
lesion_mask = nib.load('test_data/lesion_mask_sphere.nii.gz').get_fdata()

# Get middle slice
mid = mri.shape[2] // 2
slice_img = mri[:, :, mid]
slice_mask = lesion_mask[:, :, mid]

# Initialize model
config = CuriaConfig(subfolder='atlas-stroke')
tester = CuriaTester(config)
tester.load_model(load_classifier=True)

# Classify lesion region
result = tester.classify(slice_img, mask=slice_mask)
print(f"Lesion classification: {result['prediction']}")
print(f"Confidence: {result['probabilities'].max():.2%}")
```

### Workflow 5: Batch Processing Multiple Files

```python
from pathlib import Path
from curia_tester import CuriaTester, CuriaConfig, MedicalImageLoader
import json

# Setup
config = CuriaConfig(subfolder='anatomy-mri')
tester = CuriaTester(config)
tester.load_model(load_classifier=True)
loader = MedicalImageLoader()

# Process all NIfTI files in test_data
test_dir = Path('test_data')
results = {}

for nii_file in test_dir.glob('*.nii.gz'):
    print(f"Processing {nii_file.name}...")

    image, metadata = loader.load(str(nii_file))

    # Handle 3D volumes
    if image.ndim == 3:
        image = image[:, :, image.shape[2] // 2]

    result = tester.classify(image)
    results[nii_file.name] = {
        'prediction': int(result['prediction']),
        'confidence': float(result['probabilities'].max())
    }

# Save results
with open('batch_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Processed {len(results)} files")
```

### Workflow 6: Feature Extraction for Downstream Tasks

```python
import numpy as np
from curia_tester import CuriaTester, CuriaConfig

# Load base model (no classification head)
config = CuriaConfig()
tester = CuriaTester(config)
tester.load_model(load_classifier=False)

# Extract features from multiple images
images = [
    np.load('test_data/test_mri_slice.npy'),
    np.load('test_data/test_ct_slice.npy'),
]

all_features = []
for img in images:
    features = tester.extract_features(img)
    cls_token = features['last_hidden_state'][:, 0, :].numpy()  # (1, 768)
    all_features.append(cls_token)

# Stack features for downstream use
feature_matrix = np.vstack(all_features)  # (N, 768)
print(f"Feature matrix shape: {feature_matrix.shape}")

# Save for downstream tasks (classification, clustering, etc.)
np.save('curia_features.npy', feature_matrix)
```

---

## References

- **Paper**: [CURIA: A Foundation Model for Radiology](https://arxiv.org/abs/2509.06830)
- **Model**: [HuggingFace - raidium/curia](https://huggingface.co/raidium/curia)
- **Dataset**: [HuggingFace - raidium/CuriaBench](https://huggingface.co/datasets/raidium/CuriaBench)
- **GitHub**: [raidium-med/curia](https://github.com/raidium-med/curia)

---

## License

CURIA is released under **CC BY-NC-ND 4.0** (Creative Commons Attribution-NonCommercial-NoDerivatives).

- **Research use**: Allowed
- **Commercial use**: Not allowed
- **Clinical use**: Not allowed (research only)
