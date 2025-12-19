# Curia Model Capabilities Guide

**Model**: raidium/curia
**Type**: Multi-Modal Foundation Model for Radiology
**Architecture**: DINOv2-based vision transformer
**Parameters**: 86.1M
**Training Data**: 150,000 exams (130 TB) from hospital imaging

---

## Overview

Curia is a vision foundation model trained on cross-sectional medical imaging data (CT and MRI). It provides:

1. **Feature Extraction**: 768-dimensional embeddings for medical images
2. **Classification**: 15 specialized task heads for different clinical tasks
3. **Mask Prompting**: Region-specific analysis using spatial masks

---

## Available Task Heads

### Brain Imaging

| Head | Modality | Classes | Task | Input Data |
|------|----------|---------|------|------------|
| `atlas-stroke` | CT/MRI | 2 | Stroke detection (binary) | T1w brain MRI |
| `ich` | CT | 2 | Hemorrhage detection (binary) | Non-contrast head CT |
| `ixi` | MRI | 1 | Brain tissue analysis | T1/T2 brain MRI |
| `oasis` | MRI | 2 | Dementia detection (binary) | Structural brain MRI |

#### Class Labels (Verified from Model)

**OASIS (Dementia Detection)** - Binary classification
| Class | Label |
|-------|-------|
| 0 | Non-demented |
| 1 | Demented |

**ICH (Intracranial Hemorrhage)** - Binary detection
| Class | Label |
|-------|-------|
| 0 | No hemorrhage |
| 1 | Hemorrhage present |

**IXI** - Single output (regression/feature-based)
| Class | Label |
|-------|-------|
| 0 | Brain tissue |

**ATLAS-Stroke** - Binary detection
| Class | Label |
|-------|-------|
| 0 | No stroke |
| 1 | Stroke lesion present |

**Appropriate Data for Brain Tasks:**
- **ICH**: Non-contrast head CT (axial slices)
- **ATLAS-Stroke**: T1-weighted brain MRI
- **OASIS**: Structural T1-weighted brain MRI (1.5T or 3T)
- **IXI**: T1 or T2-weighted brain MRI

---

### Chest Imaging

| Head | Modality | Classes | Task | Input Data |
|------|----------|---------|------|------------|
| `covidx-ct` | CT | 3 | COVID-19 detection (normal/COVID/non-COVID) | Chest CT scans |
| `luna16-3D` | CT | 2 | Lung nodule detection (binary) | Low-dose chest CT |

**Appropriate Data for Chest Tasks:**
- Thin-slice chest CT (1-2mm)
- Include full lung fields
- Standard lung window for nodules

---

### Abdominal Imaging

| Head | Modality | Classes | Task | Input Data |
|------|----------|---------|------|------------|
| `abdominal-trauma` | CT | 2 | Trauma detection (binary) | Abdominal CT |
| `kits` | CT | 2 | Kidney tumor detection (binary) | Contrast-enhanced abdominal CT |
| `anatomy-ct` | CT | 54 | Organ identification (54 classes) | Any abdominal/pelvic CT |

**Appropriate Data for Abdominal Tasks:**
- Contrast-enhanced CT preferred for tumors
- Arterial or portal venous phase

---

### Spine Imaging

| Head | Modality | Classes | Task | Input Data |
|------|----------|---------|------|------------|
| `neural_foraminal_narrowing` | MRI | 3 | Neural foraminal stenosis (normal/moderate/severe) | Spine MRI (axial T2) |
| `spinal_canal_stenosis` | MRI | 3 | Spinal canal narrowing (normal/moderate/severe) | Spine MRI (axial T2) |
| `subarticular_stenosis` | MRI | 3 | Subarticular stenosis (normal/moderate/severe) | Spine MRI (axial T2) |

**Appropriate Data for Spine Tasks:**
- Axial T2-weighted MRI
- Lumbar or cervical spine
- 3-4mm slice thickness typical

---

### Musculoskeletal Imaging

| Head | Modality | Classes | Task | Input Data |
|------|----------|---------|------|------------|
| `kneeMRI` | MRI | 3 | Knee pathology (normal/partial/complete tear) | Knee MRI (any sequence) |

**Appropriate Data:**
- Standard knee MRI protocols
- Sagittal PD or T2 fat-sat preferred

---

### General Purpose

| Head | Modality | Classes | Task | Input Data |
|------|----------|---------|------|------------|
| `anatomy-ct` | CT | 54 | Organ/anatomy identification | Any CT scan |
| `anatomy-mri` | MRI | 56 | Organ/anatomy identification | Any MRI scan |
| `deep-lesion-site` | CT | 8 | Lesion anatomical site | Any CT with visible lesions |
| `emidec-classification-mask` | CT/MRI | 2 | Lesion classification with mask | Any scan with mask ROI |

---

## Output Types

### Classification Output
All heads output class probabilities (logits), NOT images.

```python
result = tester.classify(image)
# Returns: {
#   "logits": [...],           # Raw model outputs
#   "probabilities": [...],    # Softmax probabilities
#   "prediction": int          # Argmax class index
# }
```

### Attention Maps (Image Output)
Attention maps are the only image-like output, showing where the model focuses.

```python
attn_map = tester.get_attention_map(image)
# Returns: numpy array of shape (512, 512)
# Values: attention weights (higher = more focus)
```

### Feature Extraction
768-dimensional embeddings per slice.

```python
results = tester.process_volume(volume)
# Returns: {
#   "slice_features": [num_slices, 768],  # Per-slice features
#   "aggregated_features": [768],          # Mean across slices
#   "per_slice_predictions": [...]         # If classifier loaded
# }
```

---

## Input Requirements

### Image Format

```
Format: NumPy array
2D: (Height, Width)
3D: (Height, Width, Slices)
```

### Orientation Convention

| View | Code | Description |
|------|------|-------------|
| Axial | PL | Posterior-Left |
| Coronal | IL | Inferior-Left |
| Sagittal | IP | Inferior-Posterior |

### Preprocessing

| Modality | Preprocessing |
|----------|---------------|
| CT | Use raw Hounsfield units OR normalized values. Do NOT apply windowing before input. |
| MRI | Use raw intensity values OR normalized values. Do NOT apply windowing. |

**Note**: The model handles normalization internally. Provide unwindowed data.

---

## Modality-Head Compatibility

### CT-Only Heads
- `anatomy-ct`
- `covidx-ct`
- `deep-lesion-site`
- `ich`
- `kits`
- `luna16-3D`

### MRI-Only Heads
- `anatomy-mri`
- `ixi`
- `kneeMRI`
- `neural_foraminal_narrowing`
- `oasis`
- `spinal_canal_stenosis`
- `subarticular_stenosis`

### Both CT and MRI
- `atlas-stroke`
- `emidec-classification-mask`

---

## Feature Extraction

### What is Feature Extraction?

Feature extraction uses Curia's DINOv2 backbone to generate 768-dimensional embeddings for medical images. These can be used for:
- Similarity search between images
- Clustering similar scans
- Training custom classifiers
- Transfer learning for downstream tasks

### Slice-wise Feature Output

For 3D volumes, Curia extracts features **per-slice**:

```python
results = tester.process_volume(volume)

# Available outputs:
results["slice_features"]      # [num_slices, 768] - all slice features
results["slice_indices"]       # List of slice indices
results["num_slices"]          # Number of slices processed
results["feature_dim"]         # 768

# Aggregation methods:
results["aggregated_features"]     # Mean of all slice features
results["aggregated_features_max"] # Max pooling across slices
results["aggregated_features_std"] # Standard deviation
```

### Export Features

Features can be exported to JSON or CSV:

**JSON Export** - Full data with metadata:
```json
{
  "num_slices": 170,
  "feature_dim": 768,
  "slice_indices": [0, 1, 2, ...],
  "aggregated_features": {
    "mean": [...],
    "max": [...],
    "std": [...]
  },
  "slice_features": [[...], [...], ...],
  "per_slice_predictions": [...]
}
```

**CSV Export** - Tabular format:
```
slice_idx,feat_0,feat_1,...,feat_767
0,0.123,0.456,...,0.789
1,0.234,0.567,...,0.890
...
```

### GUI Usage

1. Load your image
2. Click "Extract Features"
3. Click "Export JSON" or "Export CSV"
4. Download the file

---

## Example Workflows

### 1. Brain Hemorrhage Detection (ICH)

```python
# Load non-contrast head CT
image = load_nifti("head_ct.nii.gz")

# Use ICH head
config = CuriaConfig(subfolder="ich")
tester = CuriaTester(config)
tester.load_model(load_classifier=True)

# Classify
result = tester.classify(image[:, :, slice_idx])
# Output: hemorrhage type (epidural, subdural, etc.)
```

### 2. Lung Nodule Detection

```python
# Load chest CT
image = load_nifti("chest_ct.nii.gz")

# Use luna16-3D head
config = CuriaConfig(subfolder="luna16-3D")
tester = CuriaTester(config)
tester.load_model(load_classifier=True)

# Process volume
result = tester.process_volume(image)
```

### 3. Anatomy Identification

```python
# Load any CT/MRI
image = load_nifti("scan.nii.gz")

# Use anatomy head (ct or mri)
config = CuriaConfig(subfolder="anatomy-ct")  # or "anatomy-mri"
tester = CuriaTester(config)
tester.load_model(load_classifier=True)

# Classify anatomy
result = tester.classify(image[:, :, slice_idx])
# Output: organ/anatomy label
```

### 4. Feature Extraction and Export

```python
from curia_tester import CuriaTester, CuriaConfig

# Load image
image = load_nifti("brain_mri.nii.gz")

# Feature extraction only (no classification head needed)
config = CuriaConfig()
tester = CuriaTester(config)
tester.load_model(load_classifier=False)

# Extract features from 3D volume
results = tester.process_volume(image)

# Access slice-wise features
print(f"Processed {results['num_slices']} slices")
print(f"Feature shape: {results['slice_features'].shape}")  # [num_slices, 768]

# Export to files
tester.export_features_to_json(results, "features.json")
tester.export_features_to_csv(results, "features.csv")
```

---

## Performance Benchmarks

| Task | Metric | Performance |
|------|--------|-------------|
| Anatomy Recognition | Accuracy | 98.1% |
| External Validation | Tasks Passed | 19/19 |

*Source: CuriaBench benchmark on external datasets*

---

## How Curia Processes Data

### 2D Slice-wise Processing

**Important**: Curia is a **2D vision transformer** that processes images slice-by-slice, NOT a true 3D model.

**For 3D volumes (NIfTI, DICOM series):**
1. The volume is loaded with shape (H, W, Z) where Z is the slice axis
2. Each slice is processed independently: `image[:, :, slice_idx]`
3. The GUI shows one slice at a time with a slider
4. Classification results are per-slice (use middle slice by default)
5. Attention maps are generated per-slice

**Best practices for 3D data:**
- Use the slice slider to find the most representative slice
- For brain MRI: axial slices through the region of interest
- For OASIS (dementia): structural T1w, use slices showing lateral ventricles
- For ICH: non-contrast CT, axial slices at hemorrhage level

---

## Limitations

1. **Research Only**: Not approved for clinical decision-making
2. **2D Processing**: Processes 3D volumes slice-by-slice, no true 3D context
3. **No Segmentation**: Provides classification, not pixel-level masks
4. **Fixed Resolution**: Best at 512x512 input size

---

## References

- Model: https://huggingface.co/raidium/curia
- Paper: https://arxiv.org/abs/2509.06830
- Benchmark: https://huggingface.co/datasets/raidium/CuriaBench
- GitHub: https://github.com/raidium-med/curia
