# CURIA Testing Framework

A comprehensive testing and evaluation framework for [CURIA](https://huggingface.co/raidium/curia), a DINOv2-based vision foundation model for radiology trained on 150,000 exams (130TB).

## Quick Start

```bash
# Activate environment
source /home/jinheej7/curia/venv/bin/activate

# Login to HuggingFace (required - gated model)
huggingface-cli login

# Launch GUI
cd /home/jinheej7/curia/curia_test
python curia_gui.py
# Open http://localhost:7860
```

## Available Classification Heads

All heads output **class probabilities** (not images). Verified class counts from model:

### Brain Imaging
| Head | Classes | Task | Input |
|------|---------|------|-------|
| `oasis` | 2 | Dementia detection (non-demented/demented) | Structural T1 MRI |
| `ich` | 2 | Hemorrhage detection (no/yes) | Non-contrast head CT |
| `atlas-stroke` | 2 | Stroke detection (no/yes) | T1w brain MRI |
| `ixi` | 1 | Brain tissue analysis | T1/T2 brain MRI |

### Chest Imaging
| Head | Classes | Task | Input |
|------|---------|------|-------|
| `covidx-ct` | 3 | COVID detection (normal/COVID/non-COVID) | Chest CT |
| `luna16-3D` | 2 | Lung nodule detection (no/yes) | Low-dose chest CT |

### Abdominal Imaging
| Head | Classes | Task | Input |
|------|---------|------|-------|
| `abdominal-trauma` | 2 | Trauma detection (no/yes) | Abdominal CT |
| `kits` | 2 | Kidney tumor detection (no/yes) | Contrast CT |
| `anatomy-ct` | 54 | Organ identification | Any CT |

### Spine Imaging
| Head | Classes | Task | Input |
|------|---------|------|-------|
| `neural_foraminal_narrowing` | 3 | Stenosis severity (normal/moderate/severe) | Spine MRI |
| `spinal_canal_stenosis` | 3 | Stenosis severity (normal/moderate/severe) | Spine MRI |
| `subarticular_stenosis` | 3 | Stenosis severity (normal/moderate/severe) | Spine MRI |

### Musculoskeletal
| Head | Classes | Task | Input |
|------|---------|------|-------|
| `kneeMRI` | 3 | Knee pathology (normal/partial/complete tear) | Knee MRI |

### General Purpose
| Head | Classes | Task | Input |
|------|---------|------|-------|
| `anatomy-mri` | 56 | Organ identification | Any MRI |
| `deep-lesion-site` | 8 | Lesion anatomical site | CT with lesions |
| `emidec-classification-mask` | 2 | Lesion classification | CT/MRI with mask |

---

## How Masks Work

Masks enable **region-specific analysis** by focusing the model on particular areas.

### Mask Types

```
┌─────────────────────────────────────────────────────────────┐
│                      MASK PROMPTING                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INPUT                    MASK                   OUTPUT     │
│  ┌─────────┐             ┌─────────┐                        │
│  │         │             │  ████   │                        │
│  │  Image  │      +      │  ████   │   →   Classification   │
│  │         │             │         │        (masked region) │
│  └─────────┘             └─────────┘                        │
│                                                             │
│  Mask = Binary array (0 or 1)                               │
│  1 = Region of interest                                     │
│  0 = Ignored                                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Creating Masks

**1. Point Prompts (GUI)**
```
Click on image → Creates circular mask (radius=15px)
Multiple clicks → Combined into single mask
```

**2. Box Prompts (CLI)**
```python
# Create rectangular mask
mask = tester.mask_prompt.create_box_mask(
    image_shape=(H, W),
    boxes=[(x1, y1, x2, y2)]
)
```

**3. Load External Mask**
```python
# From NIfTI, NumPy, or image file
mask = tester.mask_prompt.load_mask("mask.nii.gz")
```

### Mask Processing Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    MASK PROCESSING                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. RESIZE                                                  │
│     Mask resized to 512x512 (model input size)              │
│                                                             │
│  2. TOKENIZE                                                │
│     Image divided into 32x32 = 1024 patches (16px each)     │
│     Mask downsampled to patch grid                          │
│                                                             │
│  3. EXTRACT                                                 │
│     Masked patches → Average pooled features                │
│     OR: Cross-attention with mask query                     │
│                                                             │
│  4. CLASSIFY                                                │
│     Masked features → Classification head → Probabilities   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2D vs 3D Masks

| Mask Type | Shape | Behavior |
|-----------|-------|----------|
| **2D mask** | `(H, W)` | Same mask applied to ALL slices |
| **3D mask** | `(H, W, Z)` | Per-slice masks, indexed `mask[:, :, slice_idx]` |

```python
# 2D mask - broadcast to all slices
mask_2d = np.zeros((256, 256), dtype=np.float32)
mask_2d[100:150, 100:150] = 1.0

# 3D mask - different mask per slice
mask_3d = np.zeros((256, 256, 100), dtype=np.float32)
mask_3d[100:150, 100:150, 50:60] = 1.0  # Only slices 50-60

# Use with volume
results = tester.process_volume(volume, mask=mask_2d)  # Same mask all slices
results = tester.process_volume(volume, mask=mask_3d)  # Per-slice masks
```

### When to Use Masks

| Use Case | Mask Type | Example |
|----------|-----------|---------|
| Lesion classification | Point/Box | Click on tumor, classify malignancy |
| Region-specific anatomy | Loaded mask | Segment liver, classify anatomy |
| Attention focus | Any | Focus model on specific organ |
| Exclude artifacts | Inverse mask | Mask out metal artifacts |

---

## Extract Features vs Classify

| Aspect | Extract Features | Classify |
|--------|------------------|----------|
| **Output** | 768-dimensional vector per slice | Class probabilities + prediction |
| **Classifier needed** | No (uses backbone only) | Yes (requires classification head) |
| **3D behavior** | Processes ALL slices | Processes CURRENT slice only |
| **Purpose** | Downstream tasks, similarity search, clustering | Get diagnostic prediction |

### Extract Features
Uses Curia's DINOv2 backbone to generate **embeddings** (numerical representations):
```
Image → Backbone → [768 numbers]
```
- Output: Dense feature vector capturing image content
- For 3D volumes: Returns features for ALL slices `[num_slices, 768]`
- Use cases: Train your own classifier, find similar images, clustering

### Classify
Uses backbone + task-specific classification head:
```
Image → Backbone → Classification Head → [class probabilities]
```
- Output: Probability for each class (e.g., "Demented: 85%, Non-demented: 15%")
- For 3D volumes: Runs on **current slice only** (interactive, slice-wise)
- Use cases: Get diagnostic prediction for specific clinical task

### When to Use Which

| Goal | Use |
|------|-----|
| Get a diagnosis/prediction | Classify |
| Build custom ML model | Extract Features |
| Compare/cluster scans | Extract Features |
| Explore model attention | Attention Map |

---

## Output Types

### 1. Classification (All Heads)
```python
result = tester.classify(image, mask=None)
# Returns:
{
    "logits": np.array([...]),        # Raw outputs
    "probabilities": np.array([...]), # Softmax probs
    "prediction": int                  # Argmax class
}
```

### 2. Attention Map (Image Output)
```python
attn_map = tester.get_attention_map(image)
# Returns: np.array shape (512, 512)
# Heatmap showing where model focuses
```

### 3. Feature Extraction
```python
# Single slice
features = tester.extract_features(image, mask=None)
# Returns: {"last_hidden_state": [1, 1025, 768], ...}

# 3D volume - slice-wise features
results = tester.process_volume(volume)
# Returns:
{
    "slice_features": np.array([num_slices, 768]),
    "aggregated_features": torch.Tensor([768]),  # mean
    "aggregated_features_max": torch.Tensor([768]),
    "aggregated_features_std": torch.Tensor([768]),
    "per_slice_predictions": [...]  # if classifier loaded
}
```

### 4. Export Features
```python
# Export to JSON (full data)
tester.export_features_to_json(results, "features.json")

# Export to CSV (tabular)
tester.export_features_to_csv(results, "features.csv")
```

---

## Data Format

### Volume Convention: `(H, W, Z)`

- **H**: Height (rows)
- **W**: Width (columns)
- **Z**: Slices (last axis)

```python
# Correct slice access
slice_img = volume[:, :, slice_idx]

# NOT this (wrong axis)
slice_img = volume[slice_idx]  # Wrong!
```

### Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| NIfTI | `.nii`, `.nii.gz` | Use `.gz` in Gradio uploads |
| DICOM | `.dcm`, folder | Single file or series |
| NumPy | `.npy`, `.npz` | Direct array loading |

---

## Project Structure

```
curia_test/
├── README.md                 # This file
├── CURIA_CAPABILITIES.md     # Detailed head documentation
├── curia_tester.py           # Core API and CLI
├── curia_gui.py              # Gradio web interface
├── preprocessing.py          # Normalization, slice selection
├── save_visualization.py     # Save attention overlays
├── create_test_data.py       # Synthetic data generator
├── data/                     # Your uploaded scans
├── test_data/                # Synthetic test samples
└── outputs/                  # Exported features/visualizations
```

---

## GUI Features

1. **Load Model**: Select head, device, authenticate
2. **Load Image**: Upload NIfTI/DICOM/NumPy
3. **Slice Navigation**: Browse 3D volumes with slider
4. **Preprocessing**: CT windows, normalization
5. **Mask Prompting**: Click to add point masks
6. **Analysis**:
   - **Extract Features** → Processes all slices, export JSON/CSV
   - **Classify** → Prediction + attention map for current slice

---

## CLI Usage

```bash
# Interactive mode
python curia_tester.py --interactive

# Classify single image
python curia_tester.py -i scan.nii.gz --mode classify --head oasis

# Extract features
python curia_tester.py -i scan.nii.gz --mode features -o features.npz

# With mask
python curia_tester.py -i scan.nii.gz --mask roi.nii.gz --mode classify --head ich

# Point prompts
python curia_tester.py -i scan.nii.gz --points "100,100;150,150" --mode features
```

---

## References

- [CURIA Paper](https://arxiv.org/abs/2509.06830)
- [HuggingFace Model](https://huggingface.co/raidium/curia)
- [GitHub Repository](https://github.com/raidium-med/curia)

## License

- **CURIA Model**: Research-only RAIL-M License
- **This Framework**: MIT
