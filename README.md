# CURIA Testing Framework

A comprehensive testing and evaluation framework for [CURIA](https://arxiv.org/abs/2509.06830), a vision foundation model for radiology trained on 150,000 exams (130TB).

## Quick Start

```bash
# Activate environment
conda activate curia

# Login to HuggingFace (required - gated model)
huggingface-cli login

# Generate test data
python create_test_data.py

# Launch GUI
python curia_gui.py
# Open http://localhost:7860
```

## Project Structure

```
2026_curia/
├── README.md                 # This file
├── CURIA_TESTING_GUIDE.md   # Detailed usage documentation
├── requirements.txt          # Python dependencies
│
├── curia_tester.py          # Core testing API and CLI
├── curia_gui.py             # Gradio web interface
├── preprocessing.py          # Normalization and slice selection
├── create_test_data.py      # Synthetic test data generator
│
├── curia_repo/              # Cloned CURIA source (reference)
├── data/                    # Real test data
│   └── T1.nii.gz           # Sample T1 MRI
└── test_data/               # Generated synthetic data
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CURIA Testing Framework                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   curia_gui  │    │ curia_tester │    │preprocessing │       │
│  │   (Gradio)   │───▶│   (Core)     │◀───│  (Utils)     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         │                   ▼                   │                │
│         │           ┌──────────────┐            │                │
│         │           │MedicalImage  │            │                │
│         │           │   Loader     │            │                │
│         │           └──────────────┘            │                │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              HuggingFace Transformers                    │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │    │
│  │  │  Processor  │  │  Backbone   │  │ Classification  │  │    │
│  │  │(CuriaImage  │  │ (DINOv2     │  │    Heads        │  │    │
│  │  │ Processor)  │  │  ViT-B)     │  │ (atlas-stroke,  │  │    │
│  │  │             │  │             │  │  anatomy-mri)   │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. `curia_tester.py` - Core Testing API

**Classes:**
- `CuriaConfig` - Configuration dataclass (model name, device, heads)
- `MedicalImageLoader` - Loads DICOM, NIfTI, NumPy formats
- `MaskPrompt` - Creates and transforms mask prompts
- `CuriaTester` - Main testing interface

**Key Methods:**
```python
tester = CuriaTester(config)
tester.load_model(load_classifier=True)

# Feature extraction
features = tester.extract_features(image, mask=None)

# Classification
result = tester.classify(image, mask=None)

# Volume processing
results = tester.process_volume(volume, mask=None, slice_axis=-1)

# Attention visualization
attn_map = tester.get_attention_map(image)
```

### 2. `curia_gui.py` - Web Interface

Gradio-based GUI with:
- Model configuration (head selection, device)
- Preprocessing controls (modality, CT windows, normalization)
- Slice navigation for 3D volumes
- Interactive point/box prompting
- Feature extraction and classification

### 3. `preprocessing.py` - Image Preprocessing

**Classes:**
- `Normalizer` - CT windowing, MRI percentile normalization
- `SliceSelector` - Uniform, content-based, center selection
- `Resampler` - Image resizing
- `Preprocessor` - Combined pipeline

**CT Windows Available:**
| Window | Center | Width | Use Case |
|--------|--------|-------|----------|
| brain | 40 | 80 | Brain parenchyma |
| brain_stroke | 40 | 40 | Acute stroke |
| lung | -600 | 1500 | Lung parenchyma |
| soft_tissue | 40 | 400 | General soft tissue |
| bone | 400 | 2000 | Bone structures |

## Data Format Conventions

### Volume Orientation

**Standard format:** `(H, W, Z)` - Height, Width, Slices

- Slice dimension is **last** (axis=-1)
- Processor iterates `volume[:, :, i]` for each slice
- NIfTI files loaded in native (X, Y, Z) treated as (H, W, Z)

### Mask Handling

- **2D mask** `(H, W)`: Broadcast to all slices in volume
- **3D mask** `(H, W, Z)`: Per-slice masks, indexed along last axis
- For classification: 3D masks use middle slice

## Known Issues (Fixed)

The following issues have been identified and fixed:

| Issue | Location | Fix |
|-------|----------|-----|
| 3D slice axis mismatch | `curia_tester.py:207-225` | Keep NIfTI as (H,W,Z), don't transpose |
| 3D mask crash | `curia_tester.py:540-549` | Extract middle slice for 2D classifier |
| CLI 2D mask for 3D volume | `curia_tester.py:609-671` | Support 2D mask broadcast |
| Help recursion | `curia_tester.py:714-729` | Print help inline, don't recurse |
| GUI normalize checkbox | `curia_gui.py:58-68` | Store and use normalize state |

## Refactoring Recommendations

### Phase 1: Core Improvements

1. **Unified Volume Handler**
   ```python
   class VolumeHandler:
       """Consistent 3D volume handling across all components."""
       def __init__(self, volume, slice_axis=-1, metadata=None):
           self.data = volume
           self.slice_axis = slice_axis
           self.num_slices = volume.shape[slice_axis]

       def get_slice(self, idx):
           """Get slice by index, handling any axis."""
           ...

       def iterate_slices(self, indices=None):
           """Yield (index, slice) tuples."""
           ...
   ```

2. **Mask Manager**
   ```python
   class MaskManager:
       """Handle mask creation, loading, and transformation."""
       def __init__(self, volume_shape, slice_axis=-1):
           ...

       def add_point(self, point, radius=20):
           """Add point prompt, expand to volume if needed."""

       def add_box(self, box):
           """Add box prompt."""

       def get_slice_mask(self, idx):
           """Get mask for specific slice."""
   ```

3. **Configuration Validation**
   ```python
   @dataclass
   class CuriaConfig:
       ...
       def __post_init__(self):
           self.validate()

       def validate(self):
           if self.subfolder and self.subfolder not in self.AVAILABLE_HEADS:
               raise ValueError(f"Unknown head: {self.subfolder}")
   ```

### Phase 2: Testing Infrastructure

1. **Add Unit Tests**
   ```
   tests/
   ├── test_loader.py        # MedicalImageLoader tests
   ├── test_preprocessing.py # Normalization, slice selection
   ├── test_mask.py          # Mask prompting
   ├── test_volume.py        # 3D volume processing
   └── test_integration.py   # End-to-end tests
   ```

2. **Test Fixtures**
   ```python
   @pytest.fixture
   def sample_volume():
       return np.random.randn(224, 224, 100).astype(np.float32)

   @pytest.fixture
   def sample_mask_2d():
       mask = np.zeros((224, 224), dtype=np.float32)
       mask[50:150, 50:150] = 1.0
       return mask
   ```

3. **Integration Tests**
   ```python
   def test_3d_volume_with_2d_mask():
       """Ensure 2D masks work with 3D volumes."""
       ...

   def test_3d_volume_with_3d_mask():
       """Ensure 3D masks work correctly."""
       ...

   def test_gui_preprocessing_toggle():
       """Ensure preprocessing checkbox affects output."""
       ...
   ```

### Phase 3: Extension Points

1. **Plugin Architecture for Heads**
   ```python
   class ClassificationHead(ABC):
       @abstractmethod
       def predict(self, features: torch.Tensor) -> Dict:
           pass

       @property
       @abstractmethod
       def num_classes(self) -> int:
           pass

   class StrokeHead(ClassificationHead):
       ...

   class AnatomyHead(ClassificationHead):
       ...
   ```

2. **Custom Preprocessing Pipelines**
   ```python
   class PreprocessingPipeline:
       def __init__(self):
           self.steps = []

       def add_step(self, step: Callable):
           self.steps.append(step)

       def process(self, image):
           for step in self.steps:
               image = step(image)
           return image
   ```

3. **Results Export**
   ```python
   class ResultsExporter:
       def to_json(self, results, path):
           ...

       def to_nifti(self, attention_map, reference_nifti, path):
           """Save attention map as NIfTI overlay."""
           ...

       def to_dicom_sr(self, results, reference_dicom, path):
           """Save as DICOM Structured Report."""
           ...
   ```

### Phase 4: Production Readiness

1. **Logging**
   ```python
   import logging
   logger = logging.getLogger("curia")

   class CuriaTester:
       def load_model(self, ...):
           logger.info(f"Loading model from {self.config.model_name}")
           ...
   ```

2. **Error Handling**
   ```python
   class CuriaError(Exception):
       """Base exception for CURIA errors."""

   class ModelNotLoadedError(CuriaError):
       """Raised when model methods called before load."""

   class InvalidMaskError(CuriaError):
       """Raised when mask dimensions don't match image."""
   ```

3. **Caching**
   ```python
   from functools import lru_cache

   class CuriaTester:
       @lru_cache(maxsize=100)
       def _cached_preprocess(self, image_hash):
           ...
   ```

## Performance Benchmarks

| Device | Inference Time | Throughput |
|--------|----------------|------------|
| Apple M1 (MPS) | ~44ms | 22 img/sec |
| NVIDIA RTX 3090 | ~15ms | 66 img/sec |
| CPU (i9) | ~200ms | 5 img/sec |

## Usage Examples

### Basic Classification

```python
from curia_tester import CuriaTester, CuriaConfig

config = CuriaConfig(subfolder='atlas-stroke')
tester = CuriaTester(config)
tester.load_model(load_classifier=True)

import nibabel as nib
volume = nib.load('data/T1.nii.gz').get_fdata()

# Process single slice
mid_slice = volume[:, :, volume.shape[-1] // 2]
result = tester.classify(mid_slice)
print(f"Stroke probability: {result['probabilities'][0][1]:.2%}")
```

### Volume Processing with Mask

```python
from preprocessing import SliceSelector
import numpy as np

# Select informative slices
selector = SliceSelector(method='content', num_slices=10)
indices = selector.select(volume, axis=-1)

# Create region mask
mask = np.zeros(volume.shape[:2], dtype=np.float32)
mask[100:150, 100:150] = 1.0  # 2D mask broadcast to all slices

# Process volume
results = tester.process_volume(volume, mask=mask, slice_indices=indices)
print(f"Analyzed {len(indices)} slices")
```

## License

- **CURIA Model**: CC BY-NC-ND 4.0 (Research only, non-commercial)
- **This Framework**: MIT

## References

- [CURIA Paper](https://arxiv.org/abs/2509.06830)
- [HuggingFace Model](https://huggingface.co/raidium/curia)
- [GitHub Repository](https://github.com/raidium-med/curia)
