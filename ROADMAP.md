# CURIA Advanced Features Roadmap

This document outlines the development plan for extending CURIA with advanced segmentation and localization capabilities.

## Executive Summary

Three major features are planned:
1. **Natural Language Segmentation Prompts** - Text-to-region mapping
2. **Anatomic Localization Module** - Body region detection with RadLex ontology
3. **3D Segmentation Propagation** - Volumetric mask refinement

These features require significant architectural changes and will be implemented as a **separate application** (`curia-advanced`) that builds on the core testing framework.

---

## Current Architecture Analysis

### Strengths
- Clean separation: loader, preprocessor, tester, GUI
- CURIA provides 768-dim features per patch (32×32 grid)
- Mask-based attention already implemented in `modeling_dinov2.py`
- `extract_mask_features()` and attention modules ready for extension

### Extension Points
```
┌─────────────────────────────────────────────────────────────────┐
│                     Current Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  MedicalImageLoader ──▶ Preprocessor ──▶ CuriaTester            │
│         │                    │                │                  │
│         │                    │                ▼                  │
│         │                    │         ┌─────────────┐          │
│         │                    │         │   CURIA     │          │
│         │                    │         │  Backbone   │          │
│         │                    │         └──────┬──────┘          │
│         │                    │                │                  │
│         │                    │                ▼                  │
│         │                    │    768-dim features (1025 tokens) │
│         │                    │                │                  │
│         ▼                    ▼                ▼                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   EXTENSION POINTS                        │   │
│  │  • Pre-processing hooks (anatomic detection)              │   │
│  │  • Feature extraction hooks (text conditioning)           │   │
│  │  • Post-processing hooks (3D propagation)                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Proposed Architecture: `curia-advanced`

```
curia-advanced/
├── core/                      # Refactored from current codebase
│   ├── __init__.py
│   ├── config.py             # Unified configuration
│   ├── loader.py             # MedicalImageLoader (refactored)
│   ├── volume.py             # VolumeHandler class
│   ├── preprocessing.py      # Preprocessing pipeline
│   └── features.py           # Feature extraction utilities
│
├── models/                    # Model wrappers
│   ├── __init__.py
│   ├── curia_backbone.py     # CURIA DINOv2 wrapper
│   ├── text_encoder.py       # Text encoder (CLIP/BiomedCLIP)
│   ├── anatomy_classifier.py # Anatomy detection head
│   └── propagation_net.py    # 3D propagation network
│
├── prompts/                   # Prompt handling
│   ├── __init__.py
│   ├── text_prompt.py        # Natural language parser
│   ├── mask_prompt.py        # Geometric prompts (point/box)
│   ├── anatomy_prompt.py     # RadLex-based prompts
│   └── prompt_resolver.py    # Unified prompt → mask resolver
│
├── segmentation/              # Segmentation modules
│   ├── __init__.py
│   ├── slice_segmenter.py    # Per-slice segmentation
│   ├── propagator.py         # Cross-slice propagation
│   ├── refiner.py            # 3D refinement network
│   └── consistency.py        # Temporal/spatial consistency
│
├── anatomy/                   # Anatomic localization
│   ├── __init__.py
│   ├── detector.py           # Body region detector
│   ├── radlex.py             # RadLex ontology interface
│   ├── coordinate_system.py  # Anatomic coordinate mapping
│   └── metadata.py           # DICOM/NIfTI metadata enrichment
│
├── services/                  # API layer
│   ├── __init__.py
│   ├── segmentation_service.py
│   ├── anatomy_service.py
│   └── pipeline_service.py   # Combined workflows
│
├── interfaces/                # User interfaces
│   ├── __init__.py
│   ├── cli.py                # Command-line interface
│   ├── gui.py                # Gradio advanced GUI
│   └── api.py                # REST API (FastAPI)
│
├── tests/                     # Test suite
│   ├── test_text_prompts.py
│   ├── test_anatomy.py
│   ├── test_propagation.py
│   └── test_integration.py
│
└── configs/                   # Configuration files
    ├── default.yaml
    ├── anatomy_heads.yaml
    └── radlex_mapping.yaml
```

---

## Feature 1: Natural Language Segmentation Prompts

### Overview
Enable clinicians to describe regions using natural language:
- "Segment the hypodense lesion in the right frontal lobe"
- "Show me the area of ground-glass opacity"
- "Highlight the enlarged lymph nodes"

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Natural Language Segmentation Pipeline              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "Segment the hypodense lesion in the right frontal lobe"       │
│                          │                                       │
│                          ▼                                       │
│  ┌────────────────────────────────────────────────────────┐     │
│  │                   Text Encoder                          │     │
│  │  • BiomedCLIP / PubMedBERT / Clinical BERT             │     │
│  │  • Output: 512/768-dim text embedding                   │     │
│  └────────────────────────────────────────────────────────┘     │
│                          │                                       │
│                          ▼                                       │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              Semantic Parser (NLU)                      │     │
│  │  • Entity extraction: lesion, right frontal lobe        │     │
│  │  • Attribute extraction: hypodense                      │     │
│  │  • Spatial relations: in, near, adjacent to             │     │
│  │  • Output: StructuredPrompt                             │     │
│  └────────────────────────────────────────────────────────┘     │
│                          │                                       │
│                          ▼                                       │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              Prompt-to-Region Resolver                  │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │     │
│  │  │   Anatomy    │  │   Feature    │  │    Mask      │  │     │
│  │  │   Lookup     │  │   Matching   │  │  Generation  │  │     │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │     │
│  └────────────────────────────────────────────────────────┘     │
│                          │                                       │
│                          ▼                                       │
│  ┌────────────────────────────────────────────────────────┐     │
│  │           Cross-Attention Mask Conditioning             │     │
│  │  • Text embedding conditions CURIA attention            │     │
│  │  • Soft attention → hard mask via thresholding          │     │
│  │  • Optional: SAM-Med2D for refinement                   │     │
│  └────────────────────────────────────────────────────────┘     │
│                          │                                       │
│                          ▼                                       │
│                  Segmentation Mask (H, W)                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Plan

#### Phase 1.1: Text Encoder Integration (Week 1-2)
```python
# prompts/text_prompt.py
from transformers import AutoTokenizer, AutoModel

class TextPromptEncoder:
    """Encode clinical text prompts to embeddings."""

    SUPPORTED_MODELS = [
        "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "emilyalsentzer/Bio_ClinicalBERT",
    ]

    def __init__(self, model_name: str = "BiomedCLIP"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to embedding vector."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token
```

#### Phase 1.2: Semantic Parser (Week 3-4)
```python
# prompts/text_prompt.py
@dataclass
class StructuredPrompt:
    """Parsed clinical prompt."""
    target_entity: str          # "lesion", "tumor", "opacity"
    attributes: List[str]       # ["hypodense", "enhancing", "calcified"]
    anatomy_region: str         # "right frontal lobe", "lower left lung"
    spatial_relation: str       # "in", "near", "surrounding"
    modality_hint: str          # "CT", "MRI", inferred from attributes

class ClinicalNLUParser:
    """Parse clinical text into structured prompts."""

    # Medical entity vocabulary
    LESION_TERMS = ["lesion", "tumor", "mass", "nodule", "opacity", ...]
    ANATOMY_TERMS = RadLexOntology.get_all_terms()
    ATTRIBUTE_TERMS = {
        "density": ["hypodense", "hyperdense", "isodense", ...],
        "enhancement": ["enhancing", "non-enhancing", ...],
        "shape": ["round", "irregular", "spiculated", ...],
    }

    def parse(self, text: str) -> StructuredPrompt:
        """Extract structured information from clinical text."""
        # Use spaCy with scispacy models for medical NER
        # Or fine-tuned BERT for medical entity extraction
        ...
```

#### Phase 1.3: Prompt-to-Region Resolver (Week 5-6)
```python
# prompts/prompt_resolver.py
class PromptToRegionResolver:
    """Convert structured prompts to image regions."""

    def __init__(
        self,
        anatomy_detector: AnatomyDetector,
        feature_extractor: CuriaBackbone,
        text_encoder: TextPromptEncoder,
    ):
        self.anatomy = anatomy_detector
        self.features = feature_extractor
        self.text = text_encoder

    def resolve(
        self,
        image: np.ndarray,
        prompt: StructuredPrompt
    ) -> np.ndarray:
        """
        Convert prompt to segmentation mask.

        Returns:
            Binary mask (H, W) or soft attention map
        """
        # Step 1: Get anatomy region mask (coarse)
        anatomy_mask = self.anatomy.get_region_mask(
            image, prompt.anatomy_region
        )

        # Step 2: Extract CURIA features within region
        features = self.features.extract_masked(image, anatomy_mask)

        # Step 3: Match features to text embedding
        text_emb = self.text.encode(prompt.to_description())
        similarity = self._compute_similarity(features, text_emb)

        # Step 4: Generate attention mask
        attention_mask = self._features_to_mask(similarity, image.shape)

        # Step 5: Refine with thresholding or SAM
        final_mask = self._refine_mask(attention_mask, image)

        return final_mask
```

#### Phase 1.4: Training Data Collection (Week 7-8)
```yaml
# configs/text_prompt_dataset.yaml
datasets:
  - name: "radiology_reports_paired"
    description: "Reports with annotated regions"
    format:
      - text: "3cm hypodense lesion in segment 7 of liver"
        mask: "path/to/mask.nii.gz"
        image: "path/to/ct.nii.gz"

  - name: "synthetic_prompts"
    description: "Generated from existing segmentation datasets"
    source_datasets:
      - "TotalSegmentator"
      - "AMOS"
      - "BTCV"
```

### Validation Metrics
- **Text-to-Region Accuracy**: IoU between generated and ground truth masks
- **Entity Extraction F1**: Precision/recall of parsed entities
- **Clinical Relevance Score**: Radiologist evaluation of prompt understanding

---

## Feature 2: Anatomic Localization Module

### Overview
Automatic body region detection with standardized anatomical ontology:
- Classify slices/volumes by body region (brain, chest, abdomen, etc.)
- Map to RadLex anatomical coordinates
- Enable anatomy-aware preprocessing and analysis

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                Anatomic Localization Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  3D Volume (H, W, Z)                                            │
│         │                                                        │
│         ▼                                                        │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              Slice-Level Anatomy Classifier             │     │
│  │  • CURIA features → anatomy-ct/anatomy-mri heads        │     │
│  │  • Per-slice: {class_id, confidence, features}          │     │
│  └────────────────────────────────────────────────────────┘     │
│         │                                                        │
│         ▼                                                        │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              Volume-Level Aggregator                    │     │
│  │  • Sequence modeling (transformer/LSTM)                 │     │
│  │  • Body region voting across slices                     │     │
│  │  • Anatomical consistency enforcement                   │     │
│  └────────────────────────────────────────────────────────┘     │
│         │                                                        │
│         ▼                                                        │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              RadLex Ontology Mapper                     │     │
│  │  • Map predictions to RadLex RIDs                       │     │
│  │  • Hierarchical anatomy tree navigation                 │     │
│  │  • Coordinate system assignment                         │     │
│  └────────────────────────────────────────────────────────┘     │
│         │                                                        │
│         ▼                                                        │
│  ┌────────────────────────────────────────────────────────┐     │
│  │                  Metadata Service                       │     │
│  │  • Enrich DICOM/NIfTI with anatomy tags                │     │
│  │  • Expose via API for downstream tools                  │     │
│  │  • Cache and update anatomical predictions              │     │
│  └────────────────────────────────────────────────────────┘     │
│         │                                                        │
│         ▼                                                        │
│  AnatomyResult:                                                  │
│    body_region: "chest"                                          │
│    radlex_id: "RID1243"                                         │
│    structures: [{name: "lung", confidence: 0.95}, ...]          │
│    coordinate_system: "thoracic"                                 │
│    slice_annotations: [{slice: 50, region: "mediastinum"}, ...]  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Plan

#### Phase 2.1: RadLex Integration (Week 1-2)
```python
# anatomy/radlex.py
from typing import Dict, List, Optional
import json

class RadLexOntology:
    """Interface to RadLex anatomical ontology."""

    # RadLex hierarchy (simplified)
    BODY_REGIONS = {
        "RID2638": {"name": "head", "children": ["RID6434", "RID6469", ...]},
        "RID1243": {"name": "thorax", "children": ["RID1302", "RID1384", ...]},
        "RID56": {"name": "abdomen", "children": ["RID170", "RID187", ...]},
        # ... full ontology loaded from radlex.json
    }

    # Mapping from CURIA anatomy classes to RadLex
    CURIA_TO_RADLEX = {
        # anatomy-ct classes
        0: "RID2638",   # head
        1: "RID1243",   # thorax
        # ... 54 classes
    }

    def __init__(self, ontology_path: str = "configs/radlex.json"):
        self.ontology = self._load_ontology(ontology_path)

    def get_region(self, radlex_id: str) -> Dict:
        """Get region info by RadLex ID."""
        return self.ontology.get(radlex_id, {})

    def get_parent(self, radlex_id: str) -> Optional[str]:
        """Get parent region in hierarchy."""
        ...

    def get_children(self, radlex_id: str) -> List[str]:
        """Get child regions."""
        ...

    def find_by_name(self, name: str) -> List[str]:
        """Search for regions by name."""
        ...
```

#### Phase 2.2: Anatomy Detector (Week 3-4)
```python
# anatomy/detector.py
class AnatomyDetector:
    """Detect anatomical regions in medical images."""

    def __init__(
        self,
        curia_tester: CuriaTester,
        radlex: RadLexOntology,
        modality: str = "auto"
    ):
        self.tester = curia_tester
        self.radlex = radlex
        self.modality = modality

        # Load appropriate head
        head = "anatomy-ct" if modality == "ct" else "anatomy-mri"
        self.tester.config.subfolder = head
        self.tester.load_model(load_classifier=True)

    def detect_slice(self, image: np.ndarray) -> SliceAnatomyResult:
        """Detect anatomy in single slice."""
        result = self.tester.classify(image)

        # Map to RadLex
        class_id = result["prediction"]
        radlex_id = self.radlex.CURIA_TO_RADLEX.get(class_id)

        return SliceAnatomyResult(
            class_id=class_id,
            confidence=result["probabilities"].max(),
            radlex_id=radlex_id,
            radlex_name=self.radlex.get_region(radlex_id)["name"],
            features=result.get("features"),
        )

    def detect_volume(
        self,
        volume: np.ndarray,
        aggregate: str = "voting"
    ) -> VolumeAnatomyResult:
        """Detect anatomy across entire volume."""
        slice_results = []

        for idx in range(volume.shape[-1]):
            slice_img = volume[:, :, idx]
            result = self.detect_slice(slice_img)
            slice_results.append(result)

        # Aggregate
        if aggregate == "voting":
            body_region = self._majority_vote(slice_results)
        elif aggregate == "sequence":
            body_region = self._sequence_model(slice_results)

        return VolumeAnatomyResult(
            body_region=body_region,
            slice_annotations=slice_results,
            radlex_hierarchy=self.radlex.get_hierarchy(body_region.radlex_id),
        )
```

#### Phase 2.3: Anatomy Service API (Week 5-6)
```python
# services/anatomy_service.py
class AnatomyService:
    """Service layer for anatomy detection."""

    def __init__(self, detector: AnatomyDetector):
        self.detector = detector
        self.cache = {}  # Optional caching

    def get_body_region(
        self,
        volume: np.ndarray,
        confidence_threshold: float = 0.8
    ) -> str:
        """Get primary body region for volume."""
        result = self.detector.detect_volume(volume)

        if result.confidence < confidence_threshold:
            return "unknown"

        return result.body_region.radlex_name

    def get_region_mask(
        self,
        volume: np.ndarray,
        region_name: str
    ) -> np.ndarray:
        """Get binary mask for specific anatomical region."""
        # Find slices where region is detected
        result = self.detector.detect_volume(volume)

        mask = np.zeros(volume.shape, dtype=np.float32)
        for idx, slice_result in enumerate(result.slice_annotations):
            if slice_result.radlex_name == region_name:
                mask[:, :, idx] = 1.0

        return mask

    def enrich_metadata(
        self,
        volume: np.ndarray,
        metadata: Dict
    ) -> Dict:
        """Add anatomy info to existing metadata."""
        result = self.detector.detect_volume(volume)

        metadata["anatomy"] = {
            "body_region": result.body_region.radlex_name,
            "radlex_id": result.body_region.radlex_id,
            "confidence": result.confidence,
            "modality": self.detector.modality,
        }

        return metadata
```

### Validation
- Use TotalSegmentator labels for anatomy classification accuracy
- Cross-reference with DICOM BodyPartExamined tag
- Evaluate on multi-center datasets for generalization

---

## Feature 3: 3D Segmentation Propagation

### Overview
Propagate 2D segmentation masks across slices to create volumetric segmentations:
- Interpolation based on intensity similarity
- Transformer attention for feature-guided propagation
- Optional 3D UNet refinement
- Temporal consistency for cine/dynamic sequences

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│               3D Segmentation Propagation Pipeline               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Initial Mask (slice k)                                          │
│         │                                                        │
│         ▼                                                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  Propagation Engine                         │ │
│  │                                                             │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │ │
│  │  │  Intensity  │    │  Feature    │    │  Attention  │    │ │
│  │  │  Similarity │    │  Matching   │    │  Transfer   │    │ │
│  │  │ (optical    │    │ (CURIA      │    │ (cross-slice│    │ │
│  │  │  flow-like) │    │  features)  │    │  attention) │    │ │
│  │  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    │ │
│  │         │                  │                  │            │ │
│  │         └──────────────────┼──────────────────┘            │ │
│  │                            ▼                               │ │
│  │                   Propagation Weights                      │ │
│  │                            │                               │ │
│  └────────────────────────────┼───────────────────────────────┘ │
│                               ▼                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    3D Refinement                            │ │
│  │  • Lightweight 3D UNet (optional)                          │ │
│  │  • CRF post-processing                                     │ │
│  │  • Morphological operations                                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                               │                                  │
│                               ▼                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                 Consistency Checker                         │ │
│  │  • Volumetric connectivity                                 │ │
│  │  • Size/shape constraints                                  │ │
│  │  • Temporal smoothness (for dynamic)                       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                               │                                  │
│                               ▼                                  │
│  3D Mask (H, W, Z) + Provenance Metadata                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Plan

#### Phase 3.1: Basic Propagation (Week 1-2)
```python
# segmentation/propagator.py
class MaskPropagator:
    """Propagate masks across slices."""

    def __init__(
        self,
        method: str = "feature_matching",
        feature_extractor: Optional[CuriaBackbone] = None,
    ):
        self.method = method
        self.features = feature_extractor

    def propagate(
        self,
        volume: np.ndarray,
        initial_mask: np.ndarray,
        seed_slice: int,
        direction: str = "both"  # "forward", "backward", "both"
    ) -> np.ndarray:
        """
        Propagate mask from seed slice to entire volume.

        Args:
            volume: 3D image (H, W, Z)
            initial_mask: 2D mask (H, W) for seed slice
            seed_slice: Index of slice with initial mask
            direction: Propagation direction

        Returns:
            3D mask (H, W, Z)
        """
        num_slices = volume.shape[-1]
        result_mask = np.zeros(volume.shape, dtype=np.float32)
        result_mask[:, :, seed_slice] = initial_mask

        if direction in ["forward", "both"]:
            self._propagate_direction(
                volume, result_mask, seed_slice,
                range(seed_slice + 1, num_slices)
            )

        if direction in ["backward", "both"]:
            self._propagate_direction(
                volume, result_mask, seed_slice,
                range(seed_slice - 1, -1, -1)
            )

        return result_mask

    def _propagate_direction(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
        start: int,
        slice_range: range
    ):
        """Propagate in one direction."""
        prev_idx = start

        for idx in slice_range:
            prev_slice = volume[:, :, prev_idx]
            curr_slice = volume[:, :, idx]
            prev_mask = mask[:, :, prev_idx]

            # Compute propagation
            if self.method == "intensity":
                curr_mask = self._intensity_propagate(
                    prev_slice, curr_slice, prev_mask
                )
            elif self.method == "feature_matching":
                curr_mask = self._feature_propagate(
                    prev_slice, curr_slice, prev_mask
                )
            elif self.method == "attention":
                curr_mask = self._attention_propagate(
                    prev_slice, curr_slice, prev_mask
                )

            mask[:, :, idx] = curr_mask
            prev_idx = idx
```

#### Phase 3.2: Feature-Guided Propagation (Week 3-4)
```python
# segmentation/propagator.py (continued)
    def _feature_propagate(
        self,
        prev_slice: np.ndarray,
        curr_slice: np.ndarray,
        prev_mask: np.ndarray
    ) -> np.ndarray:
        """Propagate using CURIA feature matching."""
        # Extract features
        prev_features = self.features.extract(prev_slice)  # (1, 1024, 768)
        curr_features = self.features.extract(curr_slice)  # (1, 1024, 768)

        # Reshape to spatial grid
        grid_size = 32  # sqrt(1024)
        prev_grid = prev_features.view(grid_size, grid_size, -1)
        curr_grid = curr_features.view(grid_size, grid_size, -1)

        # Downsample mask to feature resolution
        mask_small = cv2.resize(
            prev_mask, (grid_size, grid_size),
            interpolation=cv2.INTER_NEAREST
        )

        # Find matching features in current slice
        mask_features = prev_grid[mask_small > 0.5]  # (N, 768)

        # Compute similarity with all positions
        similarity = torch.cosine_similarity(
            mask_features.unsqueeze(1),  # (N, 1, 768)
            curr_grid.view(-1, 768).unsqueeze(0),  # (1, 1024, 768)
            dim=-1
        )  # (N, 1024)

        # Aggregate and reshape
        max_sim = similarity.max(dim=0).values  # (1024,)
        sim_map = max_sim.view(grid_size, grid_size)

        # Upsample to original resolution
        curr_mask = cv2.resize(
            sim_map.numpy(),
            (curr_slice.shape[1], curr_slice.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

        # Threshold
        threshold = 0.5
        return (curr_mask > threshold).astype(np.float32)
```

#### Phase 3.3: 3D Refinement Network (Week 5-6)
```python
# models/propagation_net.py
class LightweightUNet3D(nn.Module):
    """Lightweight 3D UNet for mask refinement."""

    def __init__(
        self,
        in_channels: int = 2,  # image + propagated mask
        out_channels: int = 1,
        features: List[int] = [16, 32, 64]
    ):
        super().__init__()
        self.encoder = self._build_encoder(in_channels, features)
        self.decoder = self._build_decoder(features)
        self.final = nn.Conv3d(features[0], out_channels, 1)

    def forward(
        self,
        image: torch.Tensor,      # (B, 1, D, H, W)
        coarse_mask: torch.Tensor  # (B, 1, D, H, W)
    ) -> torch.Tensor:
        """Refine coarse mask using image context."""
        x = torch.cat([image, coarse_mask], dim=1)  # (B, 2, D, H, W)

        # Encoder
        enc_features = []
        for enc in self.encoder:
            x = enc(x)
            enc_features.append(x)

        # Decoder with skip connections
        for i, dec in enumerate(self.decoder):
            x = dec(x)
            if i < len(enc_features) - 1:
                x = x + enc_features[-(i+2)]

        return torch.sigmoid(self.final(x))


# segmentation/refiner.py
class MaskRefiner:
    """Refine propagated masks using 3D context."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_crf: bool = True
    ):
        self.model = LightweightUNet3D()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.use_crf = use_crf

    def refine(
        self,
        volume: np.ndarray,
        coarse_mask: np.ndarray
    ) -> np.ndarray:
        """Refine coarse 3D mask."""
        # Convert to tensors
        vol_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
        mask_tensor = torch.from_numpy(coarse_mask).unsqueeze(0).unsqueeze(0)

        # Run refinement
        with torch.no_grad():
            refined = self.model(vol_tensor, mask_tensor)

        refined_np = refined.squeeze().numpy()

        # Optional CRF post-processing
        if self.use_crf:
            refined_np = self._apply_crf(volume, refined_np)

        return refined_np
```

#### Phase 3.4: Consistency and Provenance (Week 7-8)
```python
# segmentation/consistency.py
@dataclass
class PropagationProvenance:
    """Track how each slice's mask was generated."""
    slice_idx: int
    source: str  # "manual", "propagated", "refined"
    source_slice: Optional[int]  # If propagated, which slice it came from
    confidence: float
    edits: List[Dict]  # History of manual corrections

class ConsistencyChecker:
    """Ensure volumetric segmentation consistency."""

    def check(
        self,
        mask: np.ndarray,
        constraints: Optional[Dict] = None
    ) -> ConsistencyReport:
        """Check mask for consistency issues."""
        issues = []

        # Check connectivity
        labeled, num_components = label(mask)
        if num_components > 1:
            issues.append(ConsistencyIssue(
                type="disconnected",
                slices=self._find_disconnected_slices(labeled)
            ))

        # Check size consistency
        slice_areas = [mask[:, :, i].sum() for i in range(mask.shape[-1])]
        if self._has_sudden_size_change(slice_areas):
            issues.append(ConsistencyIssue(
                type="size_discontinuity",
                slices=self._find_size_jumps(slice_areas)
            ))

        return ConsistencyReport(issues=issues, is_valid=len(issues) == 0)

    def enforce(
        self,
        mask: np.ndarray,
        constraints: Dict
    ) -> np.ndarray:
        """Enforce consistency constraints."""
        # Keep largest connected component
        if constraints.get("single_component", True):
            mask = self._keep_largest_component(mask)

        # Smooth along z-axis
        if constraints.get("smooth_z", True):
            mask = self._smooth_along_z(mask)

        return mask
```

---

## Refactoring Current Codebase

### Changes Required to Existing Files

#### 1. `curia_tester.py` → `core/features.py`
```python
# Extract feature-related code to dedicated module
class CuriaFeatureExtractor:
    """Pure feature extraction without classification logic."""

    def __init__(self, model_name: str, device: str):
        self.backbone = self._load_backbone(model_name, device)
        self.processor = self._load_processor(model_name)

    def extract(
        self,
        image: np.ndarray,
        return_spatial: bool = False
    ) -> FeatureOutput:
        """Extract features from image."""
        inputs = self.processor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.backbone(**inputs)

        if return_spatial:
            # Return (32, 32, 768) spatial features
            return self._to_spatial(outputs.last_hidden_state[:, 1:])
        else:
            # Return (768,) CLS token
            return outputs.last_hidden_state[:, 0]
```

#### 2. `preprocessing.py` → `core/preprocessing.py`
- Add anatomy-aware preprocessing hooks
- Support for preprocessing pipelines

#### 3. `curia_gui.py` → `interfaces/gui.py`
- Split into base GUI and advanced GUI
- Add text prompt input
- Add 3D mask visualization

### New Abstract Interfaces

```python
# core/interfaces.py
from abc import ABC, abstractmethod

class PromptInterface(ABC):
    """Base interface for all prompt types."""

    @abstractmethod
    def to_mask(self, image_shape: Tuple[int, ...]) -> np.ndarray:
        """Convert prompt to binary mask."""
        pass

    @abstractmethod
    def to_description(self) -> str:
        """Convert prompt to text description."""
        pass

class SegmentationInterface(ABC):
    """Base interface for segmentation methods."""

    @abstractmethod
    def segment(
        self,
        image: np.ndarray,
        prompt: PromptInterface
    ) -> np.ndarray:
        """Generate segmentation mask."""
        pass

class LocalizationInterface(ABC):
    """Base interface for anatomic localization."""

    @abstractmethod
    def localize(
        self,
        image: np.ndarray
    ) -> LocalizationResult:
        """Detect anatomical regions."""
        pass
```

---

## Implementation Timeline

```
                    CURIA Advanced Features Timeline
═══════════════════════════════════════════════════════════════════

Month 1-2: Foundation & NL Prompts
├── Week 1-2: Refactor core codebase
├── Week 3-4: Text encoder integration
├── Week 5-6: Semantic parser (NLU)
├── Week 7-8: Prompt-to-region resolver

Month 3-4: Anatomic Localization
├── Week 1-2: RadLex ontology integration
├── Week 3-4: Anatomy detector implementation
├── Week 5-6: Service API & metadata enrichment
├── Week 7-8: Validation & testing

Month 5-6: 3D Propagation
├── Week 1-2: Basic propagation methods
├── Week 3-4: Feature-guided propagation
├── Week 5-6: 3D refinement network
├── Week 7-8: Consistency & provenance

Month 7: Integration & Testing
├── Week 1-2: Integration testing
├── Week 3-4: Performance optimization
├── Week 5-6: Documentation & examples

Month 8: Deployment
├── Week 1-2: API deployment
├── Week 3-4: GUI deployment & user testing
```

---

## Dependencies

### New Required Packages
```txt
# NLP / Text Processing
transformers>=4.45
spacy>=3.7
scispacy>=0.5
en_core_sci_lg  # spaCy medical model

# Biomedical Models
open_clip_torch  # For BiomedCLIP

# 3D Processing
connected-components-3d>=3.12
pydensecrf  # CRF post-processing

# API
fastapi>=0.104
uvicorn>=0.24

# Optional: External Segmentation
segment-anything  # SAM for refinement
```

### Hardware Requirements
- **Development**: 16GB RAM, GPU with 8GB VRAM
- **Production**: 32GB RAM, GPU with 16GB+ VRAM (for 3D refinement)

---

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Text encoder domain mismatch | High | Medium | Fine-tune on radiology reports |
| RadLex mapping incomplete | Medium | Low | Manual curation, fallback to CURIA classes |
| Propagation quality degrades | High | Medium | Multi-method fusion, user correction UI |
| 3D refinement too slow | Medium | Medium | Lightweight architecture, optional toggle |
| Integration complexity | High | Medium | Modular design, extensive testing |

---

## Success Metrics

### Feature 1: NL Prompts
- Text-to-Region IoU > 0.6 on test set
- Entity extraction F1 > 0.85
- User satisfaction score > 4/5

### Feature 2: Anatomy Localization
- Body region accuracy > 95%
- RadLex mapping coverage > 90%
- Inference time < 1s per volume

### Feature 3: 3D Propagation
- Propagation Dice score > 0.8 vs manual
- Refinement improvement > 5% Dice
- Consistency check pass rate > 95%

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Prioritize features** based on immediate needs
3. **Create development branch** for `curia-advanced`
4. **Set up CI/CD** for testing
5. **Begin Phase 1** implementation
