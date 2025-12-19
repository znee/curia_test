# CURIA Advanced - Implementation Plan

## Part 1: Natural Language Segmentation Prompts

### Objective
Enable clinicians to describe regions of interest using natural language, which the system will convert to segmentation masks.

**Example inputs:**
- "Segment the hypodense lesion in the right frontal lobe"
- "Show me the ground-glass opacity in the lower left lung"
- "Highlight the enlarged lymph nodes in the mediastinum"

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NL Segmentation Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   User Input: "Segment the hypodense lesion in right frontal lobe"  │
│                              │                                       │
│                              ▼                                       │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  1. TextPromptEncoder                                         │  │
│   │     - Encode text to 768-dim embedding                        │  │
│   │     - Models: BiomedCLIP, PubMedBERT, ClinicalBERT           │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  2. ClinicalNLUParser                                         │  │
│   │     - Extract: target="lesion", attr="hypodense"              │  │
│   │     - Extract: anatomy="right frontal lobe"                   │  │
│   │     - Output: StructuredPrompt dataclass                      │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  3. PromptToRegionResolver                                    │  │
│   │     - Use anatomy detector for coarse localization            │  │
│   │     - Match CURIA features to text embedding                  │  │
│   │     - Generate attention-based mask                           │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              ▼                                       │
│                      Segmentation Mask                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. TextPromptEncoder

**Purpose:** Convert clinical text to dense vector embeddings

**Input:** `str` - Clinical description text
**Output:** `torch.Tensor` - Shape (768,) or (512,) depending on model

**Supported Models:**
| Model | Embedding Dim | Domain | Notes |
|-------|---------------|--------|-------|
| BiomedCLIP | 512 | Biomedical | Best for image-text matching |
| PubMedBERT | 768 | PubMed | Good for clinical text |
| ClinicalBERT | 768 | Clinical notes | Best for EHR text |
| BioBERT | 768 | Biomedical | General biomedical |

**Key Methods:**
```python
class TextPromptEncoder:
    def encode(self, text: str) -> torch.Tensor
    def encode_batch(self, texts: List[str]) -> torch.Tensor
    def similarity(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> float
```

### 2. ClinicalNLUParser

**Purpose:** Parse natural language into structured components

**Input:** `str` - Clinical description
**Output:** `StructuredPrompt` dataclass

**Extracted Components:**
| Component | Description | Examples |
|-----------|-------------|----------|
| `target_entity` | What to segment | lesion, tumor, nodule, opacity |
| `attributes` | Descriptive features | hypodense, enhancing, calcified |
| `anatomy_region` | Location in body | right frontal lobe, lower left lung |
| `spatial_relation` | Positional modifier | in, near, adjacent to, surrounding |
| `modality_hint` | Imaging modality | CT, MRI (inferred from attributes) |

**Medical Vocabulary:**
```python
LESION_TERMS = [
    "lesion", "tumor", "mass", "nodule", "opacity", "abnormality",
    "finding", "enhancement", "collection", "hemorrhage", "infarct"
]

CT_ATTRIBUTES = [
    "hypodense", "hyperdense", "isodense", "calcified", "cystic",
    "solid", "enhancing", "non-enhancing", "rim-enhancing"
]

MRI_ATTRIBUTES = [
    "hyperintense", "hypointense", "isointense", "diffusion-restricting",
    "T1-bright", "T2-bright", "FLAIR-hyperintense"
]
```

### 3. PromptToRegionResolver

**Purpose:** Convert structured prompt to image mask

**Input:**
- `image: np.ndarray` - Medical image (H, W) or (H, W, Z)
- `prompt: StructuredPrompt` - Parsed prompt

**Output:** `np.ndarray` - Binary mask same shape as image

**Resolution Strategy:**
1. **Anatomy Localization** (coarse): Use anatomy detector to find region
2. **Feature Extraction**: Extract CURIA features within anatomy region
3. **Text-Feature Matching**: Compute similarity between text embedding and image features
4. **Mask Generation**: Threshold similarity map to create mask
5. **Refinement**: Optional post-processing (morphological ops, CRF)

---

## Directory Structure

```
curia-advanced/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── volume.py              # Volume handling utilities
│   └── interfaces.py          # Abstract base classes
│
├── prompts/
│   ├── __init__.py
│   ├── text_encoder.py        # TextPromptEncoder
│   ├── nlu_parser.py          # ClinicalNLUParser
│   ├── structured_prompt.py   # StructuredPrompt dataclass
│   ├── prompt_resolver.py     # PromptToRegionResolver
│   └── vocabulary.py          # Medical vocabulary definitions
│
├── models/
│   ├── __init__.py
│   └── curia_wrapper.py       # CURIA feature extraction wrapper
│
└── tests/
    ├── __init__.py
    ├── test_text_encoder.py
    ├── test_nlu_parser.py
    └── test_prompt_resolver.py
```

---

## Implementation Steps

### Step 1: Core Setup
- [x] Create directory structure
- [ ] Define configuration classes
- [ ] Create abstract interfaces

### Step 2: TextPromptEncoder
- [ ] Implement base encoder class
- [ ] Add BiomedCLIP support
- [ ] Add PubMedBERT support
- [ ] Add embedding caching

### Step 3: ClinicalNLUParser
- [ ] Define medical vocabularies
- [ ] Implement entity extraction
- [ ] Implement attribute extraction
- [ ] Implement anatomy parsing
- [ ] Create StructuredPrompt dataclass

### Step 4: PromptToRegionResolver
- [ ] Implement anatomy-based coarse localization
- [ ] Implement feature-text similarity matching
- [ ] Implement mask generation
- [ ] Add refinement options

### Step 5: Integration & Testing
- [ ] Integration with existing curia_tester
- [ ] Unit tests for each component
- [ ] Integration tests with real images
- [ ] Performance benchmarking

---

## Dependencies

```txt
# Required for Part 1
transformers>=4.45.0
torch>=2.0.0
numpy>=1.24.0

# Text Encoding
open-clip-torch>=2.24.0      # For BiomedCLIP

# NLU (optional - can use regex fallback)
spacy>=3.7.0
# scispacy model: en_core_sci_lg

# Testing
pytest>=7.0.0
pytest-cov>=4.0.0
```

---

## API Examples

### Basic Usage
```python
from curia_advanced.prompts import (
    TextPromptEncoder,
    ClinicalNLUParser,
    PromptToRegionResolver
)

# Initialize components
encoder = TextPromptEncoder(model="biomedclip")
parser = ClinicalNLUParser()
resolver = PromptToRegionResolver(encoder, parser)

# Process natural language prompt
text = "Segment the hypodense lesion in the right frontal lobe"
mask = resolver.resolve(image, text)

# Or step by step
prompt = parser.parse(text)
# StructuredPrompt(
#     target_entity="lesion",
#     attributes=["hypodense"],
#     anatomy_region="right frontal lobe",
#     spatial_relation="in",
#     modality_hint="CT"
# )

text_embedding = encoder.encode(text)
mask = resolver.resolve_with_prompt(image, prompt, text_embedding)
```

### Integration with Existing Tester
```python
from curia_tester import CuriaTester, CuriaConfig
from curia_advanced.prompts import NLPromptProcessor

# Load CURIA
config = CuriaConfig(subfolder="atlas-stroke")
tester = CuriaTester(config)
tester.load_model(load_classifier=True)

# Add NL prompt processing
nl_processor = NLPromptProcessor(tester)

# Use natural language
result = nl_processor.classify_with_text_prompt(
    image,
    "Focus on the area of restricted diffusion in the left MCA territory"
)
```

---

## Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Entity Extraction F1 | > 0.85 | Test on annotated prompts |
| Anatomy Parsing Accuracy | > 0.90 | Test against RadLex terms |
| Text-to-Region IoU | > 0.50 | Compare to manual masks |
| Inference Time | < 2s | End-to-end on single slice |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| BiomedCLIP not available | High | Fallback to PubMedBERT |
| NLU parsing errors | Medium | Regex fallback, user correction UI |
| Feature mismatch with text | High | Fine-tune projection layer |
| Slow inference | Medium | Caching, batch processing |

---

## Next Steps After Part 1

1. **Part 2: Anatomic Localization**
   - Build on NLU parser's anatomy extraction
   - Create RadLex ontology integration

2. **Part 3: 3D Propagation**
   - Use text-guided masks as seeds
   - Propagate across volume slices
