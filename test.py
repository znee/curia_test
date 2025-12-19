import os
import numpy as np
import torch
from transformers import AutoModel, AutoImageProcessor

def main():
    # Choose device: MPS for Apple Silicon, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # Optional: read HF token from env (for gated model)
    hf_token = os.environ.get("HF_TOKEN")

    print("Loading processor...")
    processor = AutoImageProcessor.from_pretrained(
        "raidium/curia",
        trust_remote_code=True,
        token=hf_token,
    )

    print("Loading model...")
    model = AutoModel.from_pretrained(
        "raidium/curia",
        trust_remote_code=True,
        token=hf_token,
    ).to(device)
    model.eval()

    # Dummy single-slice image (H, W) in PL orientation for axial
    # Curia expects a 2D numpy array (H, W) with raw/normalized values.  [oai_citation:2‡Hugging Face](https://huggingface.co/raidium/curia?utm_source=chatgpt.com)
    img = np.random.rand(256, 256).astype("float32")

    # Preprocess
    inputs = processor(img, return_tensors="pt")

    # Move tensors to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Curia is a ViT-style vision backbone, so you'll typically see last_hidden_state
    if hasattr(outputs, "last_hidden_state"):
        print("last_hidden_state shape:", outputs.last_hidden_state.shape)
    else:
        print("Model outputs:", outputs)

    print("Done, Curia forward pass succeeded!")

if __name__ == "__main__":
    main()