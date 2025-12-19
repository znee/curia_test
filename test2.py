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

    print("Loading processor...")
    processor = AutoImageProcessor.from_pretrained(
        "raidium/curia",
        trust_remote_code=True,
        use_fast=False # True  # optional, see below
    )

    print("Loading model...")
    model = AutoModel.from_pretrained(
        "raidium/curia",
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # Dummy single-slice image (H, W) in PL orientation for axial
    img = np.random.rand(256, 256).astype("float32")

    # Preprocess
    inputs = processor(img, return_tensors="pt")

    # Move tensors to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    if hasattr(outputs, "last_hidden_state"):
        print("last_hidden_state shape:", outputs.last_hidden_state.shape)
    else:
        print("Model outputs:", outputs)

    print("Done, Curia forward pass succeeded!")

if __name__ == "__main__":
    main()