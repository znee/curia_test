"""
CURIA GUI - Gradio-based interface for medical image testing
=============================================================
A web-based interface for testing CURIA vision foundation model.

Usage:
    python curia_gui.py

Then open http://localhost:7860 in your browser.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    print("Gradio not installed. Install with: pip install gradio")

from curia_tester import CuriaTester, CuriaConfig, MedicalImageLoader, MaskPrompt
from preprocessing import Preprocessor, Normalizer, SliceSelector, CT_WINDOWS


class CuriaGUI:
    """Gradio-based GUI for CURIA testing."""

    def __init__(self):
        self.tester = None
        self.current_image = None
        self.current_volume = None  # Store full volume
        self.current_mask = None
        self.current_metadata = None
        self.config = CuriaConfig()
        self.preprocessor = Preprocessor()
        self.current_slice_idx = 0
        self.apply_normalize = True  # Track normalization state

    def load_model(self, head_name: str, device: str, hf_token: str) -> str:
        """Load CURIA model with specified configuration."""
        try:
            self.config.device = device
            self.config.hf_token = hf_token if hf_token else None
            self.config.subfolder = head_name if head_name != "None" else None

            self.tester = CuriaTester(self.config)
            self.tester.load_model(load_classifier=(head_name != "None"))

            return f"Model loaded successfully on {self.tester.device}"
        except Exception as e:
            return f"Error loading model: {str(e)}"

    def update_preprocessing(self, modality: str, ct_window: str, normalize: bool) -> str:
        """Update preprocessing settings."""
        try:
            self.apply_normalize = normalize  # Store normalization state
            self.preprocessor = Preprocessor(
                modality=modality.lower(),
                ct_window=ct_window.lower().replace(" ", "_"),
                target_size=(512, 512)
            )
            return f"Preprocessing: {modality}, window={ct_window}, normalize={normalize}"
        except Exception as e:
            return f"Error updating preprocessing: {str(e)}"

    def load_image(self, file) -> Tuple[Optional[np.ndarray], str, int]:
        """Load medical image from uploaded file."""
        if file is None:
            return None, "No file uploaded", 0

        try:
            loader = MedicalImageLoader()
            self.current_image, self.current_metadata = loader.load(file.name)

            # Store volume if 3D - format is (H, W, Z), slice along last axis
            if self.current_image.ndim == 3:
                self.current_volume = self.current_image.copy()
                num_slices = self.current_image.shape[-1]  # Last axis is slices
                self.current_slice_idx = num_slices // 2
                display_img = self.current_image[:, :, self.current_slice_idx]
            else:
                self.current_volume = None
                display_img = self.current_image.copy()
                num_slices = 1

            # Normalize for display
            display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min() + 1e-8)
            display_img = (display_img * 255).astype(np.uint8)

            info = f"Shape: {self.current_image.shape}\n"
            info += f"Range: [{self.current_image.min():.2f}, {self.current_image.max():.2f}]\n"
            if self.current_metadata:
                for k, v in list(self.current_metadata.items())[:5]:
                    info += f"{k}: {v}\n"

            # Auto-detect modality
            detected = self.preprocessor.normalizer.detect_modality(self.current_image)
            info += f"Detected modality: {detected.value.upper()}\n"

            return display_img, info, num_slices

        except Exception as e:
            return None, f"Error loading image: {str(e)}", 0

    def change_slice(self, slice_idx: int, apply_preprocessing: bool, ct_window: str) -> Tuple[Optional[np.ndarray], str]:
        """Change displayed slice for 3D volumes."""
        if self.current_volume is None:
            return None, "No 3D volume loaded"

        try:
            slice_idx = int(slice_idx)
            # Handle (H, W, Z) format - slice along last axis
            num_slices = self.current_volume.shape[-1]
            self.current_slice_idx = max(0, min(slice_idx, num_slices - 1))
            display_img = self.current_volume[:, :, self.current_slice_idx].copy()

            # Apply preprocessing if requested
            if apply_preprocessing:
                window_key = ct_window.lower().replace(" ", "_")
                if window_key in CT_WINDOWS:
                    self.preprocessor.normalizer.window = CT_WINDOWS[window_key]
                # Use the stored normalize state
                display_img = self.preprocessor.process_slice(
                    display_img,
                    normalize=self.apply_normalize,
                    resize=False
                )
                display_img = (display_img * 255).astype(np.uint8)
            else:
                display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min() + 1e-8)
                display_img = (display_img * 255).astype(np.uint8)

            return display_img, f"Slice {self.current_slice_idx}/{num_slices-1}"

        except Exception as e:
            return None, f"Error: {str(e)}"

    def auto_select_slices(self, method: str, num_slices: int) -> str:
        """Auto-select informative slices."""
        if self.current_volume is None:
            return "No 3D volume loaded"

        try:
            selector = SliceSelector(method=method.lower(), num_slices=int(num_slices))
            # Volume is (H, W, Z), slice along last axis
            indices = selector.select(self.current_volume, axis=-1)
            return f"Selected slices ({method}): {indices}"
        except Exception as e:
            return f"Error: {str(e)}"

    def load_mask(self, file) -> Tuple[Optional[np.ndarray], str]:
        """Load mask from uploaded file."""
        if file is None:
            self.current_mask = None
            return None, "Mask cleared"

        try:
            prompt = MaskPrompt()
            self.current_mask = prompt.load_mask(file.name)

            # Normalize for display
            display_mask = self.current_mask.copy()
            if display_mask.ndim == 3:
                display_mask = display_mask[display_mask.shape[0] // 2]

            display_mask = (display_mask * 255).astype(np.uint8)

            return display_mask, f"Mask shape: {self.current_mask.shape}"

        except Exception as e:
            return None, f"Error loading mask: {str(e)}"

    def create_point_mask(self, image: np.ndarray, evt: gr.SelectData) -> Tuple[np.ndarray, str]:
        """Create mask from clicked points."""
        if self.current_image is None:
            return image, "Load an image first"

        try:
            x, y = evt.index
            prompt = MaskPrompt()
            img_shape = self.current_image.shape[-2:]

            if self.current_mask is None:
                self.current_mask = np.zeros(img_shape, dtype=np.float32)

            point_mask = prompt.create_point_mask(img_shape, [(x, y)], radius=15)
            self.current_mask = np.maximum(self.current_mask, point_mask)

            # Overlay mask on image for display
            display_img = image.copy()
            if display_img.ndim == 2:
                display_img = np.stack([display_img] * 3, axis=-1)

            mask_overlay = (self.current_mask * 255).astype(np.uint8)
            # Red overlay
            display_img[:, :, 0] = np.where(mask_overlay > 0, 255, display_img[:, :, 0])

            return display_img, f"Added point at ({x}, {y})"

        except Exception as e:
            return image, f"Error: {str(e)}"

    def extract_features(self) -> str:
        """Extract features from current image."""
        if self.tester is None:
            return "Load model first"
        if self.current_image is None:
            return "Load image first"

        try:
            if self.current_image.ndim == 3:
                results = self.tester.process_volume(self.current_image, mask=self.current_mask)
                return (
                    f"Processed {len(results['slice_indices'])} slices\n"
                    f"Aggregated feature shape: {results['aggregated_features'].shape}"
                )
            else:
                features = self.tester.extract_features(self.current_image, mask=self.current_mask)
                output = f"Feature shape: {features['last_hidden_state'].shape}"
                if "mask_features" in features:
                    output += f"\nMask feature shape: {features['mask_features'].shape}"
                return output

        except Exception as e:
            return f"Error: {str(e)}"

    def classify(self) -> str:
        """Classify current image."""
        if self.tester is None:
            return "Load model first"
        if self.tester.classifier is None:
            return "No classification head loaded. Select a head and reload model."
        if self.current_image is None:
            return "Load image first"

        try:
            img = self.current_image
            mask = self.current_mask

            if img.ndim == 3:
                mid = img.shape[0] // 2
                img = img[mid]
                if mask is not None and mask.ndim == 3:
                    mask = mask[mid]

            result = self.tester.classify(img, mask=mask)

            output = f"Prediction: {result['prediction']}\n"
            output += f"Probabilities:\n"
            for i, p in enumerate(result['probabilities'][0]):
                output += f"  Class {i}: {p:.4f}\n"

            return output

        except Exception as e:
            return f"Error: {str(e)}"

    def get_attention(self) -> Tuple[Optional[np.ndarray], str]:
        """Get attention map."""
        if self.tester is None:
            return None, "Load model first"
        if self.current_image is None:
            return None, "Load image first"

        try:
            img = self.current_image
            if img.ndim == 3:
                img = img[img.shape[0] // 2]

            attn_map = self.tester.get_attention_map(img)

            # Normalize for display
            attn_display = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            attn_display = (attn_display * 255).astype(np.uint8)

            return attn_display, f"Attention range: [{attn_map.min():.4f}, {attn_map.max():.4f}]"

        except Exception as e:
            return None, f"Error: {str(e)}"

    def clear_mask(self) -> str:
        """Clear current mask."""
        self.current_mask = None
        return "Mask cleared"

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface."""
        head_choices = ["None"] + self.config.AVAILABLE_HEADS
        ct_window_choices = list(CT_WINDOWS.keys())

        with gr.Blocks(title="CURIA Medical Image Tester") as interface:
            gr.Markdown("# CURIA Medical Image Tester")
            gr.Markdown(
                "Test CURIA vision foundation model with DICOM and NIfTI images. "
                "Supports mask-based prompting for region-specific analysis."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Model Configuration")

                    head_dropdown = gr.Dropdown(
                        choices=head_choices,
                        value="None",
                        label="Classification Head"
                    )

                    device_dropdown = gr.Dropdown(
                        choices=["auto", "cuda", "mps", "cpu"],
                        value="auto",
                        label="Device"
                    )

                    hf_token = gr.Textbox(
                        label="HuggingFace Token (optional)",
                        type="password"
                    )

                    load_model_btn = gr.Button("Load Model", variant="primary")
                    model_status = gr.Textbox(label="Model Status", interactive=False)

                    gr.Markdown("---")
                    gr.Markdown("## Preprocessing")

                    modality_dropdown = gr.Dropdown(
                        choices=["Auto", "CT", "MRI"],
                        value="Auto",
                        label="Modality"
                    )

                    ct_window_dropdown = gr.Dropdown(
                        choices=ct_window_choices,
                        value="soft_tissue",
                        label="CT Window"
                    )

                    apply_preprocessing = gr.Checkbox(
                        label="Apply Preprocessing",
                        value=True
                    )

                    preprocess_status = gr.Textbox(label="Preprocessing Status", interactive=False)

                with gr.Column(scale=2):
                    gr.Markdown("## Image Input")

                    with gr.Row():
                        image_upload = gr.File(
                            label="Upload Medical Image (DICOM/NIfTI/NPY)",
                            file_types=[".dcm", ".nii", ".nii.gz", ".npy", ".npz"]
                        )
                        mask_upload = gr.File(
                            label="Upload Mask (optional)",
                            file_types=[".nii", ".nii.gz", ".npy", ".npz", ".png"]
                        )

                    with gr.Row():
                        image_display = gr.Image(label="Image Preview", interactive=True)
                        mask_display = gr.Image(label="Mask Preview")

                    image_info = gr.Textbox(label="Image Info", interactive=False, lines=6)

                    gr.Markdown("### Slice Navigation (for 3D volumes)")
                    with gr.Row():
                        slice_slider = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=50,
                            step=1,
                            label="Slice Index"
                        )
                        slice_info = gr.Textbox(label="Slice Info", interactive=False)

                    with gr.Row():
                        slice_method = gr.Dropdown(
                            choices=["Uniform", "Content", "Center"],
                            value="Uniform",
                            label="Slice Selection Method"
                        )
                        num_slices = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=10,
                            step=1,
                            label="Number of Slices"
                        )
                        auto_select_btn = gr.Button("Auto-Select Slices")
                    selected_slices_info = gr.Textbox(label="Selected Slices", interactive=False)

            with gr.Row():
                gr.Markdown("## Analysis")

            with gr.Row():
                with gr.Column():
                    extract_btn = gr.Button("Extract Features")
                    features_output = gr.Textbox(label="Features", interactive=False, lines=3)

                with gr.Column():
                    classify_btn = gr.Button("Classify")
                    classify_output = gr.Textbox(label="Classification", interactive=False, lines=5)

                with gr.Column():
                    attention_btn = gr.Button("Get Attention Map")
                    attention_display = gr.Image(label="Attention Map")
                    attention_info = gr.Textbox(label="Attention Info", interactive=False)

            with gr.Row():
                clear_mask_btn = gr.Button("Clear Mask")
                mask_status = gr.Textbox(label="Mask Status", interactive=False)

            gr.Markdown("---")
            gr.Markdown(
                "**Tips:**\n"
                "- Click on the image to add point prompts (creates a circular mask)\n"
                "- Upload a mask file for precise region selection\n"
                "- Use the slice slider to navigate through 3D volumes\n"
                "- Select CT window preset for different anatomical regions (brain, lung, etc.)\n"
                "- Use Auto-Select Slices to find informative slices automatically"
            )

            # Event handlers
            load_model_btn.click(
                fn=self.load_model,
                inputs=[head_dropdown, device_dropdown, hf_token],
                outputs=model_status
            )

            # Preprocessing settings change
            modality_dropdown.change(
                fn=self.update_preprocessing,
                inputs=[modality_dropdown, ct_window_dropdown, apply_preprocessing],
                outputs=preprocess_status
            )

            ct_window_dropdown.change(
                fn=self.update_preprocessing,
                inputs=[modality_dropdown, ct_window_dropdown, apply_preprocessing],
                outputs=preprocess_status
            )

            # Image loading - now also updates slice slider
            def load_and_update_slider(file):
                img, info, num = self.load_image(file)
                return img, info, gr.update(maximum=max(1, num-1), value=num//2)

            image_upload.change(
                fn=load_and_update_slider,
                inputs=image_upload,
                outputs=[image_display, image_info, slice_slider]
            )

            # Slice navigation
            slice_slider.change(
                fn=self.change_slice,
                inputs=[slice_slider, apply_preprocessing, ct_window_dropdown],
                outputs=[image_display, slice_info]
            )

            # Auto-select slices
            auto_select_btn.click(
                fn=self.auto_select_slices,
                inputs=[slice_method, num_slices],
                outputs=selected_slices_info
            )

            mask_upload.change(
                fn=self.load_mask,
                inputs=mask_upload,
                outputs=[mask_display, mask_status]
            )

            image_display.select(
                fn=self.create_point_mask,
                inputs=image_display,
                outputs=[image_display, mask_status]
            )

            extract_btn.click(
                fn=self.extract_features,
                outputs=features_output
            )

            classify_btn.click(
                fn=self.classify,
                outputs=classify_output
            )

            attention_btn.click(
                fn=self.get_attention,
                outputs=[attention_display, attention_info]
            )

            clear_mask_btn.click(
                fn=self.clear_mask,
                outputs=mask_status
            )

        return interface


def main():
    if not HAS_GRADIO:
        print("Please install Gradio: pip install gradio")
        return

    gui = CuriaGUI()
    interface = gui.create_interface()

    print("Starting CURIA GUI...")
    print("Open http://localhost:7860 in your browser")

    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
