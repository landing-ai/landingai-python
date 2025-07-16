"""
Multi-format Model Inference CLI

This script loads TorchScript models from published model bundles and runs inference on them.
It automatically discovers available TorchScript models in the bundle:
- TorchScript models (.pt files) - QAT models saved with torch.jit.save()

Model Loading Strategy:
- .pt files: Use torch.jit.load() for TorchScript models
- Optimized for cross-environment compatibility and deployment

The script supports both directory bundles and zip file bundles.
When a zip file is provided, it will be automatically extracted to a temporary directory.

Usage:
    # Run inference and show all predictions
    python run_inference_qat_model_exported.py --bundle_path /path/to/published_model_bundle --image_path /path/to/image.jpg
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Standard preprocessing constants (ensure float32)
IMAGENET_DEFAULT_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_DEFAULT_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Color palette for visualization
COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
    (0, 128, 0),  # Dark Green
    (139, 69, 19),  # Brown
]


class TorchScriptInference:
    """TorchScript model inference (.pt files) - for both regular TorchScript and QAT models."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """Load TorchScript model."""
        try:
            # Use torch.jit.load for TorchScript models
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            logger.info(f"Successfully loaded TorchScript model from {self.model_path}")

            # Log some debug information about the model
            logger.debug(f"TorchScript model type: {type(self.model)}")
            try:
                # Try to get input types from the model
                logger.debug(f"Model graph: {self.model.graph}")
            except:
                pass
        except Exception as e:
            logger.error(f"Failed to load TorchScript model: {e}")
            logger.error(
                f"Make sure the model was saved using torch.jit.save() or torch.jit.trace()"
            )
            raise

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for TorchScript model."""
        # Preprocessing to match the model input spec
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.resize(image, (512, 512))
            image = image.astype(np.float32)

            # Normalize using ImageNet statistics
            image = (image - IMAGENET_DEFAULT_MEAN[None, None]) / IMAGENET_DEFAULT_STD[
                None, None
            ]

            # Convert to CHW format and add batch dimension
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, axis=0)

            # Ensure float32 dtype and move to device
            tensor = torch.from_numpy(image).float().to(self.device)
            return tensor
        else:
            raise ValueError("Unsupported image format")

    def inference(self, input_data: torch.Tensor) -> Any:
        """Run inference."""
        # Debug information about input tensor
        logger.debug(f"Input tensor shape: {input_data.shape}")
        logger.debug(f"Input tensor dtype: {input_data.dtype}")
        logger.debug(f"Input tensor device: {input_data.device}")

        with torch.no_grad():
            return self.model(input_data)

    def postprocess(self, output: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract boxes, scores, labels from output."""
        # Handle the object detection model output format
        if isinstance(output, (list, tuple)) and len(output) == 6:
            # Format: logits, offsets, centers, bboxes, scores, labels
            try:
                logits, offsets, centers, bboxes, scores, labels = output

                # Convert tensors to numpy
                if isinstance(bboxes, torch.Tensor):
                    bboxes = bboxes.cpu().numpy()
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()

                # Remove batch dimension if present
                if len(bboxes.shape) == 3 and bboxes.shape[0] == 1:
                    bboxes = bboxes[0]
                if len(scores.shape) == 2 and scores.shape[0] == 1:
                    scores = scores[0]
                if len(labels.shape) == 2 and labels.shape[0] == 1:
                    labels = labels[0]

                return bboxes, scores, labels

            except Exception as e:
                logger.error(f"Failed to parse 6-element output: {e}")

        # Handle different output formats (legacy support)
        if isinstance(output, torch.Tensor):
            # Single tensor output - might be [batch, num_detections, 6] format (x1,y1,x2,y2,conf,class)
            if len(output.shape) == 3 and output.shape[2] >= 6:
                output_np = output.cpu().numpy()[0]  # Remove batch dimension
                boxes = output_np[:, :4]
                scores = output_np[:, 4]
                labels = output_np[:, 5].astype(int)
                return boxes, scores, labels
        elif isinstance(output, (list, tuple)):
            # Multiple outputs - try to identify boxes, scores, labels
            if len(output) >= 3:
                boxes = (
                    output[0].cpu().numpy()
                    if isinstance(output[0], torch.Tensor)
                    else output[0]
                )
                scores = (
                    output[1].cpu().numpy()
                    if isinstance(output[1], torch.Tensor)
                    else output[1]
                )
                labels = (
                    output[2].cpu().numpy()
                    if isinstance(output[2], torch.Tensor)
                    else output[2]
                )
                return boxes, scores, labels

        # Default empty return
        return np.array([]), np.array([]), np.array([])


def extract_bundle_if_zip(bundle_path: str) -> str:
    """Extract bundle if it's a zip file, otherwise return the original path."""
    bundle_path = Path(bundle_path)

    if not bundle_path.exists():
        raise ValueError(f"Bundle path does not exist: {bundle_path}")

    # If it's a zip file, extract it to a temporary directory
    if bundle_path.is_file() and bundle_path.suffix.lower() == ".zip":
        logger.info(f"Extracting zip file: {bundle_path}")

        # Create a temporary directory for extraction
        temp_dir = tempfile.mkdtemp(prefix="model_bundle_")

        try:
            with zipfile.ZipFile(bundle_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            # Find the extracted bundle directory
            # Sometimes zip files have a root folder, sometimes they don't
            extracted_items = list(Path(temp_dir).iterdir())

            if len(extracted_items) == 1 and extracted_items[0].is_dir():
                # Single directory in the zip - use that as the bundle path
                bundle_dir = extracted_items[0]
            else:
                # Multiple items or files - use the temp directory itself
                bundle_dir = Path(temp_dir)

            logger.info(f"Extracted bundle to: {bundle_dir}")
            return str(bundle_dir)

        except Exception as e:
            # Clean up temp directory if extraction fails
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ValueError(f"Failed to extract zip file: {e}")

    # If it's already a directory, return as-is
    elif bundle_path.is_dir():
        return str(bundle_path)
    else:
        raise ValueError(
            f"Bundle path must be either a directory or a zip file: {bundle_path}"
        )


def discover_bundle_resources(bundle_path: str) -> Dict[str, Any]:
    """Discover available TorchScript resources in a published model bundle."""
    bundle_path = Path(bundle_path)

    if not bundle_path.exists():
        raise ValueError(f"Bundle path does not exist: {bundle_path}")

    resources = {
        "models": {},
        "defect_map": None,
        "meta": None,
        "config": None,
        "model_meta": {},
    }

    # Look for TorchScript models (.pt files) in the model subdirectory
    model_dir = bundle_path / "model"
    if model_dir.exists():
        for model_file in model_dir.iterdir():
            if model_file.is_file():
                suffix = model_file.suffix.lower()
                name = model_file.stem

                if suffix == ".pt":  # Only TorchScript models
                    resources["models"][name] = model_file
                elif name == "original_code" and suffix == ".py":
                    resources["config"] = model_file

    # Look for model metadata (contains training configuration)
    model_meta_dir = bundle_path / "model_meta"
    if model_meta_dir.exists():
        for meta_file in model_meta_dir.iterdir():
            if meta_file.is_file():
                resources["model_meta"][meta_file.stem] = meta_file

    # Look for defect map
    dm_file = bundle_path / "dm.json"
    if dm_file.exists():
        resources["defect_map"] = dm_file

    # Look for meta information
    meta_file = bundle_path / "meta.json"
    resources["meta"] = meta_file

    logger.info(f"Discovered TorchScript models: {list(resources['models'].keys())}")
    logger.info(f"Model metadata: {list(resources['model_meta'].keys())}")

    return resources


def select_model_from_bundle(
    models: Dict[str, Path], preferred_model: Optional[str] = None
) -> Tuple[str, Path]:
    """Select which TorchScript model to use from available models."""
    if not models:
        raise ValueError(
            "No TorchScript models (.pt files) found in bundle. Use inspect_model_bundle.py to analyze other model formats."
        )

    # Priority order for TorchScript model selection
    priority_order = [
        "qat_eval-saved_model",  # QAT evaluation model (TorchScript) - best for inference
        "qat-saved_model",  # QAT training model (TorchScript)
    ]

    # If user specified a preferred model, try to use it
    if preferred_model:
        if preferred_model in models:
            return preferred_model, models[preferred_model]
        else:
            logger.warning(
                f"Preferred model '{preferred_model}' not found. Available TorchScript models: {list(models.keys())}"
            )

    # Select based on priority order
    for preferred in priority_order:
        if preferred in models:
            logger.info(f"Selected TorchScript model: {preferred}")
            return preferred, models[preferred]

    # Fallback to first available model
    model_name = list(models.keys())[0]
    logger.info(f"Using fallback TorchScript model: {model_name}")
    return model_name, models[model_name]


def load_meta_info(meta_path: str) -> Dict[str, Any]:
    """Load and validate meta.json file."""
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Validate required fields
        required_fields = ["modelId", "bundleVersion", "modelFlavor"]
        missing_fields = [field for field in required_fields if field not in meta]
        if missing_fields:
            logger.warning(f"Missing required fields in meta.json: {missing_fields}")

        # Log meta information
        logger.info(f"Model metadata:")
        logger.info(f"  Model ID: {meta.get('modelId', 'Unknown')}")
        logger.info(f"  Bundle Version: {meta.get('bundleVersion', 'Unknown')}")
        logger.info(f"  Model Flavor: {meta.get('modelFlavor', 'Unknown')}")
        logger.info(f"  Model Version: {meta.get('modelVersion', 'Unknown')}")
        logger.info(f"  Torch Model: {meta.get('torchModel', 'Unknown')}")

        return meta

    except Exception as e:
        raise ValueError(f"Failed to load meta.json: {e}")


def load_defect_map(dm_path: Optional[str]) -> Dict[str, Any]:
    """Load defect map from JSON file in its original format."""
    if not dm_path or not os.path.exists(dm_path):
        raise FileNotFoundError(
            f"Defect map not found at {dm_path}. This is required for inference."
        )

    try:
        with open(dm_path, "r") as f:
            defect_map = json.load(f)

        if not defect_map:
            raise ValueError("Defect map is empty - cannot determine class labels")

        logger.info(
            f"Loaded defect map with {len(defect_map)} classes: {list(defect_map.keys())}"
        )
        return defect_map
    except Exception as e:
        raise ValueError(f"Failed to load defect map: {e}")


def load_image(image_path: str) -> np.ndarray:
    """Load image from file using OpenCV."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def draw_bboxes_on_image(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    defect_map: Dict[int, str],
) -> np.ndarray:
    """Draw bounding boxes on image for all predictions."""
    if len(boxes) == 0:
        return image

    # Convert to PIL for drawing
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()

    height, width = image.shape[:2]

    # Expand color palette if needed for more classes
    extended_colors = COLORS.copy()
    while len(extended_colors) < len(defect_map):
        # Generate additional colors by varying the existing ones
        for base_color in COLORS:
            r, g, b = base_color
            # Create variations by adjusting brightness
            new_color = (min(255, r + 50), min(255, g + 50), min(255, b + 50))
            extended_colors.append(new_color)
            if len(extended_colors) >= len(defect_map):
                break

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        # Handle different box formats
        if len(box) == 4:
            if np.all(box <= 1.0):
                # Normalized coordinates
                x1, y1, x2, y2 = box
                x1, x2 = x1 * width, x2 * width
                y1, y2 = y1 * height, y2 * height
            else:
                # Absolute coordinates
                x1, y1, x2, y2 = box
        else:
            logger.warning(f"Unsupported box format: {box}")
            continue

        # Ensure coordinates are within image bounds
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(width, int(x2)), min(height, int(y2))

        # Skip invalid rectangles (final safety check)
        if x2 <= x1 or y2 <= y1:
            logger.warning(
                f"Skipping invalid rectangle: [{x1}, {y1}, {x2}, {y2}] for detection {i}"
            )
            continue

        # Get color for this detection - use label-based coloring for consistency
        color_index = int(label) % len(extended_colors)
        color = extended_colors[color_index]

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label and score
        label_name = defect_map.get(int(label), f"class_{int(label)}")
        text = f"{label_name}: {score:.2f}"

        # Calculate text size and position
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position text above the box if possible
        text_y = y1 - text_height - 2 if y1 - text_height - 2 > 0 else y1 + 2
        text_x = x1

        # Draw text background
        draw.rectangle(
            [text_x, text_y, text_x + text_width, text_y + text_height], fill=color
        )

        # Draw text
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

    return np.array(pil_image)


def get_model_inference(model_path: str) -> TorchScriptInference:
    """Create TorchScript inference object for .pt files."""
    model_path = Path(model_path)

    if model_path.suffix.lower() == ".pt":
        return TorchScriptInference(str(model_path))
    else:
        raise ValueError(
            f"Unsupported model format: {model_path.suffix}. This script only supports TorchScript (.pt) files. "
            f"For PyTorch model inspection, use inspect_model_bundle.py"
        )


def main():
    parser = argparse.ArgumentParser(
        description="TorchScript Model Inference CLI - Optimized for deployment models"
    )
    parser.add_argument(
        "--bundle_path",
        required=True,
        help="Path to published model bundle directory or zip file",
    )
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument(
        "--model_name",
        help="Specific TorchScript model name to use (e.g., 'qat_eval-saved_model', 'qat-saved_model')",
    )
    parser.add_argument(
        "--output_path", help="Path to save output image with annotations"
    )

    parser.add_argument("--show", action="store_true", help="Display the output image")
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List available TorchScript models in bundle and exit",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    extracted_bundle_path = None
    temp_cleanup_needed = False

    try:
        # Handle zip file extraction if needed
        logger.info(f"Processing bundle: {args.bundle_path}")
        extracted_bundle_path = extract_bundle_if_zip(args.bundle_path)

        # Check if we extracted to a temporary directory (cleanup needed later)
        if extracted_bundle_path != args.bundle_path:
            temp_cleanup_needed = True

        # Discover bundle resources
        logger.info(f"Discovering resources in bundle: {extracted_bundle_path}")
        resources = discover_bundle_resources(extracted_bundle_path)

        # List models if requested
        if args.list_models:
            print("Available TorchScript models in bundle:")
            if resources["models"]:
                for model_name, model_path in resources["models"].items():
                    print(f"  - {model_name} ({model_path.suffix})")
            else:
                print("  No TorchScript (.pt) models found.")
                print("  Use inspect_model_bundle.py to analyze other model formats.")
            return

        # Load and validate meta information
        meta_info = load_meta_info(str(resources["meta"]))

        # Load defect map once (required for visualization)
        raw_defect_map = load_defect_map(
            str(resources["defect_map"]) if resources["defect_map"] else None
        )

        # Check if there are any TorchScript models available
        if not resources["models"]:
            logger.error("No TorchScript (.pt) models found in bundle.")
            logger.error("This script only supports TorchScript models for inference.")
            logger.error(
                "To inspect other model formats (.pth files), use: python inspect_model_bundle.py"
            )
            return

        # Select model to use
        model_name, model_path = select_model_from_bundle(
            resources["models"], args.model_name
        )

        # Load TorchScript model
        logger.info(f"Loading TorchScript model: {model_name} from {model_path}")
        inference = get_model_inference(str(model_path))
        inference.load_model()

        # Load image
        logger.info(f"Loading image from {args.image_path}")
        image = load_image(args.image_path)
        original_image = image.copy()

        # Convert raw defect map to visualization format (int keys for label mapping)
        defect_map_for_viz = {}
        for key, value in raw_defect_map.items():
            if isinstance(value, dict) and "name" in value:
                defect_map_for_viz[int(key)] = value["name"]
            else:
                defect_map_for_viz[int(key)] = str(value)

        logger.info(f"Using defect map for visualization: {defect_map_for_viz}")

        # Preprocess
        logger.info("Preprocessing image")
        input_data = inference.preprocess(image)

        # Run inference
        logger.info("Running inference")
        try:
            output = inference.inference(input_data)
        except RuntimeError as e:
            if "expected scalar type Double but found Float" in str(e):
                logger.error(
                    "Data type mismatch error. Trying to convert input to double precision..."
                )
                # Try converting to double precision
                input_data = input_data.double()
                logger.info(f"Converted input tensor to dtype: {input_data.dtype}")
                output = inference.inference(input_data)
            else:
                raise e

        # Postprocess
        logger.info("Postprocessing output")
        boxes, scores, labels = inference.postprocess(output)

        # Filter out invalid bounding boxes where x1 < x0 or y1 < y0
        if len(boxes) > 0:
            valid_indices = []
            for i, box in enumerate(boxes):
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    # Check for NaN or infinite values
                    if np.any(np.isnan(box)) or np.any(np.isinf(box)):
                        continue
                    # Check if coordinates form a valid rectangle (x2 > x1 and y2 > y1)
                    # Use > instead of >= to ensure non-zero area
                    if x2 > x1 and y2 > y1:
                        valid_indices.append(i)

            valid_indices = np.array(valid_indices)
            original_count = len(boxes)

            if len(valid_indices) > 0:
                boxes = boxes[valid_indices]
                scores = scores[valid_indices]
                labels = labels[valid_indices]
                removed_count = original_count - len(boxes)
                if removed_count > 0:
                    logger.info(
                        f"Removed {removed_count} invalid bounding boxes (NaN, infinite, or x2 <= x1 or y2 <= y1)"
                    )
            else:
                # All boxes are invalid
                boxes = np.array([])
                scores = np.array([])
                labels = np.array([])
                logger.info(
                    f"Removed all {original_count} detections due to invalid bounding box coordinates"
                )

        # Log all predictions (after filtering invalid boxes)
        logger.info(f"Found {len(boxes)} valid detections")

        # Print all results (unfiltered)
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            label_name = defect_map_for_viz.get(label, f"class_{label}")
            logger.info(f"Detection {i+1}: {label_name} ({score:.3f}) at {box}")

        # Show all predictions in visualization
        logger.info(f"Showing all {len(boxes)} detections in visualization")

        # Draw bounding boxes (all predictions)
        if len(boxes) > 0:
            result_image = draw_bboxes_on_image(
                original_image, boxes, scores, labels, defect_map_for_viz
            )
        else:
            result_image = original_image
            logger.info("No detections to visualize")

        # Save output image
        if args.output_path:
            result_pil = Image.fromarray(result_image)
            result_pil.save(args.output_path)
            logger.info(f"Saved result image to {args.output_path}")

        # Show image
        if args.show:
            result_pil = Image.fromarray(result_image)
            result_pil.show()

        logger.info("Inference completed successfully")

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    finally:
        # Clean up temporary directory if we extracted a zip file
        if temp_cleanup_needed and extracted_bundle_path:
            try:
                import shutil

                # Get the parent temp directory to clean up
                temp_dir = (
                    Path(extracted_bundle_path).parent
                    if Path(extracted_bundle_path).parent.name.startswith(
                        "model_bundle_"
                    )
                    else extracted_bundle_path
                )
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to clean up temporary directory: {cleanup_error}"
                )


if __name__ == "__main__":
    main()
