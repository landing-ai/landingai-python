"""
Model Bundle Inspector

This script inspects published model bundles and displays relevant information about models and state dictionaries.
It provides detailed analysis of:
- Model architecture and parameters
- State dict structure and sizes
- Bundle metadata and configuration
- Defect map information

Usage:
    # Inspect a model bundle
    python inspect_model_bundle.py --bundle_path /path/to/published_model_bundle

    # Inspect specific model
    python inspect_model_bundle.py --bundle_path /path/to/bundle --model_name qat_eval-saved_model

    # Show detailed state dict analysis
    python inspect_model_bundle.py --bundle_path /path/to/bundle --detailed --show_weights
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            extracted_items = list(Path(temp_dir).iterdir())

            if len(extracted_items) == 1 and extracted_items[0].is_dir():
                bundle_dir = extracted_items[0]
            else:
                bundle_dir = Path(temp_dir)

            logger.info(f"Extracted bundle to: {bundle_dir}")
            return str(bundle_dir)

        except Exception as e:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ValueError(f"Failed to extract zip file: {e}")

    elif bundle_path.is_dir():
        return str(bundle_path)
    else:
        raise ValueError(
            f"Bundle path must be either a directory or a zip file: {bundle_path}"
        )


def discover_bundle_resources(bundle_path: str) -> Dict[str, Any]:
    """Discover available resources in a published model bundle."""
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

    # Look for models in the model subdirectory
    model_dir = bundle_path / "model"
    if model_dir.exists():
        for model_file in model_dir.iterdir():
            if model_file.is_file():
                suffix = model_file.suffix.lower()
                name = model_file.stem

                if suffix in [".pth", ".pt"]:
                    resources["models"][name] = model_file
                elif name == "original_code" and suffix == ".py":
                    resources["config"] = model_file

    # Look for model metadata
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
    if meta_file.exists():
        resources["meta"] = meta_file

    return resources


def load_defect_map(dm_path: Optional[str]) -> Dict[str, Any]:
    """Load defect map from JSON file."""
    if not dm_path or not os.path.exists(dm_path):
        return {}

    try:
        with open(dm_path, "r") as f:
            defect_map = json.load(f)
        return defect_map
    except Exception as e:
        logger.warning(f"Failed to load defect map: {e}")
        return {}


def load_meta_info(meta_path: Optional[str]) -> Dict[str, Any]:
    """Load and return meta.json file."""
    if not meta_path or not os.path.exists(meta_path):
        return {}

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return meta
    except Exception as e:
        logger.warning(f"Failed to load meta.json: {e}")
        return {}


def load_model_metadata(model_meta: Dict[str, Path]) -> Dict[str, Any]:
    """Load model metadata files."""
    metadata = {}

    for name, path in model_meta.items():
        try:
            if path.suffix.lower() in [".yaml", ".yml"]:
                import yaml

                with open(path, "r") as f:
                    metadata[name] = yaml.safe_load(f)
            elif path.suffix.lower() == ".json":
                with open(path, "r") as f:
                    metadata[name] = json.load(f)
            else:
                with open(path, "r") as f:
                    metadata[name] = f.read()
        except Exception as e:
            logger.warning(f"Failed to load {name} metadata: {e}")
            metadata[name] = f"Error loading: {e}"

    return metadata


def analyze_pytorch_model(
    model_path: Path, detailed: bool = False, show_weights: bool = False
) -> Dict[str, Any]:
    """Analyze a PyTorch model file (.pth files containing state dicts or complete models)."""
    analysis = {
        "file_path": str(model_path),
        "file_size": model_path.stat().st_size,
        "type": "unknown",
        "loadable": False,
        "error": None,
    }

    try:
        # Try to load the file with torch.load (for .pth files)
        data = torch.load(model_path, map_location="cpu")
        analysis["loadable"] = True

        if isinstance(data, torch.nn.Module):
            analysis["type"] = "complete_model"
            analysis["model_class"] = type(data).__name__
            analysis["model_info"] = analyze_model_structure(
                data, detailed, show_weights
            )

        elif isinstance(data, dict):
            analysis["type"] = "state_dict"
            analysis["state_dict_info"] = analyze_state_dict(
                data, detailed, show_weights
            )

        else:
            analysis["type"] = "other"
            analysis["data_type"] = type(data).__name__

    except Exception as e:
        analysis["error"] = str(e)
        # Don't log here, let the print_model_analysis function handle error display

    return analysis


def analyze_torchscript_model(
    model_path: Path, detailed: bool = False
) -> Dict[str, Any]:
    """Analyze a TorchScript model file (.pt files that contain TorchScript models)."""
    analysis = {
        "file_path": str(model_path),
        "file_size": model_path.stat().st_size,
        "type": "torchscript",
        "loadable": False,
        "error": None,
    }

    try:
        # Try to load the TorchScript model with torch.jit.load
        model = torch.jit.load(model_path, map_location="cpu")
        analysis["loadable"] = True
        analysis["model_class"] = type(model).__name__

        # Get basic model information
        try:
            # Try to get the graph
            graph_str = str(model.graph)
            analysis["has_graph"] = True
            if detailed:
                analysis["graph"] = (
                    graph_str[:1000] + "..." if len(graph_str) > 1000 else graph_str
                )
        except:
            analysis["has_graph"] = False

        # Try to get parameters
        try:
            params = list(model.parameters())
            analysis["num_parameters"] = len(params)
            analysis["total_params"] = sum(p.numel() for p in params)
            analysis["trainable_params"] = sum(
                p.numel() for p in params if p.requires_grad
            )
        except:
            analysis["num_parameters"] = "unknown"

    except Exception as e:
        analysis["error"] = str(e)
        # Don't log here, let the print_model_analysis function handle error display

    return analysis


def analyze_model_structure(
    model: torch.nn.Module, detailed: bool = False, show_weights: bool = False
) -> Dict[str, Any]:
    """Analyze the structure of a PyTorch model."""
    info = {}

    # Basic model information
    info["class_name"] = type(model).__name__
    info["training_mode"] = model.training

    # Parameter analysis
    params = list(model.parameters())
    info["num_parameters"] = len(params)
    info["total_params"] = sum(p.numel() for p in params)
    info["trainable_params"] = sum(p.numel() for p in params if p.requires_grad)

    # Module analysis
    modules = list(model.modules())
    info["num_modules"] = len(modules)

    if detailed:
        # Detailed module breakdown
        module_types = {}
        for module in modules:
            module_type = type(module).__name__
            module_types[module_type] = module_types.get(module_type, 0) + 1
        info["module_breakdown"] = module_types

        # Named modules
        named_modules = dict(model.named_modules())
        info["module_names"] = list(named_modules.keys())[:20]  # First 20

        if show_weights:
            # Parameter details
            param_details = {}
            for name, param in model.named_parameters():
                param_details[name] = {
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                    "requires_grad": param.requires_grad,
                    "numel": param.numel(),
                }
            info["parameter_details"] = param_details

    return info


def analyze_state_dict(
    state_dict: Dict[str, torch.Tensor],
    detailed: bool = False,
    show_weights: bool = False,
) -> Dict[str, Any]:
    """Analyze a PyTorch state dictionary."""
    info = {}

    # Basic information
    info["num_keys"] = len(state_dict)
    info["total_params"] = sum(
        tensor.numel()
        for tensor in state_dict.values()
        if isinstance(tensor, torch.Tensor)
    )

    # Key analysis
    keys = list(state_dict.keys())
    info["sample_keys"] = keys[:10]  # First 10 keys

    # Data type analysis
    dtypes = {}
    shapes = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            dtype = str(value.dtype)
            dtypes[dtype] = dtypes.get(dtype, 0) + 1

            shape_str = str(list(value.shape))
            shapes[shape_str] = shapes.get(shape_str, 0) + 1

    info["dtype_distribution"] = dtypes

    if detailed:
        info["shape_distribution"] = dict(list(shapes.items())[:20])  # Top 20 shapes

        # Layer analysis (try to infer layer types from key names)
        layer_types = {}
        for key in keys:
            if "conv" in key.lower():
                layer_types["conv"] = layer_types.get("conv", 0) + 1
            elif "linear" in key.lower() or "fc" in key.lower():
                layer_types["linear"] = layer_types.get("linear", 0) + 1
            elif "bn" in key.lower() or "batchnorm" in key.lower():
                layer_types["batchnorm"] = layer_types.get("batchnorm", 0) + 1
            elif "bias" in key.lower():
                layer_types["bias"] = layer_types.get("bias", 0) + 1
            elif "weight" in key.lower():
                layer_types["weight"] = layer_types.get("weight", 0) + 1

        info["inferred_layer_types"] = layer_types

        if show_weights:
            # Detailed parameter information
            param_details = {}
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor):
                    param_details[key] = {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "numel": value.numel(),
                        "mean": float(value.float().mean()) if value.numel() > 0 else 0,
                        "std": float(value.float().std()) if value.numel() > 0 else 0,
                        "min": float(value.min()) if value.numel() > 0 else 0,
                        "max": float(value.max()) if value.numel() > 0 else 0,
                    }
                else:
                    param_details[key] = {
                        "type": type(value).__name__,
                        "value": str(value),
                    }
            info["parameter_details"] = param_details

    return info


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def print_bundle_summary(
    resources: Dict[str, Any],
    meta_info: Dict[str, Any],
    defect_map: Dict[str, Any],
    model_metadata: Dict[str, Any],
):
    """Print a summary of the bundle."""
    print("\n" + "=" * 80)
    print("MODEL BUNDLE SUMMARY")
    print("=" * 80)

    # Meta information
    if meta_info:
        print(f"\nMeta Information:")
        print(f"  Model ID: {meta_info.get('modelId', 'Unknown')}")
        print(f"  Bundle Version: {meta_info.get('bundleVersion', 'Unknown')}")
        print(f"  Model Flavor: {meta_info.get('modelFlavor', 'Unknown')}")
        print(f"  Model Version: {meta_info.get('modelVersion', 'Unknown')}")
        print(f"  Torch Model: {meta_info.get('torchModel', 'Unknown')}")
        print(f"  Input Type: {meta_info.get('inputType', 'Unknown')}")
        print(f"  Output Type: {meta_info.get('outputType', 'Unknown')}")

    # Defect map
    if defect_map:
        print(f"\nDefect Map ({len(defect_map)} classes):")
        for key, value in defect_map.items():
            if isinstance(value, dict):
                name = value.get("name", str(value))
                class_id = value.get("id", "None")
                print(f"  {key}: {name} (ID: {class_id})")
            else:
                print(f"  {key}: {value}")

    # Available models
    print(f"\nAvailable Models ({len(resources['models'])}):")
    for name, path in resources["models"].items():
        file_size = format_file_size(path.stat().st_size)
        print(f"  {name}{path.suffix} ({file_size})")


def print_model_analysis(name: str, analysis: Dict[str, Any], detailed: bool = False):
    """Print analysis results for a single model."""
    print(f"\n{'='*80}")
    print(f"MODEL ANALYSIS: {name}")
    print("=" * 80)

    print(f"File: {analysis['file_path']}")
    print(f"Size: {format_file_size(analysis['file_size'])}")
    print(f"Type: {analysis['type']}")
    print(f"Loadable: {analysis['loadable']}")

    if analysis.get("error"):
        print(f"Error: {analysis['error']}")
        return

    if analysis["type"] == "complete_model":
        model_info = analysis["model_info"]
        print(f"Model Class: {model_info['class_name']}")
        print(f"Training Mode: {model_info['training_mode']}")
        print(f"Total Parameters: {model_info['total_params']:,}")
        print(f"Trainable Parameters: {model_info['trainable_params']:,}")
        print(f"Number of Modules: {model_info['num_modules']}")

        if detailed and "module_breakdown" in model_info:
            print(f"\nModule Breakdown:")
            for module_type, count in model_info["module_breakdown"].items():
                print(f"  {module_type}: {count}")

    elif analysis["type"] == "state_dict":
        state_info = analysis["state_dict_info"]
        print(f"Number of Keys: {state_info['num_keys']}")
        print(f"Total Parameters: {state_info['total_params']:,}")

        print(f"\nData Type Distribution:")
        for dtype, count in state_info["dtype_distribution"].items():
            print(f"  {dtype}: {count} tensors")

        print(f"\nSample Keys:")
        for key in state_info["sample_keys"]:
            print(f"  {key}")

        if detailed and "inferred_layer_types" in state_info:
            print(f"\nInferred Layer Types:")
            for layer_type, count in state_info["inferred_layer_types"].items():
                print(f"  {layer_type}: {count}")

    elif analysis["type"] == "torchscript":
        print(f"Model Class: {analysis['model_class']}")
        print(f"Has Graph: {analysis['has_graph']}")
        if "total_params" in analysis:
            print(f"Total Parameters: {analysis['total_params']:,}")
            print(f"Trainable Parameters: {analysis['trainable_params']:,}")


def main():
    parser = argparse.ArgumentParser(description="Model Bundle Inspector")
    parser.add_argument(
        "--bundle_path",
        required=True,
        help="Path to published model bundle directory or zip file",
    )
    parser.add_argument(
        "--model_name",
        help="Specific model name to analyze (if not provided, analyzes all models)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed analysis including module breakdown and layer types",
    )
    parser.add_argument(
        "--show_weights",
        action="store_true",
        help="Show detailed weight statistics (implies --detailed)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.show_weights:
        args.detailed = True

    extracted_bundle_path = None
    temp_cleanup_needed = False

    try:
        # Handle zip file extraction if needed
        logger.info(f"Processing bundle: {args.bundle_path}")
        extracted_bundle_path = extract_bundle_if_zip(args.bundle_path)

        # Check if we extracted to a temporary directory
        if extracted_bundle_path != args.bundle_path:
            temp_cleanup_needed = True

        # Discover bundle resources
        logger.info(f"Discovering resources in bundle: {extracted_bundle_path}")
        resources = discover_bundle_resources(extracted_bundle_path)

        # Load bundle information
        meta_info = load_meta_info(
            str(resources["meta"]) if resources["meta"] else None
        )
        defect_map = load_defect_map(
            str(resources["defect_map"]) if resources["defect_map"] else None
        )
        model_metadata = load_model_metadata(resources["model_meta"])

        # Print bundle summary
        print_bundle_summary(resources, meta_info, defect_map, model_metadata)

        # Analyze models
        models_to_analyze = {}
        if args.model_name:
            if args.model_name in resources["models"]:
                models_to_analyze[args.model_name] = resources["models"][
                    args.model_name
                ]
            else:
                print(f"\nError: Model '{args.model_name}' not found in bundle.")
                print(f"Available models: {list(resources['models'].keys())}")
                return
        else:
            models_to_analyze = resources["models"]

        # Analyze each model
        for name, path in models_to_analyze.items():
            # Ensure output is flushed before starting new model analysis
            sys.stdout.flush()
            sys.stderr.flush()

            if path.suffix.lower() == ".pth":
                analysis = analyze_pytorch_model(path, args.detailed, args.show_weights)
            elif path.suffix.lower() == ".pt":
                analysis = analyze_torchscript_model(path, args.detailed)
            else:
                continue

            print_model_analysis(name, analysis, args.detailed)

            # Flush output after each model analysis
            sys.stdout.flush()
            sys.stderr.flush()

        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Error during inspection: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    finally:
        # Clean up temporary directory if we extracted a zip file
        if temp_cleanup_needed and extracted_bundle_path:
            try:
                import shutil

                temp_dir = (
                    Path(extracted_bundle_path).parent
                    if Path(extracted_bundle_path).parent.name.startswith(
                        "model_bundle_"
                    )
                    else extracted_bundle_path
                )
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to clean up temporary directory: {cleanup_error}"
                )


if __name__ == "__main__":
    main()
