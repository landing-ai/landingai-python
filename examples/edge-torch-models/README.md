# QAT Model Tools

This directory contains tools for working with published model bundles containing QAT (Quantization Aware Training) models. The tools support model inspection and inference on TorchScript models.

## Available Scripts

### 1. Model Bundle Inspector (`inspect_model_bundle.py`)

A diagnostic tool that inspects published model bundles and displays detailed information about models and state dictionaries.

**Features:**
- Model architecture and parameter analysis  
- State dict structure and sizes
- Bundle metadata and configuration inspection
- Defect map information
- Support for both .pth and .pt model files
- Detailed error reporting for failed model loads

**Usage:**
```bash
# Inspect a model bundle
python inspect_model_bundle.py --bundle_path /path/to/published_model_bundle

# Inspect specific model
python inspect_model_bundle.py --bundle_path /path/to/bundle --model_name qat_eval-saved_model

# Show detailed state dict analysis
python inspect_model_bundle.py --bundle_path /path/to/bundle --detailed --show_weights
```

### 2. Model Inference CLI (`run_inference_qat_model_exported.py`)

A tool that loads TorchScript models from published model bundles and runs object detection inference on them.

**Features:**
- **Automatic Resource Discovery**: Scans the bundle directory to find available models, defect maps, and configuration files
- **TorchScript Model Support**: Optimized for .pt files using torch.jit.load()
- **Intelligent Model Selection**: Automatically selects the best available model or allows manual selection
- **Bounding Box Visualization**: Draws detection results on images with labels and confidence scores
- **Flexible Output Options**: Save results to file, display on screen, or both

## Bundle Structure

Both tools expect a published model bundle with the following structure:

```
published_model_bundle/
├── model/
│   ├── before_train-saved_model.pth    # Pre-training PyTorch model
│   ├── qat-saved_model.pt              # QAT training TorchScript model  
│   ├── qat_eval-saved_model.pt         # QAT evaluation TorchScript model
│   ├── saved_model.tflite              # TensorFlow Lite model (not supported)
│   ├── original_code.py                # Model configuration/code
│   └── original_graph.txt              # Model graph description
├── dm.json                             # Defect map (class labels)
├── meta.json                           # Model metadata
└── original_dm.json                    # Original defect map
```

**Note**: The inference tool (`run_inference_qat_model_exported.py`) only supports TorchScript models (.pt files), while the inspector (`inspect_model_bundle.py`) supports both .pth and .pt files.

## Usage Examples

### Model Bundle Inspection

### Model Bundle Inspection

```bash
# Basic bundle inspection
python inspect_model_bundle.py --bundle_path /path/to/published_model_bundle

# Inspect specific model with detailed analysis
python inspect_model_bundle.py \
    --bundle_path /path/to/published_model_bundle \
    --model_name qat_eval-saved_model \
    --detailed

# Show detailed weight statistics
python inspect_model_bundle.py \
    --bundle_path /path/to/published_model_bundle \
    --detailed \
    --show_weights
```

### Model Inference

```bash
# Basic inference
python run_inference_qat_model_exported.py \
    --bundle_path /path/to/published_model_bundle \
    --image_path /path/to/image.jpg \
    --output_path result.jpg
```

# List available models for inference
python run_inference_qat_model_exported.py \
    --bundle_path /path/to/published_model_bundle \
    --image_path dummy.jpg \
    --list_models

# Use specific model for inference
python run_inference_qat_model_exported.py \
    --bundle_path /path/to/published_model_bundle \
    --image_path /path/to/image.jpg \
    --model_name qat_eval-saved_model \
    --output_path result.jpg \
    --score_threshold 0.3

# Display inference results on screen
python run_inference_qat_model_exported.py \
    --bundle_path /path/to/published_model_bundle \
    --image_path /path/to/image.jpg \
    --show \
    --verbose
```

## Command Line Arguments

### Model Bundle Inspector

- `--bundle_path`: Path to the published model bundle directory or zip file (required)
- `--model_name`: Specific model to analyze (optional, analyzes all models if not provided)
- `--detailed`: Show detailed analysis including module breakdown and layer types
- `--show_weights`: Show detailed weight statistics (implies --detailed)
- `--verbose`: Enable verbose logging

### Model Inference CLI

- `--bundle_path`: Path to the published model bundle directory or zip file (required)
- `--image_path`: Path to the input image (required)
- `--model_name`: Specific model to use (optional, auto-selects if not provided)
- `--output_path`: Path to save the annotated output image (optional)
- `--score_threshold`: Confidence threshold for detections (default: 0.5)
- `--show`: Display the result image on screen
- `--list_models`: List available models in the bundle and exit
- `--verbose`: Enable verbose logging

## Model Analysis Output (Inspector)

The model bundle inspector provides detailed analysis including:

- **File Information**: Path, size, type, and loadability status
- **Model Architecture**: Class name, training mode, parameter counts, module counts
- **State Dictionary Analysis**: Key counts, data type distribution, sample keys, inferred layer types
- **TorchScript Models**: Graph availability, parameter information
- **Bundle Metadata**: Model ID, version, flavor, input/output types
- **Defect Map**: Class labels and IDs for object detection

## Model Selection Priority (Inference)

When `--model_name` is not specified, the inference CLI automatically selects TorchScript models (.pt files) in this priority order:

1. `qat_eval-saved_model` - QAT evaluation model (recommended for inference)
2. `qat-saved_model` - QAT training model
3. `saved_model` - Generic saved model

**Note**: Only TorchScript (.pt) models are supported for inference. PyTorch (.pth) models are only supported by the inspector tool.

## Requirements

- Python 3.11+
- PyTorch
- OpenCV (for inference tool)
- PIL/Pillow (for inference tool)
- NumPy
- Optional: AVI libraries for enhanced preprocessing

## Supported Model Types

### TorchScript Models (.pt files)
- **Inspector**: Full analysis including graph structure, parameters, and metadata
- **Inference**: Direct loading with torch.jit.load() for optimized inference
- QAT models converted to TorchScript format for deployment

### PyTorch Models (.pth files)  
- **Inspector only**: Analysis of state dictionaries and model checkpoints
- **Not supported for inference**: Requires model architecture reconstruction

**Important Note**: PyTorch models (.pth files) will likely fail to load due to missing dependencies such as custom model classes or proprietary components that are not included in the bundle or publicly available. For such models:
- Use the inspection script (`inspect_model_bundle.py`) for basic file analysis
- Use external inspection tools like [Netron](https://netron.app/) for detailed model visualization and structure analysis
- The inspection script will provide error details and basic file information even when models fail to load

## Output Formats

### Inspector Output
- Console display with detailed model analysis
- Bundle summary with metadata and available resources
- Model-specific analysis including architecture and parameters
- Error reporting for models that fail to load

### Inference Output
1. **Console logs**: Detection results with bounding box coordinates, confidence scores, and class labels
2. **Annotated image**: Input image with bounding boxes, labels, and confidence scores drawn on top
3. **Detection statistics**: Number of detections found above the specified threshold

## Error Handling

Both tools include robust error handling for:
- Missing bundle directories or files
- Unsupported model formats  
- Image loading failures (inference tool)
- Model loading errors
- Inference failures (inference tool)

Use `--verbose` flag for detailed error information and debugging.

## Files

- `inspect_model_bundle.py` - Model bundle inspection and analysis tool
- `run_inference_qat_model_exported.py` - TorchScript model inference tool  
- `requirements.txt` - Python dependencies
- `README.md` - This documentation
