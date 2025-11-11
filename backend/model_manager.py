"""
Model management and caching for YOLO models.
Handles model loading, validation, and caching.
"""

import os
from ultralytics import YOLO


# Model cache: stores loaded models by filename to avoid reloading
model_cache = {}

# Keep track of custom uploaded model names (registered at runtime)
custom_model_names = set()

# List of allowed models for security (updated per user request)
ALLOWED_MODELS = {
    'yolov8n.pt',
    'yolov9t.pt',
    'yolov10n.pt',
    'yolo11n.pt',
    'yolo12n.pt',
    'yolov8n-oiv7.pt'
}


def get_model(model_name: str):
    """
    Load and cache YOLO models.
    
    Args:
        model_name: Name of the model file (e.g., 'yolov8n.pt')
        
    Returns:
        YOLO model instance
        
    Raises:
        ValueError: If model name is not in allowed list
    """
    if not model_name:
        return None
    
    # Return cached model if already loaded (includes uploaded models)
    if model_name in model_cache:
        return model_cache[model_name]

    # Validate model name against whitelist for built-in models
    if model_name not in ALLOWED_MODELS:
        raise ValueError(f'Model not allowed. Choose from built-ins or upload a custom model.')

    # Check for local models first to avoid re-downloading
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)

    local_model_path = os.path.join(models_dir, model_name)
    model_to_load = None

    # Load from local path if exists, otherwise let YOLO auto-download
    if os.path.isfile(local_model_path):
        model_to_load = local_model_path
    else:
        model_to_load = model_name

    # Instantiate and cache the model
    m = YOLO(model_to_load)
    model_cache[model_name] = m
    return m


def register_custom_model(name: str, model_instance) -> None:
    """
    Register a custom-loaded model instance in memory so it can be selected by name.

    Args:
        name: Display name to register (will be used in dropdowns)
        model_instance: Loaded model object (e.g., YOLO instance)
    """
    # Normalize name
    display = name.strip()
    model_cache[display] = model_instance
    custom_model_names.add(display)


def list_models():
    """Return a sorted list of available model names (built-ins + uploaded)."""
    built = sorted(list(ALLOWED_MODELS))
    customs = sorted(list(custom_model_names))
    return built + customs


def preload_default_model():
    """Preload the default model into cache for faster first request."""
    try:
        get_model('yolov8n.pt')
    except Exception:
        pass
