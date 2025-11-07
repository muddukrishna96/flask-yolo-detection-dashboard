# Backend Module Architecture

## Overview
The backend has been refactored into a modular structure for easier maintenance, debugging, and testing.

## Module Breakdown

### 1. `backend/model_manager.py` (70 lines)
**Purpose:** Centralized model management
- `ALLOWED_MODELS` - Whitelist of permitted models
- `get_model(model_name)` - Load and cache YOLO models
- `preload_default_model()` - Preload default model on startup

**Benefits:**
- Single source of truth for model handling
- Easy to add/remove allowed models
- Cached instances prevent redundant loading

### 2. `backend/image_video_processor.py` (115 lines)
**Purpose:** File upload processing
- `process_image(filepath, model_name)` - Image detection
- `process_video(filepath, model_name)` - Video frame-by-frame processing
- `get_file_extension(filename)` - Utility function

**Benefits:**
- Isolated file processing logic
- Easy to test image/video pipelines separately
- Clean error handling boundaries

### 3. `backend/single_camera.py` (48 lines)
**Purpose:** Single webcam real-time detection
- `get_webcam_frame(model_name)` - Generator for MJPEG stream

**Benefits:**
- Focused single-camera logic
- Independent testing and debugging
- Easy to optimize frame rate

### 4. `backend/dual_camera.py` (90 lines)
**Purpose:** Dual webcam simultaneous detection
- `get_dual_webcam_frame(model0, model1)` - Parallel camera processing

**Benefits:**
- Isolated multi-camera complexity
- Easy to extend to N cameras
- Clear frame stacking logic

### 5. `backend/video_stream.py` (85 lines)
**Purpose:** Video streaming utilities
- `get_frame()` - Stream default output.mp4
- `stream_file_mp4_as_mjpeg(mp4_path)` - Stream any MP4
- `find_latest_video_folder()` - Helper for runs/detect
- `get_video_from_folder(folder)` - Locate video in subfolder

**Benefits:**
- Reusable streaming functions
- Clean separation of video I/O
- Easy to add new streaming formats

### 6. `webapp.py` (155 lines, down from 384!)
**Purpose:** Flask routes and request handling only
- Route definitions
- Request parameter extraction
- Response rendering

**Benefits:**
- 60% reduction in file size
- Clear separation: routes vs. processing
- Easy to add new endpoints

## Comparison

### Before Refactoring
- **1 file:** webapp.py (384 lines)
- All logic mixed together
- Hard to debug specific features
- Difficult to test in isolation

### After Refactoring
- **7 files:** webapp.py + 6 backend modules
- Clean separation of concerns
- Each module < 120 lines
- Easy to locate and fix bugs
- Modular testing possible

## Benefits of New Structure

1. **Maintainability:** Each module has a single, clear purpose
2. **Debuggability:** Issues isolated to specific modules
3. **Testability:** Can test each processor independently
4. **Scalability:** Easy to add new features (e.g., triple camera)
5. **Readability:** Small, focused files are easier to understand
6. **Reusability:** Backend modules can be imported by other apps

## How to Extend

### Adding a new model:
Edit `backend/model_manager.py` - Add to `ALLOWED_MODELS`

### Adding triple camera:
Create `backend/triple_camera.py` similar to dual_camera.py

### Adding new file type:
Extend `backend/image_video_processor.py` with new function

### Optimizing webcam FPS:
Edit only `backend/single_camera.py` or `backend/dual_camera.py`

## Migration Notes
- Original webapp.py backed up as `webapp_old.py`
- All functionality preserved
- No breaking changes to API or templates
- Same routes, same behavior, better structure
