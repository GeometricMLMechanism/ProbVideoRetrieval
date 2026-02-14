# ImageBind Setup & Debugging

## Issue: ModuleNotFoundError in ImageBind Server

### Root Causes
The `Imagebind_text_encoder_server.py` failed to start due to multiple version incompatibility issues:

1. **PyTorch/torchvision version mismatch**: The environment had `torch==2.8.0` and `torchvision==0.23.0`, but `pytorchvideo` (pinned to a specific commit) requires older versions that include the `torchvision.transforms.functional_tensor` module, which was removed in recent versions.

2. **NumPy 2.x incompatibility**: PyTorch 1.13.0 was compiled with NumPy 1.x and crashes when NumPy 2.0.2 is installed.

3. **Import path resolution**: The server script's sys.path setup didn't properly handle module imports when launched via `conda run`.

### Solutions Applied

#### 1. Downgraded PyTorch Stack
Installed compatible versions with CUDA 11.7 support:
```bash
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --index-url https://download.pytorch.org/whl/cu117
```

#### 2. Downgraded NumPy
```bash
pip install 'numpy<2'
```
Installed `numpy==1.26.4` to maintain compatibility with PyTorch 1.13.0.

#### 3. Fixed Import Paths in Imagebind_text_encoder_server.py
Updated sys.path setup to use absolute paths and correct import statements:
- Changed `sys.path.append()` to `sys.path.insert(0, ...)` for priority
- Used `os.path.abspath(__file__)` for reliable path resolution
- Updated imports from `ImageBind.models` to `models` to work with the corrected sys.path

### Result
The ImageBind text encoder server now starts successfully. Deprecation warnings from torchvision about moved modules are expected and harmless.

### Environment Details
- **imagebind environment**: Python 3.9 with PyTorch 1.13.0+cu117, torchvision 0.14.0, NumPy <2
- **pytorchvideo**: Installed from GitHub commit 28fe037 (specific version required by ImageBind)
