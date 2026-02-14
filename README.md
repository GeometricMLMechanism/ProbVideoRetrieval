# Probabilistic Video Retrieval - Environment Setup Guide

This guide provides step-by-step instructions for setting up all required conda environments for the Multi-Agent Video Retrieval system on a completely new machine.

## Overview

This project requires **three separate conda environments**:

1. **`multi_agent_retrieval`** - Main environment for the retrieval system
2. **`lavis`** - Environment for BLIP2 text encoder server
3. **`imagebind`** - Environment for ImageBind text encoder server

---

## Prerequisites

- CUDA-capable GPU (CUDA 11.7 or compatible)
- Conda or Miniconda installed
- Git installed
- Sufficient disk space (~20GB for all environments and models)

---

## 1. Main Environment: `multi_agent_retrieval`

This is the primary environment for running the multi-agent retrieval system.

### Create and Activate Environment

```bash
conda create -n multi_agent_retrieval python=3.9 -y
conda activate multi_agent_retrieval
```

### Install PyTorch with CUDA Support

```bash
# Install PyTorch 2.0+ with CUDA 11.7 (or adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

### Install Core Dependencies

```bash
# Transformers and vision-language models
pip install transformers==4.45.0
pip install qwen-vl-utils
pip install accelerate
pip install flash-attn --no-build-isolation

# Video processing and retrieval
pip install opencv-python
pip install decord
pip install einops
pip install ftfy
pip install timm

# Data processing and evaluation
pip install numpy pandas scipy scikit-learn
pip install h5py
pip install tqdm

# Experiment tracking
pip install wandb

# CLIP model
pip install git+https://github.com/openai/CLIP.git
```

### Install Project Submodules

The project includes several submodules that need to be set up:

#### InternVid (ViCLIP)
```bash
# Already included in the repository
# Ensure viclip model checkpoint is downloaded to tool_models/viCLIP/
```

#### IITV (Improved Image-Text-Video)
```bash
# Already included in the repository
# Ensure model checkpoint is in IITV/checkpoints/
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "from qwen_vl_utils import process_vision_info; print('Qwen VL Utils: OK')"
python -c "import clip; print('CLIP: OK')"
```

---

## 2. LAVIS Environment: `lavis`

This environment runs the BLIP2 text encoder server.

### Create and Activate Environment

```bash
conda create -n lavis python=3.9 -y
conda activate lavis
```

### Install PyTorch (Same as Main Environment)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

### Install LAVIS and Dependencies

```bash
# Navigate to LAVIS directory
cd LAVIS

# Install LAVIS in development mode
pip install -e .

# Install additional requirements
pip install -r requirements.txt

# Key dependencies (if not installed by requirements.txt)
pip install transformers==4.33.2
pip install timm==0.4.12
pip install opencv-python-headless==4.5.5.64
pip install decord
pip install einops
pip install fairscale==0.4.4
pip install omegaconf
pip install ftfy
pip install sentencepiece

cd ..
```

### Verify LAVIS Installation

```bash
python -c "from lavis.models import load_model_and_preprocess; print('LAVIS: OK')"
```

### Test BLIP2 Server

```bash
# Create tmp directory if it doesn't exist
mkdir -p tmp

# Test the BLIP2 server (it should start without errors)
python BLIP2_text_encoder_server.py --device=cuda:0 --BLIP2_server=tmp/BLIP2_test.sock --BLIP2_feature_file=tmp/BLIP2_test.npy
# Press Ctrl+C to stop after seeing "✅ BLIP2 server ready for connections!"
```

---

## 3. ImageBind Environment: `imagebind`

This environment runs the ImageBind text encoder server.

### Create and Activate Environment

```bash
conda create -n imagebind python=3.9 -y
conda activate imagebind
```

### Install PyTorch 1.13.0 (Required for ImageBind)

**IMPORTANT:** ImageBind requires PyTorch 1.13.0 due to dependencies on `pytorchvideo` and `torchvision.transforms.functional_tensor`.

```bash
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --index-url https://download.pytorch.org/whl/cu117
```

### Install NumPy <2.0 (Critical for PyTorch 1.13.0)

```bash
pip install 'numpy<2'
```

**Why NumPy <2?** PyTorch 1.13.0 was compiled against NumPy 1.x and will crash with NumPy 2.x.

### Install ImageBind Dependencies

```bash
# Install pytorchvideo from specific commit (required by ImageBind)
pip install git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d

# Install other dependencies
pip install timm==0.6.7
pip install ftfy
pip install regex
pip install einops
pip install fvcore
pip install decord==0.6.0
pip install iopath
pip install matplotlib
```

### Verify ImageBind Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import sys; sys.path.insert(0, 'ImageBind'); from models import imagebind_model; print('ImageBind: OK')"
```

### Test ImageBind Server

```bash
# Create tmp directory if it doesn't exist
mkdir -p tmp

# Test the ImageBind server (it should start without errors)
python Imagebind_text_encoder_server.py --device=cuda:0 --ImageBind_server=tmp/ImageBind_test.sock --ImageBind_feature_file=tmp/ImageBind_test.npy
# Press Ctrl+C to stop after seeing "✅ ImageBind server ready for connections!"
```

---

## Common Issues and Solutions

### Issue 1: ModuleNotFoundError in ImageBind Server

**Problem:** `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`

**Solution:** This occurs when PyTorch/torchvision versions are too new. Ensure you're using PyTorch 1.13.0 and torchvision 0.14.0 in the `imagebind` environment:

```bash
conda activate imagebind
pip uninstall torch torchvision torchaudio -y
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --index-url https://download.pytorch.org/whl/cu117
```

### Issue 2: NumPy 2.x Incompatibility

**Problem:** Crashes or errors related to NumPy when using PyTorch 1.13.0

**Solution:** Downgrade NumPy to version <2.0:

```bash
conda activate imagebind
pip install 'numpy<2'
```

### Issue 3: CUDA Out of Memory

**Problem:** GPU runs out of memory when running multiple models

**Solution:** Distribute models across multiple GPUs using the device arguments:

```bash
# Example: Run main model on cuda:0, servers on cuda:1
python main.py --vlm_device=cuda:0 --search_device=cuda:0 --server_device=cuda:1
```

### Issue 4: Flash Attention Installation Fails

**Problem:** `flash-attn` fails to install in main environment

**Solution:** Try installing without build isolation or use a pre-built wheel:

```bash
pip install flash-attn --no-build-isolation
# OR
pip install flash-attn==2.5.0  # Use a specific version
```

### Issue 5: Transformers Version Conflicts

**Problem:** Different transformers versions required by different components

**Solution:**
- Main environment: Use `transformers==4.45.0` (for Qwen models)
- LAVIS environment: Use `transformers==4.33.2` (for BLIP2)

---

## Running the System

### 1. Start the Servers (Optional - if using IITV search model)

The servers are automatically started by the main script when using `search_model_name='IITV'`. However, you can start them manually for testing:

**Terminal 1 - BLIP2 Server:**
```bash
conda activate lavis
python BLIP2_text_encoder_server.py --device=cuda:1 --BLIP2_server=tmp/BLIP2_ViSA.sock --BLIP2_feature_file=tmp/BLIP2_ViSA_feature.npy
```

**Terminal 2 - ImageBind Server:**
```bash
conda activate imagebind
python Imagebind_text_encoder_server.py --device=cuda:1 --ImageBind_server=tmp/ImageBind_ViSA.sock --ImageBind_feature_file=tmp/ImageBind_ViSA_feature.npy
```

### 2. Run the Main Retrieval System

**Terminal 3 - Main System:**
```bash
conda activate multi_agent_retrieval
python main.py \
    --MLLM_model_id=/path/to/Qwen2.5-VL-7B-Instruct/ \
    --search_model_name=viclip \
    --featurename=viclip_vid_feature \
    --rootpath=/path/to/data/ \
    --eval_k=1000 \
    --examine_number=20 \
    --vlm_device=cuda:0 \
    --search_device=cuda:0 \
    --server_device=cuda:1
```

---

## Environment Summary

| Environment | Python | PyTorch | Key Packages | Purpose |
|-------------|--------|---------|--------------|---------|
| `multi_agent_retrieval` | 3.9 | 2.0+ | transformers 4.45.0, qwen-vl-utils, clip, wandb | Main retrieval system |
| `lavis` | 3.9 | 1.10+ | lavis, transformers 4.33.2, timm 0.4.12 | BLIP2 text encoder server |
| `imagebind` | 3.9 | 1.13.0 | pytorchvideo, numpy<2, timm 0.6.7 | ImageBind text encoder server |

---

## Model Checkpoints

Ensure the following model checkpoints are downloaded:

1. **Qwen2.5-VL-7B-Instruct** or **Qwen3-VL-8B-Instruct**
   - Download from HuggingFace: `Qwen/Qwen2.5-VL-7B-Instruct`
   
2. **ViCLIP Model**
   - Place in: `tool_models/viCLIP/ViClip-InternVid-10M-FLT.pth`
   
3. **IITV Model**
   - Place in: `IITV/checkpoints/model_best.pth.match.tar`

4. **ImageBind Model**
   - Automatically downloaded on first use to `~/.cache/`

5. **BLIP2 Model**
   - Automatically downloaded on first use via LAVIS

---

## Quick Start Checklist

- [ ] Install conda/miniconda
- [ ] Create `multi_agent_retrieval` environment
- [ ] Create `lavis` environment
- [ ] Create `imagebind` environment
- [ ] Download model checkpoints
- [ ] Prepare video dataset and features
- [ ] Test BLIP2 server (if using IITV)
- [ ] Test ImageBind server (if using IITV)
- [ ] Run main retrieval system

---

## Additional Resources

- **InternVid/ViCLIP:** [GitHub](https://github.com/OpenGVLab/InternVideo/tree/main/InternVid)
- **LAVIS/BLIP2:** [GitHub](https://github.com/salesforce/LAVIS)
- **ImageBind:** [GitHub](https://github.com/facebookresearch/ImageBind)
- **Qwen-VL:** [HuggingFace](https://huggingface.co/Qwen)

---

## Troubleshooting Tips

1. **Always activate the correct environment** before running scripts
2. **Check CUDA availability:** `python -c "import torch; print(torch.cuda.is_available())"`
3. **Monitor GPU memory:** `nvidia-smi -l 1`
4. **Check server logs** if connection fails
5. **Ensure socket files are cleaned up** between runs: `rm tmp/*.sock`

---

## Contact & Support

For issues related to:
- **Environment setup:** Refer to this README and the debug notes in `README2.md`
- **Model-specific issues:** Check the respective model repositories
- **CUDA/GPU issues:** Verify CUDA installation and driver compatibility

---

**Last Updated:** February 2026
