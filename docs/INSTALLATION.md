# Installation Guide

This guide provides detailed instructions for setting up the Federated Generative AI Framework for Glaucoma Detection.

## Prerequisites

Before installation, ensure your system meets these requirements:

### Hardware Requirements
- **CPU**: 8+ cores recommended (Intel i7/i9 or AMD Ryzen 7/9)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better) for faster training
  - CUDA 11.x compatible
  - Optional but highly recommended for generative model training
- **Storage**: 50GB free disk space
  - 40GB for datasets
  - 10GB for outputs and model checkpoints

### Software Requirements
- **Operating System**: Linux, macOS, or Windows 10/11
- **Python**: Version 3.8, 3.9, 3.10, or 3.11
- **Git**: For cloning the repository
- **Jupyter**: Will be installed via requirements.txt

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/[username]/federated-glaucoma-detection.git
cd federated-glaucoma-detection
```

### 2. Create a Virtual Environment

Using Python's built-in venv:

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

Alternative using conda:

```bash
conda create -n glaucoma-fed python=3.10
conda activate glaucoma-fed
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- TensorFlow 2.8+ (deep learning framework)
- PyTorch 1.12+ (deep learning framework)
- OpenCV (computer vision)
- Albumentations (image augmentation)
- Scikit-learn (machine learning utilities)
- Jupyter (interactive notebooks)
- And other required packages

**Note**: Installation may take 5-15 minutes depending on your internet speed.

### 4. GPU Setup (Optional but Recommended)

If you have an NVIDIA GPU, verify CUDA installation:

```bash
# Check CUDA version
nvcc --version

# Test PyTorch GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test TensorFlow GPU
python -c "import tensorflow as tf; print(f'GPU devices: {tf.config.list_physical_devices(\"GPU\")}')"
```

If CUDA is not detected:
1. Install [NVIDIA CUDA Toolkit 11.x](https://developer.nvidia.com/cuda-toolkit)
2. Install [cuDNN](https://developer.nvidia.com/cudnn)
3. Reinstall PyTorch with CUDA support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### 5. Download Datasets

The framework requires three public glaucoma datasets. Links are provided in [data/dataset_links.txt](../data/dataset_links.txt):

1. **EyePACS-AIROGS-light-V2** (Kaggle)
   - Visit: https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2
   - Requires Kaggle account
   - Download size: ~15-20 GB

2. **REFUGE2** (Kaggle)
   - Visit: https://www.kaggle.com/datasets/victorlemosml/refuge2
   - Requires Kaggle account
   - Download size: ~10 GB

3. **PAPILA** (Figshare)
   - Visit: https://figshare.com/articles/dataset/PAPILA/14798004
   - Direct download available
   - Download size: ~8 GB

**Recommended dataset directory structure:**
```
/path/to/datasets/
├── EyePACS/
│   ├── train/
│   │   ├── glaucoma/
│   │   └── normal/
│   ├── val/
│   └── test/
├── REFUGE2/
│   ├── train/
│   ├── val/
│   └── test/
└── PAPILA/
    ├── train/
    ├── val/
    └── test/
```

### 6. Launch Jupyter Notebook

```bash
jupyter notebook notebooks/federated_generative_framework.ipynb
```

This will open the main notebook in your default web browser.

## Verification

After installation, verify everything is working:

```bash
# Check Python version
python --version

# Check installed packages
pip list | grep -E "tensorflow|torch|opencv"

# Launch Jupyter
jupyter notebook --version
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size in the notebook:
```python
BATCH_SIZE = 4  # Instead of 8
```

### Issue: Package conflicts

**Solution**: Create a fresh virtual environment and reinstall:
```bash
deactivate
rm -rf venv/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Jupyter kernel not found

**Solution**: Register the kernel manually:
```bash
python -m ipykernel install --user --name=glaucoma-fed
```

### Issue: Slow dataset download

**Solution**: Use Kaggle API for faster downloads:
```bash
pip install kaggle
# Configure API token (see Kaggle documentation)
kaggle datasets download deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2
```

## Next Steps

After successful installation:
1. Read [USAGE.md](USAGE.md) for instructions on running the framework
2. Review [ARCHITECTURE.md](ARCHITECTURE.md) to understand the system design
3. Run the first few cells of the notebook to verify dataset loading

## System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| GPU | Optional | NVIDIA RTX 3070+ (8GB VRAM) |
| Storage | 50 GB | 100 GB |
| Python | 3.8 | 3.10 |
| OS | Any | Linux/Ubuntu |

## Additional Resources

- [TensorFlow Installation Guide](https://www.tensorflow.org/install)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [Jupyter Documentation](https://jupyter.org/install)
