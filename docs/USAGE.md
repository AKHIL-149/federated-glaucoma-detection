# Usage Guide

This guide explains how to run the Federated Generative AI Framework for Glaucoma Detection.

## Overview

The framework is implemented as a comprehensive Jupyter notebook with 62 cells organized into 11 distinct phases. Each phase represents a critical component of the federated learning pipeline.

## Before You Begin

Ensure you have completed:
1. Installation (see [INSTALLATION.md](INSTALLATION.md))
2. Dataset download and organization
3. Virtual environment activation

## Running the Framework

### Step 1: Launch Jupyter Notebook

```bash
cd federated-glaucoma-detection
source venv/bin/activate  # Activate virtual environment
jupyter notebook notebooks/federated_generative_framework.ipynb
```

### Step 2: Configure Paths

In the first few cells of the notebook, update the dataset paths to match your local setup:

```python
# Example configuration
DATA_PATHS = {
    'eyepacs': '/path/to/EyePACS/',
    'refuge': '/path/to/REFUGE2/',
    'papila': '/path/to/PAPILA/'
}
```

### Step 3: Execute Cells Sequentially

Run the notebook cells in order. The framework is organized into 11 phases:

## Phase-by-Phase Guide

### Phase 1: Data Loading & Augmentation (Cells 1-5)

**Purpose**: Load retinal fundus images and set up data generators

**What happens**:
- Loads images from three datasets (EyePACS, REFUGE2, PAPILA)
- Creates train/validation/test splits
- Applies basic augmentation (flips, rotations, brightness adjustments)

**Expected output**:
```
Found 15000 images from EyePACS
Found 1200 images from REFUGE2
Found 800 images from PAPILA
Train: 12000, Val: 3000, Test: 2000
```

**Runtime**: ~5-10 minutes

---

### Phase 2: Image Preprocessing (Cells 6-10)

**Purpose**: Enhance image quality and standardize inputs

**What happens**:
- Resizes images to 512×512 pixels
- Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Crops optic disc region (brightest area detection)
- Normalizes pixel values using ImageNet statistics

**Expected output**:
- Preprocessed images saved to `outputs/preprocessed_outputs/`
- Quality score distribution histogram

**Runtime**: ~15-30 minutes (depends on dataset size)

---

### Phase 3: Federated Feature Learning (Cells 11-15)

**Purpose**: Extract multi-scale visual features using H-UQ-MFF architecture

**What happens**:
- Initializes ResNet-50 backbone (pretrained on ImageNet)
- Implements multi-scale feature extraction (layers 1-4)
- Applies channel and spatial attention mechanisms
- Fuses features hierarchically into 512-dimensional vectors

**Expected output**:
```
Feature extraction complete
Feature shape: (N, 512)
Attention weights computed
```

**Runtime**: ~20-40 minutes (GPU recommended)

---

### Phase 4: Federated Learning (Cells 16-30)

**Purpose**: Train models across distributed datasets using federated algorithms

**What happens**:
1. Initializes 3 federated clients (one per dataset)
2. For each round (10 rounds total):
   - Clients download global model
   - Clients train locally for 5 epochs
   - Clients upload model updates
   - Server aggregates updates using FedAvg or FedProx
3. Tracks convergence and performance metrics

**Expected output**:
```
Round 1/10: Avg Loss: 0.65, Avg Accuracy: 62.3%
Round 2/10: Avg Loss: 0.52, Avg Accuracy: 68.7%
...
Round 10/10: Avg Loss: 0.28, Avg Accuracy: 82.5%
Convergence achieved at round 9
```

**Runtime**: ~4-6 hours (most time-intensive phase)

**Outputs saved**:
- Training curves: `outputs/federated_learning/training_curves_fedavg.png`
- Algorithm comparison: `outputs/federated_learning/algorithm_comparison.png`
- Model checkpoints: `saved_models/federated_model_round_*.pt` (not tracked in Git)

---

### Phase 5: Generative AI Module (Cells 31-35)

**Purpose**: Generate synthetic fundus images for data augmentation

**What happens**:
- Trains StyleGAN2 generator (256×256 resolution, 100k iterations)
- Trains diffusion model (1000 timesteps, U-Net backbone)
- Generates synthetic samples
- Computes quality metrics (FID, SSIM, PSNR, LPIPS)

**Expected output**:
```
StyleGAN2 training: Iteration 100000/100000
FID Score: 2.2e-05 (excellent quality)
Generated 500 synthetic glaucoma samples
Generated 500 synthetic normal samples
```

**Runtime**: ~2-3 hours (GPU strongly recommended)

**Outputs saved**:
- Synthetic samples: `outputs/generative_results/stylegan2_samples.png`
- Quality metrics: `outputs/generative_results/quality_metrics.png`

---

### Phase 6: Vision-Language Learning (Cells 36-40)

**Purpose**: Align visual features with clinical language descriptions

**What happens**:
- Initializes MedGemma-4B language model
- Projects visual features into language embedding space
- Fine-tunes using QLoRA (parameter-efficient)
- Learns vision-language correspondences

**Expected output**:
```
MedGemma-4B loaded (4.2B parameters)
QLoRA fine-tuning: 176,770 trainable params (34.94%)
Epoch 1/20: Loss 0.156
...
Epoch 20/20: Loss 0.023
```

**Runtime**: ~1-2 hours

---

### Phase 7: Clinical Report Generation (Cells 41-45)

**Purpose**: Generate automated clinical explanations with uncertainty

**What happens**:
- For each test image:
  - Extracts visual features
  - Runs 50 Monte Carlo Dropout passes
  - Computes mean prediction and uncertainty
  - Generates clinical text explanation
- Outputs structured reports

**Expected output**:
```
Processing test images: 100%
Generated 2000 clinical reports
Average confidence: 87.3%
High-risk predictions flagged: 45
```

**Outputs saved**:
- Reports CSV: `outputs/clinical_explanations/EyePACS_clinical_explanations.csv`
- Sample explanations: `outputs/clinical_explanations/EyePACS_explanation_samples.txt`
- Confidence distribution: `outputs/clinical_explanations/EyePACS_confidence_distribution.png`

**Runtime**: ~1 hour

---

### Phase 8: Cross-Dataset Validation (Cells 46-50)

**Purpose**: Test model generalization on external datasets

**What happens**:
- Tests model trained on EyePACS against REFUGE2 and PAPILA
- Tests model trained on REFUGE2 against EyePACS and PAPILA
- Computes cross-dataset performance metrics

**Expected output**:
```
Train: EyePACS → Test: REFUGE2  | Accuracy: 78.5%
Train: EyePACS → Test: PAPILA   | Accuracy: 76.2%
Train: REFUGE2 → Test: EyePACS  | Accuracy: 80.1%
```

**Runtime**: ~30-45 minutes

---

### Phase 9: Expert Evaluation (Cells 51-54)

**Purpose**: Validate predictions against ophthalmologist assessments

**What happens**:
- Compares AI predictions with expert annotations
- Computes agreement metrics (Cohen's kappa)
- Identifies cases where AI and experts disagree

**Expected output**:
```
Expert agreement: 89.3%
Cohen's kappa: 0.82 (substantial agreement)
Disagreement cases: 214 (10.7%)
```

**Runtime**: ~15 minutes

---

### Phase 10: Model Compression (Cells 55-58)

**Purpose**: Quantize models for efficient deployment

**What happens**:
- Converts models to GGUF format
- Applies 8-bit quantization
- Validates post-quantization accuracy

**Expected output**:
```
Original model size: 95 MB
Quantized model size: 24 MB (74% reduction)
Accuracy drop: 0.3% (acceptable)
```

**Runtime**: ~20 minutes

---

### Phase 11: Final Testing (Cells 59-62)

**Purpose**: Comprehensive performance evaluation

**What happens**:
- Tests on holdout test set
- Generates confusion matrix
- Computes all classification metrics
- Creates final performance report

**Expected output**:
```
Final Test Results:
- Accuracy: 84.8%
- Precision: 84.69%
- Recall: 85.69%
- F1-Score: 85.19%
- AUC-ROC: 92.4%
```

**Outputs saved**:
- Confusion matrix: `outputs/multimodal_fusion/confusion_matrix.png`
- Classification report: `outputs/validation_results/final_metrics.json`

**Runtime**: ~20 minutes

---

## Total Expected Runtime

| Phase | Runtime (GPU) | Runtime (CPU) |
|-------|---------------|---------------|
| Phases 1-3 | ~1 hour | ~2-3 hours |
| Phase 4 (Federated) | ~4-6 hours | ~12-18 hours |
| Phase 5 (Generative) | ~2-3 hours | ~8-12 hours |
| Phases 6-11 | ~2-3 hours | ~4-6 hours |
| **Total** | **~9-13 hours** | **~26-39 hours** |

**Recommendation**: Use GPU for Phases 4-5. CPU is sufficient for other phases.

## Running Specific Phases

You don't need to run all phases every time. To run specific components:

### Train Only Federated Learning
```python
# Run cells 1-5 (data loading)
# Run cells 16-30 (federated training)
```

### Generate Only Synthetic Images
```python
# Run cells 1-5 (data loading)
# Run cells 31-35 (generative models)
```

### Evaluate Pretrained Model
```python
# Load pretrained weights (not in repo, available on request)
model.load_state_dict(torch.load('pretrained_federated_model.pt'))
# Run cells 59-62 (final testing)
```

## Output Files

All outputs are saved to the `outputs/` directory:

```
outputs/
├── federated_learning/        # Training curves, convergence plots
├── multimodal_fusion/         # Confusion matrices, performance metrics
├── clinical_explanations/     # Generated reports and confidence scores
├── generative_results/        # Synthetic images and quality metrics
├── preprocessed_outputs/      # Preprocessed images (not tracked in Git)
├── saved_models/              # Model checkpoints (not tracked in Git)
└── validation_results/        # Cross-validation metrics
```

## Monitoring Progress

During execution, you can monitor:

1. **GPU utilization**: `nvidia-smi` (in separate terminal)
2. **Memory usage**: `htop` or Activity Monitor
3. **Cell outputs**: Progress bars using tqdm
4. **TensorBoard**: Launch with `tensorboard --logdir=outputs/logs`

## Troubleshooting

### Out of Memory Errors

**Solution**: Reduce batch size or image resolution:
```python
BATCH_SIZE = 4  # Instead of 8
IMG_SIZE = 256  # Instead of 512
```

### Long Training Times

**Solution**:
- Reduce number of federated rounds: `ROUNDS = 5`
- Reduce local epochs: `LOCAL_EPOCHS = 3`
- Skip generative training (Phases 5) if not needed

### Kernel Crashes

**Solution**: Restart kernel and run cells sequentially (don't run all at once):
```
Kernel → Restart & Clear Output
Run cells one by one
```

## Best Practices

1. **Save checkpoints**: Enable checkpointing in Phase 4 to resume if interrupted
2. **Monitor logs**: Keep terminal open to see detailed logs
3. **Validate intermediate outputs**: Check outputs after each phase
4. **Use GPU**: Phases 4-5 are 3-5× faster on GPU
5. **Start small**: Test on subset of data first before full run

## Next Steps

After running the framework:
- Review results in [RESULTS.md](RESULTS.md)
- Understand architecture in [ARCHITECTURE.md](ARCHITECTURE.md)
- Explore technical details in [IMPLEMENTATION.md](IMPLEMENTATION.md)

## Support

For issues or questions:
- Check [INSTALLATION.md](INSTALLATION.md) for setup problems
- Review error messages in notebook outputs
- Open an issue in the GitHub repository
