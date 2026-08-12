# Implementation Details

This document provides comprehensive technical specifications for the Federated Generative AI Framework for Glaucoma Detection.

## Technology Stack

### Deep Learning Frameworks

| Framework | Version | Usage |
|-----------|---------|-------|
| **TensorFlow** | 2.8.0+ | Keras models, data augmentation |
| **PyTorch** | 1.12.0+ | Generative models, federated learning |
| **Torchvision** | 0.13.0+ | Pretrained models, transforms |

### Computer Vision Libraries

| Library | Version | Usage |
|---------|---------|-------|
| **OpenCV** | 4.6.0+ | Image preprocessing, quality assessment |
| **Albumentations** | 1.2.0+ | Advanced augmentation pipeline |
| **Scikit-image** | 0.19.0+ | Image quality metrics, filtering |

### Machine Learning Utilities

| Library | Version | Usage |
|---------|---------|-------|
| **Scikit-learn** | 1.1.0+ | Metrics, train/test split |
| **SciPy** | 1.8.0+ | Statistical tests, optimization |
| **NumPy** | 1.22.0+ | Numerical operations |
| **Pandas** | 1.4.0+ | Data management, CSV handling |

### Visualization

| Library | Version | Usage |
|---------|---------|-------|
| **Matplotlib** | 3.5.0+ | Plotting, visualizations |
| **Seaborn** | 0.11.0+ | Statistical plots |

### Jupyter Environment

| Package | Version | Usage |
|---------|---------|-------|
| **Jupyter** | 1.0.0+ | Notebook interface |
| **IPyKernel** | 6.15.0+ | Python kernel |

---

## Model Architectures

### 1. H-UQ-MFF (Hierarchical Uncertainty Quantification Multi-scale Feature Fusion)

**Backbone: ResNet-50**

```python
import torchvision.models as models

class HUQMFFEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)

        # Extract multi-scale feature layers
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # Output: 256×256×256
        self.layer2 = nn.Sequential(*list(resnet.children())[5])   # Output: 128×128×512
        self.layer3 = nn.Sequential(*list(resnet.children())[6])   # Output: 64×64×1024
        self.layer4 = nn.Sequential(*list(resnet.children())[7])   # Output: 32×32×2048

        # Channel attention modules
        self.channel_attn1 = ChannelAttention(256)
        self.channel_attn2 = ChannelAttention(512)
        self.channel_attn3 = ChannelAttention(1024)
        self.channel_attn4 = ChannelAttention(2048)

        # Spatial attention modules
        self.spatial_attn1 = SpatialAttention()
        self.spatial_attn2 = SpatialAttention()
        self.spatial_attn3 = SpatialAttention()
        self.spatial_attn4 = SpatialAttention()

        # Fusion layer
        self.fusion = HierarchicalFusion(in_channels=[256, 512, 1024, 2048],
                                        out_channels=512)
```

**Channel Attention**:
```python
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```

**Spatial Attention**:
```python
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                             padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)
```

---

### 2. StyleGAN2 Generator

**Architecture**:
```python
class StyleGAN2Generator(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, img_resolution=256):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim

        # Mapping network: Z → W
        self.mapping = MappingNetwork(z_dim, w_dim, num_layers=8)

        # Synthesis network: W → RGB
        self.synthesis = SynthesisNetwork(w_dim, img_resolution)

    def forward(self, z, truncation_psi=1.0):
        w = self.mapping(z)
        if truncation_psi < 1.0:
            w = self.truncate(w, truncation_psi)
        img = self.synthesis(w)
        return img
```

**Mapping Network**:
```python
class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = z_dim if i == 0 else w_dim
            layers.append(nn.Linear(in_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)
```

**Synthesis Network**:
```python
class SynthesisNetwork(nn.Module):
    def __init__(self, w_dim, img_resolution):
        super().__init__()
        self.num_layers = int(np.log2(img_resolution)) - 1

        # 4×4 → 8×8 → ... → 256×256
        self.blocks = nn.ModuleList()
        for res in [2**i for i in range(2, 2 + self.num_layers)]:
            self.blocks.append(SynthesisBlock(w_dim, res))

        self.to_rgb = nn.Conv2d(512, 3, kernel_size=1)

    def forward(self, w):
        x = self.blocks[0](None, w)
        for block in self.blocks[1:]:
            x = block(x, w)
        return torch.tanh(self.to_rgb(x))
```

---

### 3. Latent Diffusion Model

**U-Net Backbone**:
```python
class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, model_channels=128):
        super().__init__()
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )

        # Encoder
        self.down1 = ResBlock(in_channels, model_channels)
        self.down2 = ResBlock(model_channels, model_channels * 2)
        self.down3 = ResBlock(model_channels * 2, model_channels * 4)
        self.down4 = ResBlock(model_channels * 4, model_channels * 4)

        # Middle
        self.middle = AttentionBlock(model_channels * 4)

        # Decoder
        self.up4 = ResBlock(model_channels * 8, model_channels * 4)
        self.up3 = ResBlock(model_channels * 6, model_channels * 2)
        self.up2 = ResBlock(model_channels * 3, model_channels)
        self.up1 = ResBlock(model_channels * 2, out_channels)
```

**Noise Schedule**:
```python
def linear_beta_schedule(timesteps=1000):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def get_alphas(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas, alphas_cumprod
```

---

### 4. Vision-Language Fusion

**Vision Projection Head**:
```python
class VisionProjection(nn.Module):
    def __init__(self, in_dim=512, out_dim=768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.proj(x)
```

**QLoRA Configuration**:
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                # Rank
    lora_alpha=16,      # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Apply to Q,V attention
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply to MedGemma-4B
model = get_peft_model(medgemma_model, lora_config)
```

---

## Training Configuration

### Hyperparameters

#### Federated Learning

```python
FEDERATED_CONFIG = {
    # Global settings
    'num_clients': 3,
    'num_rounds': 50,
    'client_fraction': 1.0,  # All clients participate each round

    # Local training
    'local_epochs': 10,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'momentum': 0.9,

    # FedProx specific
    'mu': 0.01,  # Proximal term coefficient

    # Optimization
    'optimizer': 'Adam',
    'scheduler': 'CosineAnnealingLR',
    'warmup_rounds': 5,
}
```

#### Generative Models

```python
STYLEGAN2_CONFIG = {
    'z_dim': 512,
    'w_dim': 512,
    'img_resolution': 256,
    'num_iterations': 100000,
    'batch_size': 16,
    'learning_rate': 0.002,
    'beta1': 0.0,
    'beta2': 0.99,
    'r1_gamma': 10.0,  # R1 regularization weight
    'style_mixing_prob': 0.9,
}

DIFFUSION_CONFIG = {
    'timesteps': 1000,
    'beta_schedule': 'linear',
    'beta_start': 1e-4,
    'beta_end': 0.02,
    'img_size': 512,
    'latent_size': 64,
    'learning_rate': 1e-4,
    'batch_size': 8,
    'num_epochs': 100,
}
```

#### Vision-Language Fusion

```python
MULTIMODAL_CONFIG = {
    'vision_dim': 512,
    'language_dim': 768,
    'fusion_dim': 768,
    'dropout': 0.2,
    'learning_rate': 5e-5,
    'batch_size': 16,
    'num_epochs': 20,
    'warmup_steps': 500,
}
```

---

## Data Preprocessing

### Image Preprocessing Pipeline

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size=512):
    return A.Compose([
        # Resize
        A.Resize(img_size, img_size),

        # CLAHE enhancement
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.8),

        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, p=0.7),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                          rotate_limit=15, p=0.5),

        # Color augmentations
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10,
                            sat_shift_limit=20,
                            val_shift_limit=10, p=0.3),

        # Noise
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),

        # Normalization (ImageNet stats)
        A.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),

        ToTensorV2(),
    ])

def get_test_transforms(img_size=512):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
```

### Optic Disc Cropping

```python
import cv2

def crop_optic_disc(image, crop_size=256):
    """Detect and crop optic disc region (brightest area)"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Find brightest region (optic disc)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blurred)

    # Crop around optic disc
    x, y = maxLoc
    x1 = max(0, x - crop_size // 2)
    y1 = max(0, y - crop_size // 2)
    x2 = min(image.shape[1], x1 + crop_size)
    y2 = min(image.shape[0], y1 + crop_size)

    return image[y1:y2, x1:x2]
```

---

## Federated Learning Implementation

### FedAvg Algorithm

```python
class FedAvgServer:
    def __init__(self, model, clients):
        self.global_model = model
        self.clients = clients

    def train(self, num_rounds):
        for round_num in range(num_rounds):
            # Distribute global model
            for client in self.clients:
                client.set_model(self.global_model)

            # Local training
            client_weights = []
            client_sizes = []
            for client in self.clients:
                weights = client.train(epochs=10)
                client_weights.append(weights)
                client_sizes.append(len(client.dataset))

            # Aggregate
            self.global_model = self.aggregate(client_weights, client_sizes)

            # Evaluate
            acc = self.evaluate()
            print(f"Round {round_num}: Accuracy = {acc:.2%}")

    def aggregate(self, client_weights, client_sizes):
        """Weighted averaging"""
        total_size = sum(client_sizes)
        global_weights = {}

        for key in client_weights[0].keys():
            global_weights[key] = sum([
                (size / total_size) * weights[key]
                for weights, size in zip(client_weights, client_sizes)
            ])

        return global_weights
```

### FedProx Algorithm

```python
class FedProxClient:
    def __init__(self, model, data_loader, device, mu=0.01):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.mu = mu  # Proximal term coefficient
        self.global_weights = None

    def train(self, epochs):
        self.global_weights = copy.deepcopy(self.model.state_dict())
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

        for epoch in range(epochs):
            for batch in self.data_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)

                # Add proximal term
                proximal_term = 0.0
                for name, param in self.model.named_parameters():
                    proximal_term += ((param - self.global_weights[name]) ** 2).sum()
                loss += (self.mu / 2) * proximal_term

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.model.state_dict()
```

---

## Uncertainty Quantification

### Monte Carlo Dropout

```python
def mc_dropout_predict(model, input, num_samples=50):
    """Perform MC Dropout inference"""
    model.train()  # Enable dropout at inference

    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            output = model(input)
            predictions.append(torch.softmax(output, dim=1))

    predictions = torch.stack(predictions)

    # Compute statistics
    mean_pred = predictions.mean(dim=0)
    uncertainty = predictions.var(dim=0)

    return mean_pred, uncertainty
```

### Calibration Metrics

```python
def expected_calibration_error(predictions, confidences, labels, num_bins=10):
    """Compute ECE for calibration"""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)

    return ece
```

---

## Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

def compute_metrics(y_true, y_pred, y_prob):
    """Compute all classification metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'specificity': recall_score(y_true, y_pred, pos_label=0),
        'auc_roc': roc_auc_score(y_true, y_prob),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }
```

### Generative Quality Metrics

```python
from scipy.linalg import sqrtm
import lpips

def calculate_fid(real_features, fake_features):
    """Fréchet Inception Distance"""
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # Compute FID
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real @ sigma_fake)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid

def calculate_lpips(real_images, fake_images):
    """Learned Perceptual Image Patch Similarity"""
    lpips_model = lpips.LPIPS(net='alex')
    distances = []

    for real, fake in zip(real_images, fake_images):
        distance = lpips_model(real, fake)
        distances.append(distance.item())

    return np.mean(distances)
```

---

## Computational Requirements

### GPU Memory Footprint

| Component | Memory (GB) | Notes |
|-----------|------------|-------|
| ResNet-50 backbone | 0.5 | Pretrained weights |
| H-UQ-MFF features | 2.5 | Multi-scale intermediate activations |
| StyleGAN2 generator | 3.8 | Mapping + synthesis networks |
| Diffusion U-Net | 4.2 | Encoder + decoder |
| MedGemma-4B | 8.5 | Transformer with QLoRA |
| Batch data (8 images) | 1.2 | 512×512×3 images |
| **Total Peak** | **15.2 GB** | During multi-modal training |

**Recommendation**: NVIDIA V100 (16GB), A100 (40GB), or RTX 3090 (24GB)

### Training Time Estimates

Based on NVIDIA V100 GPU:

| Phase | Time | Parallelizable |
|-------|------|----------------|
| Data preprocessing | 30 min | ✅ CPU parallel |
| StyleGAN2 training | 2.5 hours | ✅ Multi-GPU |
| Diffusion training | 2.1 hours | ✅ Multi-GPU |
| Federated training (50 rounds) | 4.5 hours | ✅ Client parallel |
| Multi-modal fusion | 1.8 hours | ❌ Single GPU |
| Evaluation | 20 min | ✅ CPU parallel |

---

## Code Structure

The Jupyter notebook (`federated_generative_framework.ipynb`) is organized into 62 cells:

```
Cells 1-5:   Data Loading & Augmentation
             - Dataset class definitions
             - Data generators
             - Train/val/test splits

Cells 6-10:  Image Preprocessing
             - CLAHE enhancement
             - Optic disc detection
             - Quality filtering

Cells 11-15: Federated Feature Learning
             - H-UQ-MFF architecture
             - Multi-scale feature extraction
             - Attention mechanisms

Cells 16-30: Federated Learning
             - Client initialization
             - FedAvg/FedProx implementation
             - Aggregation logic
             - Training loop

Cells 31-35: Generative AI Module
             - StyleGAN2 training
             - Diffusion model training
             - Synthetic image generation
             - Quality metric computation

Cells 36-40: Vision-Language Learning
             - MedGemma loading
             - QLoRA configuration
             - Projection head training
             - Multimodal fusion

Cells 41-45: Clinical Report Generation
             - MC Dropout inference
             - Uncertainty quantification
             - Template-based reports
             - LLM-generated explanations

Cells 46-50: Cross-Dataset Validation
             - Train/test splits across datasets
             - Generalization metrics

Cells 51-54: Expert Evaluation
             - Agreement metrics
             - Disagreement analysis

Cells 55-58: Model Compression
             - GGUF quantization
             - Size/accuracy tradeoff

Cells 59-62: Final Testing
             - Comprehensive evaluation
             - Confusion matrix
             - Performance summary
```

---

## Reproducibility

### Random Seed Setting

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Environment Export

```bash
# Export exact package versions
pip freeze > requirements_exact.txt

# Export conda environment
conda env export > environment.yml
```

---

## Deployment Considerations

### Model Export

```python
# PyTorch to ONNX
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=['input'], output_names=['output'])

# TensorFlow SavedModel
model.save('saved_model/')

# GGUF Quantization
!python convert_to_gguf.py --model model.pt --output model.gguf --quant q8_0
```

### API Endpoint

```python
from fastapi import FastAPI, File, UploadFile
import torch

app = FastAPI()
model = torch.load("federated_model.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = load_image(await file.read())
    preprocessed = preprocess(image)

    # MC Dropout inference
    mean_pred, uncertainty = mc_dropout_predict(model, preprocessed, num_samples=50)

    return {
        "diagnosis": "glaucoma" if mean_pred > 0.5 else "normal",
        "probability": float(mean_pred),
        "confidence": float(1 - uncertainty),
        "recommendation": get_recommendation(mean_pred, uncertainty)
    }
```

---

## Troubleshooting

### Common Issues

**Issue**: CUDA Out of Memory
```python
# Solution: Reduce batch size
BATCH_SIZE = 4  # Instead of 8

# Or use gradient accumulation
accumulation_steps = 2
for i, batch in enumerate(data_loader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Issue**: Slow Training
```python
# Solution: Use mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in data_loader:
    with autocast():
        output = model(batch)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

For usage instructions, see [USAGE.md](USAGE.md).

For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md).
