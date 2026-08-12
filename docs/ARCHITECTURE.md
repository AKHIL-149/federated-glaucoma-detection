# System Architecture

This document describes the technical architecture of the Federated Generative AI Framework for Glaucoma Detection.

## Overview

The framework implements a three-stage pipeline that combines federated learning, generative models, and multi-modal fusion to enable privacy-preserving glaucoma detection across distributed healthcare institutions.

For a visual representation, see [architecture_diagram.pdf](architecture_diagram.pdf).

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Federated Framework                       │
│                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ Hospital │  │ Hospital │  │ Hospital │                  │
│  │    A     │  │    B     │  │    C     │                  │
│  │ (EyePACS)│  │(REFUGE2) │  │ (PAPILA) │                  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
│       │             │             │                          │
│       │  Model      │  Model      │  Model                  │
│       │  Weights    │  Weights    │  Weights                │
│       │             │             │                          │
│       └─────────────┴─────────────┘                          │
│                     │                                        │
│              ┌──────▼──────┐                                │
│              │   Federated  │                                │
│              │   Server     │                                │
│              │  (Aggregator)│                                │
│              └──────┬───────┘                                │
│                     │                                        │
│              ┌──────▼───────┐                               │
│              │ Global Model  │                               │
│              └──────┬───────┘                                │
│                     │                                        │
│         ┌───────────┴───────────┐                           │
│         │                       │                           │
│    ┌────▼─────┐         ┌──────▼──────┐                   │
│    │ Vision   │         │ Language     │                   │
│    │ Features │         │ Generation   │                   │
│    └────┬─────┘         └──────┬───────┘                   │
│         │                       │                           │
│         └───────────┬───────────┘                           │
│                     │                                        │
│              ┌──────▼──────┐                                │
│              │ Multi-Modal  │                                │
│              │   Fusion     │                                │
│              └──────┬───────┘                                │
│                     │                                        │
│              ┌──────▼──────┐                                │
│              │  Clinical    │                                │
│              │  Prediction  │                                │
│              │  + Report    │                                │
│              └──────────────┘                                │
└───────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Data Layer

**Purpose**: Load, preprocess, and augment retinal fundus images

**Components**:
- **ImageDataGenerator**: Creates augmented batches for training
- **CLAHE Enhancement**: Improves contrast in fundus images
- **Optic Disc Localization**: Detects and crops region of interest
- **Quality Assessment**: Filters low-quality images using Laplacian variance

**Data Flow**:
```
Raw Images → Resize (512×512) → CLAHE → Optic Disc Crop →
Augmentation → Normalization → Batches
```

**Augmentations Applied**:
- Horizontal/vertical flips (p=0.5)
- Rotation (±180°)
- Brightness adjustment (±5%)
- Gaussian noise (σ ~ U(0, 10))

---

### 2. Generative Layer

**Purpose**: Synthesize realistic fundus images to augment training data

#### 2.1 StyleGAN2 Generator

**Architecture**:
```
Latent Z (512-dim) →
  Mapping Network (8 layers, 512 units) →
    Style Codes →
      Synthesis Network:
        - Progressive upsampling: 4×4 → 8×8 → ... → 256×256
        - Adaptive Instance Normalization (AdaIN) at each layer
        - Style modulation per resolution
      → RGB Image (256×256×3)
```

**Training**:
- Progressive growing from 4×4 to 256×256
- 100,000 iterations
- Loss: Non-saturating GAN loss + R1 regularization
- Optimizer: Adam (lr=0.002, β1=0, β2=0.99)

#### 2.2 Latent Diffusion Model

**Architecture**:
```
Image (512×512) →
  VAE Encoder → Latent (64×64) →
    Diffusion Process (1000 timesteps):
      Forward: Add Gaussian noise
      Reverse: U-Net denoising
    → Denoised Latent →
  VAE Decoder → Generated Image (512×512)
```

**Training**:
- Timesteps: T=1000
- Noise schedule: Linear (β₁=1e-4, βₜ=0.02)
- U-Net: 4 down-blocks, 4 up-blocks, attention at 16×16 resolution

**Quality Metrics**:
- FID (Fréchet Inception Distance): Measures distribution similarity
- SSIM (Structural Similarity): Compares structural features
- LPIPS (Learned Perceptual Similarity): Perceptual quality

---

### 3. Federated Learning Layer

**Purpose**: Train models across distributed datasets without sharing raw data

#### 3.1 Federation Setup

**Clients**: 3 independent hospitals
- Client A: EyePACS-AIROGS dataset
- Client B: REFUGE2 dataset
- Client C: PAPILA dataset

**Data Characteristics**:
- Heterogeneous: Different imaging devices, resolutions, patient demographics
- Imbalanced: Varying glaucoma prevalence across sites
- Private: Data never leaves local institutions

#### 3.2 H-UQ-MFF Architecture

**Hierarchical Uncertainty Quantification Multi-scale Feature Fusion**

```
Input Image (512×512×3) →
  ResNet-50 Backbone:
    ├─ Layer 1 (256×256×256)  ──┐
    ├─ Layer 2 (128×128×512)  ──┤
    ├─ Layer 3 (64×64×1024)   ──┼─ Multi-scale Features
    └─ Layer 4 (32×32×2048)   ──┘
                                  │
    ┌─────────────────────────────┘
    │
    ├─ Channel Attention:
    │    Global Avg Pool → FC → ReLU → FC → Sigmoid
    │    Weights: [w₁, w₂, w₃, w₄]
    │
    ├─ Spatial Attention:
    │    Conv 7×7 → Sigmoid
    │    Attention Maps: [A₁, A₂, A₃, A₄]
    │
    └─ Hierarchical Fusion:
         F = Σᵢ wᵢ × (Fᵢ ⊙ Aᵢ)
         → Feature Vector (512-dim)
         → Uncertainty Head (MC Dropout, 50 passes)
         → Prediction + Confidence
```

**Key Components**:
1. **Multi-scale Extraction**: Captures features at 4 resolutions
2. **Channel Attention**: Learns importance of feature channels
3. **Spatial Attention**: Focuses on relevant image regions
4. **Hierarchical Fusion**: Combines scales with learned weights
5. **Uncertainty Quantification**: Monte Carlo Dropout for confidence

#### 3.3 Federated Algorithms

**FedAvg (Federated Averaging)**:
```
Server:
  Initialize global model w₀
  For round t = 1 to T:
    Send wₜ to all clients
    For each client k in parallel:
      wₖᵗ⁺¹ = LocalUpdate(wₜ, Dₖ, E)
    Aggregate: wₜ₊₁ = Σₖ (nₖ/n) × wₖᵗ⁺¹
  Return wₜ

LocalUpdate(w, D, E):
  For epoch e = 1 to E:
    For batch b in D:
      w ← w - η∇L(w, b)
  Return w
```

**FedProx (Federated Proximal)**:
```
LocalUpdate(w, D, E, μ):
  wᵍˡᵒᵇᵃˡ = w  # Store global model
  For epoch e = 1 to E:
    For batch b in D:
      # Add proximal term
      w ← w - η(∇L(w, b) + μ(w - wᵍˡᵒᵇᵃˡ))
  Return w
```

**Hyperparameters**:
- Rounds (T): 50
- Local epochs (E): 10
- Batch size: 8
- Learning rate (η): 1e-4
- Proximal term (μ): 0.01 (FedProx only)

**Communication Protocol**:
1. Server broadcasts global model weights
2. Clients train locally (no data sharing)
3. Clients upload gradients/weights (compressed)
4. Server aggregates using weighted average
5. Repeat until convergence

**Privacy Guarantees**:
- Raw images never leave hospitals
- Only model parameters communicated
- Optional: Differential privacy via gradient clipping

---

### 4. Multi-Modal Fusion Layer

**Purpose**: Combine visual features with clinical language for enhanced diagnosis

#### 4.1 Vision Encoder

```
Image Features (512-dim from H-UQ-MFF) →
  Projection Head:
    - FC Layer 1: 512 → 768 (GELU)
    - Dropout: 0.1
    - FC Layer 2: 768 → 768 (GELU)
    - Layer Norm
  → Vision Embeddings (768-dim)
```

#### 4.2 Language Model

**MedGemma-4B**:
- Parameters: 4.2 billion
- Architecture: Transformer decoder (32 layers, 32 attention heads)
- Vocabulary: 256k tokens (medical domain)
- Context length: 8192 tokens

**QLoRA Fine-tuning**:
- Trainable: LoRA adapters only (rank=8, alpha=16)
- Total params: 505,986
- Trainable params: 176,770 (34.94%)
- Frozen: Base model weights

#### 4.3 Fusion Strategy

**Early Fusion**:
```
Vision (768-dim) + Language (768-dim) →
  Concatenation: [Vision; Language] (1536-dim) →
    FC: 1536 → 768 (GELU) →
    Dropout: 0.2 →
    FC: 768 → 2 (Binary Classification)
```

**Attention-based Fusion**:
```
Q = Linear(Vision)
K, V = Linear(Language)
Attention = Softmax(QKᵀ/√d) × V
Fused = Attention + Vision
```

---

### 5. Explanation Layer

**Purpose**: Generate interpretable clinical reports with uncertainty

#### 5.1 Uncertainty Quantification

**Monte Carlo Dropout**:
```
For i = 1 to N (N=50):
  Enable dropout at inference
  pᵢ = model(x) with dropout

Mean prediction: p̄ = (1/N) Σᵢ pᵢ
Uncertainty: σ² = (1/N) Σᵢ (pᵢ - p̄)²
Confidence: c = 1 - σ²
```

#### 5.2 Report Generation

**Template-based Generation**:
```
IF p̄ > 0.5:
  Diagnosis: "Glaucoma suspected"
ELSE:
  Diagnosis: "No glaucoma detected"

IF c > 0.9:
  Confidence: "High confidence"
ELIF c > 0.7:
  Confidence: "Moderate confidence"
ELSE:
  Confidence: "Low confidence - recommend expert review"

Report: f"{Diagnosis} with {Confidence} (probability: {p̄:.2%})"
```

**LLM-based Generation** (optional):
```
Prompt: "Generate clinical report for retinal image with:
  - Predicted probability: {p̄}
  - Uncertainty: {σ²}
  - Key features: {feature_summary}

Report should include diagnosis, confidence, and recommendations."

MedGemma-4B(Prompt) → Structured Clinical Report
```

---

## Data Flow

### Training Phase

```
1. Data Loading:
   Hospitals A, B, C load local datasets

2. Local Preprocessing:
   Each hospital preprocesses independently

3. Generative Augmentation:
   Each hospital generates synthetic samples locally

4. Federated Training (Round t):
   a. Server sends global model wₜ to all clients
   b. Each client trains locally:
      - Extract features with H-UQ-MFF
      - Train classifier with local data
      - Compute loss and gradients
   c. Clients send wₖᵗ⁺¹ to server
   d. Server aggregates: wₜ₊₁ = Σₖ (nₖ/n) × wₖᵗ⁺¹

5. Vision-Language Fusion:
   Fine-tune MedGemma on combined features

6. Validation:
   Test on cross-dataset holdout sets
```

### Inference Phase

```
Input: Fundus Image (512×512×3)
  ↓
Preprocessing: CLAHE + Normalization
  ↓
Feature Extraction: H-UQ-MFF → (512-dim)
  ↓
Uncertainty Estimation: MC Dropout (50 passes)
  ↓
Vision-Language Fusion: Features → MedGemma
  ↓
Prediction: p̄ ± σ (probability ± uncertainty)
  ↓
Report Generation: Clinical explanation
  ↓
Output: {diagnosis, confidence, report}
```

---

## Scalability

### Horizontal Scaling

**Adding New Hospitals**:
1. Initialize new federated client
2. Client trains locally on private data
3. Participates in federated aggregation
4. No code changes to server required

### Vertical Scaling

**Supporting More Datasets**:
- Framework handles heterogeneous data automatically
- FedProx manages statistical heterogeneity
- Multi-scale features adapt to resolution differences

### Computational Efficiency

**Model Compression**:
- GGUF quantization (8-bit)
- 75% size reduction (95 MB → 24 MB)
- Minimal accuracy drop (<1%)

**Communication Efficiency**:
- Gradient compression (sparsification)
- Model weight quantization before upload
- Reduces bandwidth by ~80%

---

## Security & Privacy

### Threat Model

**Trusted**: Server (aggregator)
**Untrusted**: External adversaries, other clients (semi-honest)

### Privacy Mechanisms

1. **Federated Learning**: Data never leaves hospitals
2. **Secure Aggregation** (optional): Encrypted model updates
3. **Differential Privacy** (optional): Add noise to gradients
   - Noise scale: σ = C × √(2ln(1.25/δ)) / ε
   - Privacy budget: ε=1.0, δ=1e-5

### Data Protection

- Images stored locally only
- Model weights communicated via TLS
- No patient identifiers in logs
- HIPAA-compliant architecture

---

## Performance Optimizations

### GPU Utilization

- Mixed precision training (FP16)
- Batch accumulation for effective batch size
- Multi-GPU data parallelism

### Memory Optimization

- Gradient checkpointing
- Dynamic batch sizing
- Offload to CPU during inference

### Training Speed

- Cached preprocessing outputs
- Prefetching with DataLoader
- Asynchronous federated updates

---

## Key Design Decisions

### Why ResNet-50?

- **Proven**: Strong performance on ImageNet
- **Transferable**: Learned features generalize to medical images
- **Efficient**: 25M parameters (not too large)
- **Multi-scale**: Easy to extract intermediate features

### Why StyleGAN2 + Diffusion?

- **StyleGAN2**: High-quality, diverse samples
- **Diffusion**: Better mode coverage, no mode collapse
- **Complementary**: Different inductive biases capture different aspects

### Why FedAvg + FedProx?

- **FedAvg**: Simple, effective baseline
- **FedProx**: Handles heterogeneous data better
- **Comparison**: Demonstrates robustness to data distribution

### Why Monte Carlo Dropout?

- **Bayesian**: Provides calibrated uncertainty
- **Efficient**: No ensemble required
- **Clinically useful**: Identifies uncertain cases for human review

---

## Limitations & Future Work

### Current Limitations

1. **Scalability**: Tested on 3 clients (need 10+ for real-world validation)
2. **Communication**: Assumes reliable network (need fault tolerance)
3. **Heterogeneity**: Limited to image resolution differences (need device diversity)

### Future Enhancements

1. **Asynchronous Federated Learning**: Don't wait for slow clients
2. **Personalized Models**: Client-specific fine-tuning layers
3. **Continual Learning**: Handle data distribution shifts over time
4. **Multi-task Learning**: Simultaneous glaucoma + AMD + DR detection

---

## References

For detailed implementation, see:
- [IMPLEMENTATION.md](IMPLEMENTATION.md) - Technical specifications
- [USAGE.md](USAGE.md) - How to run the framework
- [architecture_diagram.pdf](architecture_diagram.pdf) - Visual architecture
