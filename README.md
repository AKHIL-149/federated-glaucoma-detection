# Federated Generative AI Framework for Glaucoma Detection

A privacy-preserving federated learning framework combining generative AI with multi-modal fusion for automated glaucoma detection across heterogeneous clinical datasets.

## Overview

This repository implements a novel federated learning approach for automated glaucoma detection from retinal fundus images. The framework addresses key challenges in medical AI deployment: data privacy, cross-institutional collaboration, and clinical interpretability.

**Key Innovation**: Combines federated learning with generative augmentation and multi-modal vision-language fusion to achieve robust glaucoma detection while preserving patient privacy across multiple healthcare institutions.

## Key Features

- **Privacy-Preserving Federated Learning**: Train on distributed hospital datasets (EyePACS, REFUGE2, PAPILA) without centralizing patient data. Only model weights are shared across institutions.

- **Generative Data Augmentation**: Uses StyleGAN2 and diffusion models to synthesize realistic fundus images, addressing class imbalance and improving model generalization.

- **Multi-Modal Vision-Language Fusion**: Integrates visual features with clinical language descriptions for enhanced diagnostic accuracy and explainability.

- **Clinical Interpretability**: Provides uncertainty-aware predictions with automated clinical explanations to support ophthalmologist decision-making.

- **Cross-Dataset Validation**: Evaluated on three independent public datasets to ensure generalization across different imaging protocols and patient populations.

- **Federated Algorithms**: Implements both FedAvg and FedProx for handling heterogeneous data distributions across medical institutions.

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 84.8% |
| Precision | 84.69% |
| Recall | 85.69% |
| F1-Score | 85.19% |
| Specificity | 83.88% |
| AUC-ROC | 92.4% |
| MCC | 0.696 |

**Federated Learning Results**:
- Initial accuracy: 54.05% (random initialization)
- Final accuracy: 82.48%
- Best accuracy: 87.95% (at round 40)
- Total improvement: 28.43%

## Repository Structure

```
federated-glaucoma-detection/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── CITATION.bib                      # Academic citations
├── notebooks/
│   └── federated_generative_framework.ipynb  # Main implementation (62 cells)
├── data/
│   └── dataset_links.txt             # Links to public datasets
├── docs/
│   ├── INSTALLATION.md               # Setup instructions
│   ├── USAGE.md                      # How to run the framework
│   ├── ARCHITECTURE.md               # System design
│   ├── RESULTS.md                    # Detailed experimental results
│   ├── IMPLEMENTATION.md             # Technical specifications
│   └── architecture_diagram.pdf      # Visual architecture (79 KB)
└── outputs/
    ├── README.md                     # Output documentation
    ├── federated_learning/           # Algorithm comparisons, training curves
    ├── multimodal_fusion/            # Confusion matrices, modality analysis
    ├── clinical_explanations/        # Sample explanations with uncertainty
    └── generative_results/           # Synthetic image samples
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA 11.x (optional, for GPU acceleration)
- 16GB RAM minimum
- 50GB free disk space for datasets

### Installation

1. Clone the repository:
```bash
git clone https://github.com/[username]/federated-glaucoma-detection.git
cd federated-glaucoma-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download datasets (see [data/dataset_links.txt](data/dataset_links.txt)):
   - EyePACS-AIROGS-light-V2 (Kaggle)
   - REFUGE2 (Kaggle)
   - PAPILA (Figshare)

5. Launch the Jupyter notebook:
```bash
jupyter notebook notebooks/federated_generative_framework.ipynb
```

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for detailed setup instructions.

## Usage

The framework is organized into 11 phases within the Jupyter notebook:

1. **Data Loading & Augmentation**: Load fundus images and apply preprocessing
2. **Image Preprocessing**: CLAHE enhancement, resizing, normalization
3. **Federated Feature Learning**: Extract multi-scale features with H-UQ-MFF architecture
4. **Federated Learning**: Train using FedAvg or FedProx across distributed datasets
5. **Generative AI Module**: Synthesize realistic fundus images for data augmentation
6. **Vision-Language Learning**: Align visual features with clinical language
7. **Clinical Report Generation**: Generate automated explanations with uncertainty
8. **Cross-Dataset Validation**: Test generalization on external datasets
9. **Expert Evaluation**: Validate predictions against ophthalmologist assessments
10. **Model Compression**: Quantize models for deployment
11. **Final Testing**: Comprehensive performance evaluation

See [docs/USAGE.md](docs/USAGE.md) for detailed instructions.

## Datasets

This framework uses three public glaucoma detection datasets:

1. **EyePACS-AIROGS-light-V2** - Retinal fundus images with glaucoma labels
2. **REFUGE2** - REFUGE Challenge dataset with segmentation masks
3. **PAPILA** - Paired fundus images with clinical metadata

Dataset links are provided in [data/dataset_links.txt](data/dataset_links.txt).

## Architecture

The framework implements a three-stage pipeline:

1. **Federated Generative Augmentation**: Each hospital client trains generative models (StyleGAN2, Diffusion) locally to augment their private datasets
2. **Federated Multi-Modal Training**: Clients train vision-language models locally, server aggregates weights using FedAvg/FedProx
3. **Clinical Explanation Generation**: Uncertainty quantification and automated report generation for clinical decision support

For detailed architecture, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/architecture_diagram.pdf](docs/architecture_diagram.pdf).

## Results

Comprehensive results are available in [docs/RESULTS.md](docs/RESULTS.md) and visualized in the [outputs/](outputs/) directory:

- **Federated Learning**: Training curves show convergence at round 29, achieving 87.95% peak accuracy
- **Multi-Modal Fusion**: Vision-language fusion improves accuracy by ~3-5% over vision-only baselines
- **Generative Quality**: FID score of 2.2e-05 indicates high-quality synthetic image generation
- **Cross-Dataset Validation**: Model generalizes well across different imaging protocols

## Pretrained Models

Due to GitHub size constraints, pretrained model weights (43 MB) are not included in this repository. If you need pretrained weights to reproduce results without full training:

- Open an issue in this repository, or
- Contact the authors directly

Alternatively, you can train models from scratch by running the complete notebook (estimated 6-8 hours on GPU).

## Technical Details

- **Deep Learning**: TensorFlow 2.8, PyTorch 1.12
- **Backbone Architecture**: ResNet-50 (pretrained on ImageNet)
- **Generative Models**: StyleGAN2 (256×256), Latent Diffusion (1000 timesteps)
- **Federated Setup**: 3 clients, 10 rounds, 5 local epochs, batch size 8
- **Vision-Language**: MedGemma-4B with QLoRA fine-tuning (34.94% trainable parameters)

See [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for complete specifications.

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@software{federated_glaucoma_detection,
  title={Federated Generative AI Framework for Glaucoma Detection},
  author={AKHIL-149},
  year={2026},
  url={https://github.com/[username]/federated-glaucoma-detection}
}
```

For dataset citations, see [CITATION.bib](CITATION.bib).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset providers: EyePACS-AIROGS, REFUGE Challenge, PAPILA
- Base paper authors and reference work contributors
- Open-source community for frameworks (TensorFlow, PyTorch, Hugging Face)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions, collaborations, or access to pretrained models, please open an issue in this repository.

---

**Note**: This repository has been optimized for GitHub hosting (~12 MB) from the original research outputs (251 MB). Full experimental outputs can be regenerated by running the complete notebook.
