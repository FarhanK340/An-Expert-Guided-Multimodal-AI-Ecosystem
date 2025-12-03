# Expert MoME+ FYP Project

## Overview

This project implements a **Mixture of Modality Experts (MoME+)** architecture for medical image segmentation with continual learning capabilities and LLM-based report generation. The system is designed to handle multi-modal MRI data (T1, T1ce, T2, FLAIR) for brain tumor segmentation across different datasets (BraTS, OASIS, ISLES).

## Architecture

### Core Components

1. **MoME+ Segmentation Model**
   - Four modality-specific expert networks (3D U-Nets)
   - Hierarchical gating network for dynamic expert fusion
   - CBAM/SE attention mechanisms
   - Outputs: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET)

2. **Continual Learning**
   - Elastic Weight Consolidation (EWC) for knowledge retention
   - Experience replay buffer
   - Task-based and lesion-based learning scenarios

3. **LLM Integration**
   - MedAlpaca-7B fine-tuning with LoRA/PEFT
   - Structured report generation from segmentation results
   - Factual consistency verification

4. **API & Deployment**
   - Django REST Framework backend
   - Async inference with Celery + Redis
   - 3D visualization with glTF export
   - Docker containerization

## Project Structure

```
expert_mome_fyp/
├── configs/                 # Configuration files
├── data/                    # Raw and processed datasets
├── src/                     # Source code
│   ├── preprocessing/       # Data preprocessing
│   ├── models/             # MoME+ architecture
│   ├── training/           # Training pipeline
│   ├── inference/          # Inference and export
│   ├── api/                # Django REST API
│   ├── llm/                # LLM integration
│   └── utils/              # Utilities
├── experiments/            # Checkpoints and results
├── notebooks/              # Jupyter notebooks
├── deployment/             # Docker and deployment
└── tests/                  # Unit tests
```

## Setup Instructions

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (recommended)
- Docker (for deployment)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd expert_mome_fyp
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Quick Start

1. **Data Preprocessing**:
```bash
python src/preprocessing/data_preprocessing.py --config configs/train_config.yaml
```

2. **Training**:
```bash
python src/training/trainer.py --config configs/train_config.yaml
```

3. **Inference**:
```bash
python src/inference/inference_engine.py --input data/samples/ --output results/
```

4. **API Server**:
```bash
python src/api/app.py
```

## Configuration

Key configuration files in `configs/`:

- `train_config.yaml`: Training hyperparameters
- `model_config.yaml`: Architecture settings
- `continual_learning.yaml`: CL parameters
- `inference_config.yaml`: Inference settings

## Datasets

Supported datasets:
- **BraTS 2021**: Brain tumor segmentation
- **OASIS**: Alzheimer's disease
- **ISLES**: Ischemic stroke

Place raw data in `data/raw/` following the expected directory structure.

## API Endpoints

- `POST /api/predict`: Upload MRI for segmentation
- `GET /api/report/{case_id}`: Generate clinical report
- `POST /api/feedback`: Submit clinician feedback

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
```bash
black src/
flake8 src/
```

### Docker Deployment
```bash
docker-compose up --build
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[Add your license information here]

## Contact

[Add your contact information here]

