# Medical AI Dashboard - Expert-Guided Multimodal AI Ecosystem

A comprehensive full-stack medical AI dashboard for brain tumor segmentation with AI-powered report generation and continual learning capabilities.

## 🎯 Project Overview

This system implements a complete medical AI workflow with:

- **Multi-modal MRI Analysis** (T1, T1ce, T2, FLAIR)
- **AI Segmentation** using MoME+ architecture  
- **LLM-Powered Report Generation** with MedAlpaca-7B
- **Interactive Report Editing** with evidence traceability
- **Continual Learning** for model adaptation
- **3D Visualization** and PDF export
- **Clinician-in-the-Loop** feedback system

## 🏗️ Architecture

### Frontend
- **Framework**: React 18 + TypeScript + Vite
- **UI Library**: Modern component library with medical-grade UX
- **State Management**: React Context + Hooks
- **Visualization**: 3D rendering with Three.js/glTF
- **Styling**: CSS Modules with premium medical design

### Backend
- **Framework**: Django 4.2 + Django REST Framework
- **Database**: PostgreSQL with optimized schema
- **Task Queue**: Celery + Redis for async processing
- **Authentication**: JWT with role-based access (clinician/admin/researcher)
- **API Docs**: OpenAPI/Swagger with drf-spectacular
- **File Storage**: Local + Cloud (GCP/AWS S3) support

### ML Pipeline
- **Segmentation**: MoME+ model with 3D U-Net experts
- **Report Generation**: Med Alpaca-7B with LoRA/PEFT
- **Continual Learning**: EWC + Replay buffer
- **Inference**: Async with Celery, ONNX optimization

## 📁 Project Structure

```
An-Expert-Guided-Multimodal-AI-Ecosystem/
├── frontend/                # React TypeScript frontend
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Page components
│   │   ├── services/       # API client services
│   │   ├── contexts/       # React contexts
│   │   ├── hooks/          # Custom hooks
│   │   ├── types/          # TypeScript definitions
│   │   └── styles/         # Global styles
│   └── public/
│
├── backend/                 # Django REST backend
│   ├── medical_ai_backend/ # Project settings
│   ├── users/              # User management
│   ├── cases/              # Case & MRI management
│   ├── reports/            # Report generation
│   ├── inference/          # ML inference tasks
│   ├── media/              # User-uploaded files
│   └── requirements.txt
│
├── src/                     # ML models & training
│   ├── models/             # MoME+ architecture
│   ├── training/           # Training pipeline
│   ├── inference/          # Inference engine
│   ├── preprocessing/      # Data preprocessing
│   └── llm/                # LLM integration
│
├── configs/                 # Configuration files
├── data/                    # Datasets
├── experiments/             # Model checkpoints
├── docker/                  # Docker configurations
└── docs/                    # Documentation

```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+ and npm
- PostgreSQL 14+
- Redis 6+
- CUDA-compatible GPU (recommended for ML inference)
- Docker & Docker Compose (for deployment)

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Create .env file
copy .env.example .env
# Edit .env with your configuration

# Run database migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Run development server
python manage.py runserver
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Create .env file
copy .env.example .env

# Run development server
npm run dev
```

### Celery Worker (for async tasks)

```bash
# In backend directory
celery -A medical_ai_backend worker -l info
```

### Redis (required for Celery)

```bash
# Using Docker
docker run -d -p 6379:6379 redis:latest

# Or install Redis locally
```

## 📊 Database Setup

### PostgreSQL

```bash
# Create database
createdb medical_ai_db

# Or using psql
psql -U postgres
CREATE DATABASE medical_ai_db;

CREATE USER medical_ai_user WITH PASSWORD 'fyp_pwd';
GRANT ALL PRIVILEGES ON DATABASE medical_ai_db TO medical_ai_user;
```

Update your `.env` file:
```
DATABASE_URL=postgresql://medical_ai_user:fyp_pwd@localhost:5432/medical_ai_db
```

## 🔑 Environment Variables

### Backend (.env)

Key variables (see `.env.example` for complete list):

```bash
SECRET_KEY=your-secret-key
DEBUG=True
DATABASE_URL=postgresql://...
REDIS_URL=redis://localhost:6379/0
SEGMENTATION_MODEL_PATH=../experiments/checkpoints/best_model.pth
LLM_MODEL_PATH=../models/medalpaca-7b
```

### Frontend (. env)

```bash
VITE_API_URL=http://localhost:8000/api/v1
VITE_WS_URL=ws://localhost:8000/ws
```

## 🎨 Key Features

### 1. Case Management
- Upload multi-modal MRI scans (NIfTI format)
- De-identified patient data management
- Case status tracking (created → processing → completed)

### 2. AI Segmentation
- Automatic brain tumor segmentation
- Three tumor regions: WT (Whole Tumor), TC (Tumor Core), ET (Enhancing Tumor)
- Confidence scores and volume metrics

### 3. Report Generation
- LLM-generated professional radiology reports
- Interactive editing with traceability
- Evidence mapping (click sentence → see supporting data)

### 4. 3D Visualization
- Interactive 3D tumor rendering
- Multiple 2D slice views (axial, coronal, sagittal)
- glTF export for external viewers

### 5. PDF Export
- Professional report PDF with visualizations
- Includes 2D slices and 3D renders
- Watermarked for research use

### 6. Clinician Feedback
- Flag segmentation errors
- Provide corrections for continual learning
- Feedback stored for model improvement

### 7. Continual Learning (Admin)
- Introduce new labeled datasets
- EWC + Replay anti-forgetting strategies
- Model versioning and rollback

## 🔒 Security Features

- ✅ JWT authentication with refresh tokens
- ✅ Role-based access control (RBAC)
- ✅ HTTPS/TLS encryption in transit
- ✅ Data encryption at rest
- ✅ bcrypt password hashing
- ✅ Session management and timeouts
- ✅ CORS configuration
- ✅ Input validation and sanitization

## 📖 API Documentation

Once the server is running, access:

- **Swagger UI**: http://localhost:8000/api/docs/
- **OpenAPI Schema**: http://localhost:8000/api/schema/

### Key Endpoints

```
POST   /api/v1/auth/register          # Register new user
POST   /api/v1/auth/login             # Login
POST   /api/v1/auth/refresh           # Refresh JWT token

GET    /api/v1/cases/                 # List cases
POST   /api/v1/cases/                 # Create new case
GET    /api/v1/cases/{id}/            # Get case details
POST   /api/v1/cases/{id}/upload/     # Upload MRI images

POST   /api/v1/inference/segment/     # Start segmentation task
GET    /api/v1/inference/tasks/{id}/  # Get task status

GET    /api/v1/reports/{id}/          # Get report
PATCH  /api/v1/reports/{id}/          # Edit report
POST   /api/v1/reports/{id}/export/   # Export to PDF

POST   /api/v1/feedback/              # Submit feedback
```

## 🐳 Docker Deployment

```bash
# Build and run all services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v --cov=.
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 📈 Performance Requirements

- **End-to-end processing**: < 5 minutes
- **UI responsiveness**: < 2 seconds
- **Concurrent users**: 3+ simultaneous analyses
- **Model inference**: Optimized with quantization and ONNX

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

See [LICENSE](../LICENSE) file for details.

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Contact the development team

## 🙏 Acknowledgments

- BraTS 2021 dataset
- OASIS dataset
- ISLES dataset
- MedAlpaca LLM
- MONAI framework

---

**Note**: This is a research prototype intended for investigational and research purposes only. It is not a certified medical device and should not be used for primary diagnosis.
