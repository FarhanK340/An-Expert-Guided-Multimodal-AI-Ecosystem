# System Architecture Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MEDICAL AI DASHBOARD                               │
│                   Expert-Guided Multimodal AI Ecosystem                   │
└─────────────────────────────────────────────────────────────────────────┘

┌───────────────────────┐          ┌──────────────────────────────────────┐
│                       │          │                                      │
│   REACT FRONTEND      │◄────────►│     DJANGO REST BACKEND              │
│   (TypeScript + Vite) │   HTTP   │     (Python 3.12)                    │
│                       │   JWT    │                                      │
│   Port: 5173          │          │     Port: 8000                       │
│                       │          │                                      │
│   ┌───────────────┐   │          │   ┌──────────────────────────────┐   │
│   │ Auth Pages    │   │          │   │  Users App                   │   │
│   │ Dashboard     │   │          │   │  - JWT Authentication        │   │
│   │ Case Mgmt     │   │          │   │  - User Profiles             │   │
│   │ Upload UI     │   │          │   └──────────────────────────────┘   │
│   │ 2D Viewer     │   │          │   ┌──────────────────────────────┐   │
│   │ 3D Viewer     │   │          │   │  Cases App                   │   │
│   │ Report Editor │   │          │   │  - Case Management           │   │
│   │ PDF Export    │   │          │   │  - MRI Upload                │   │
│   └───────────────┘   │          │   │  - Segmentation Results      │   │
│                       │          │   └──────────────────────────────┘   │
└───────────────────────┘          │   ┌──────────────────────────────┐   │
                                   │   │  Reports App                 │   │
                                   │   │  - LLM Report Generation     │   │
                                   │   │  - Report Editing            │   │
                                   │   │  - PDF Export                │   │
                                   │   │  - Traceability Links        │   │
                                   │   └──────────────────────────────┘   │
                                   │   ┌──────────────────────────────┐   │
                                   │   │  Inference App               │   │
                                   │   │  - ML Task Management        │   │
                                   │   │  - Model Versioning          │   │
                                   │   │  - Continual Learning        │   │
                                   │   └──────────────────────────────┘   │
                                   │                                      │
                                   └──────────────────────────────────────┘
                                             │           │
                         ┌───────────────────┼───────────┼────────────────┐
                         │                   │           │                │
                         ▼                   ▼           ▼                ▼
              ┌──────────────────┐ ┌─────────────┐ ┌──────────┐ ┌────────────┐
              │   POSTGRESQL     │ │   REDIS     │ │ CELERY   │ │  ML MODELS │
              │   Database       │ │   Cache &   │ │ Worker   │ │            │
              │                  │ │   Broker    │ │          │ │  MoME+     │
              │   Port: 5432     │ │ Port: 6379  │ │          │ │  MedAlpaca │
              │                  │ │             │ │          │ │            │
              │  ┌────────────┐  │ │             │ │ Tasks:   │ │ ONNX       │
              │  │ Users      │  │ │             │ │ • Segment│ │ Optimized  │
              │  │ Cases      │  │ │             │ │ • Report │ │            │
              │  │ MRI Images │  │ │             │ │ • 3D Gen │ │ Path:      │
              │  │ Results    │  │ │             │ │ • PDF    │ │ /experiments│
              │  │ Reports    │  │ │             │ │ • CL     │ │ /models    │
              │  │ Tasks      │  │ │             │ │          │ │            │
              │  └────────────┘  │ │             │ │          │ │            │
              └──────────────────┘ └─────────────┘ └──────────┘ └────────────┘
```

## Data Flow

### 1. Case Upload & Processing
```
User → Frontend → Backend API → Database (Case Created)
                     ↓
              MRI Files Upload → File Storage
                     ↓
              Validation & Metadata Extraction
                     ↓
              Celery Task Queued (Redis)
                     ↓
              Celery Worker → Load MoME+ Model → Inference
                     ↓
              Segmentation Results → Database
                     ↓
              3D Visualization Generated (glTF)
                     ↓
              2D Slices Generated (PNG)
                     ↓
              Status Update → Frontend (via polling/websocket)
```

### 2. Report Generation
```
Segmentation Complete → Trigger Report Generation
              ↓
    Extract Structured Data (JSON)
              ↓
    Populate Report Template
              ↓
    LLM (MedAlpaca) → Generate Narrative
              ↓
    Create Traceability Links
              ↓
    Save Draft Report → Database
              ↓
    Notify User → Frontend
```

### 3. Clinician Review & Edit
```
User Views Report → Frontend Editor
              ↓
    Make Edits → PATCH API
              ↓
    Track Changes → ReportEdit Model
              ↓
    Click Sentence → Highlight Evidence (Traceability)
              ↓
    Finalize Report
              ↓
    Export to PDF → Celery Task
              ↓
    PDF Generated → Download
```

### 4. Continual Learning (Admin)
```
Admin → Upload New Dataset
              ↓
    Configure CL Task (EWC + Replay)
              ↓
    Celery Worker → Load Base Model
              ↓
    Train on New Data (EWC Loss + Replay Buffer)
              ↓
    Evaluate (New Task + Old Tasks)
              ↓
    Save New Model Version
              ↓
    Admin Reviews Metrics → Activate Model
```

## Technology Stack

### Frontend
```
React 18
├── TypeScript
├── Vite (Build Tool)
├── React Router (Navigation)
├── React Query (API State)
├── React Hook Form (Forms)
├── Zod (Validation)
├── shadcn/ui (Components)
├── Three.js (3D Visualization)
└── TailwindCSS (Styling - optional)
```

### Backend
```
Django 4.2
├── Django REST Framework
├── PostgreSQL (Database)
├── Redis (Cache & Message Broker)
├── Celery (Async Tasks)
├── JWT (Authentication)
├── drf-spectacular (API Docs)
├── ReportLab (PDF Generation)
└── python-decouple (Config)
```

### ML Stack
```
PyTorch 2.0
├── MONAI (Medical Imaging)
├── Transformers (LLM)
├── PEFT/LoRA (Fine-tuning)
├── ONNX Runtime (Optimization)
├── Nibabel (NIfTI Processing)
└── Trimesh (3D Mesh Generation)
```

## Security Architecture

```
┌─────────────────────────────────────────────────┐
│              SECURITY LAYERS                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. HTTPS/TLS (Transport Layer)                 │
│     └─ All traffic encrypted                    │
│                                                 │
│  2. JWT Authentication                          │
│     ├─ Access Token (60 min)                    │
│     ├─ Refresh Token (24 hours)                 │
│     └─ Token Rotation                           │
│                                                 │
│  3. Role-Based Access Control (RBAC)            │
│     ├─ Clinician (read/write cases)             │
│     ├─ Admin (all permissions)                  │
│     └─ Researcher (read-only)                   │
│                                                 │
│  4. Data Isolation                              │
│     └─ User can only access own cases           │
│                                                 │
│  5. Password Security                           │
│     └─ bcrypt hashing with salt                 │
│                                                 │
│  6. Session Management                          │
│     ├─ Session tracking                         │
│     └─ Auto-logout on inactivity                │
│                                                 │
│  7. Input Validation                            │
│     ├─ File type checking                       │
│     ├─ Size limits                              │
│     └─ Sanitization                             │
│                                                 │
│  8. Rate Limiting (TODO)                        │
│     └─ Prevent abuse                            │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Deployment Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│                         NGINX                                │
│                    (Reverse Proxy)                           │
│                       Port 80/443                            │
│                           │                                  │
│         ┌─────────────────┼─────────────────┐                │
│         │                 │                 │                │
│         ▼                 ▼                 ▼                │
│   Static Files      Backend API      Frontend SPA           │
│   (Django)          (Gunicorn)       (React Build)          │
│                           │                                  │
│         ┌─────────────────┼─────────────────┐                │
│         │                 │                 │                │
│         ▼                 ▼                 ▼                │
│   PostgreSQL          Redis          Celery Workers         │
│   (Persistent)      (In-Memory)      (Background)           │
│                                                              │
│   All in Docker Containers                                  │
│   Orchestrated by Docker Compose                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## File Storage Strategy

```
media/
├── cases/
│   └── {case_id}/
│       ├── {modality}.nii.gz      (Original MRI)
│       ├── wt_mask.nii.gz         (Segmentation)
│       ├── tc_mask.nii.gz
│       ├── et_mask.nii.gz
│       ├── tumor_3d.gltf          (3D Model)
│       └── slices/
│           ├── axial_50.png
│           ├── coronal_50.png
│           └── sagittal_50.png
│
└── reports/
    └── pdfs/
        └── {report_id}_v{version}.pdf
```

---

**This architecture supports all SRS requirements while maintaining scalability, security, and performance.**
