# NeuroAI Backend - Django REST API

Django-based REST API backend for the Expert-Guided Multimodal AI Ecosystem, providing authentication, case management, MRI image handling, and AI inference orchestration for brain tumor segmentation.

## Features

- **User Management**: JWT-based authentication with role-based access control
- **Case Management**: CRUD operations for medical cases with patient information
- **MRI Image Handling**: Upload, storage, and retrieval of multi-modal MRI scans (T1, T2, T1CE, FLAIR)
- **Inference Orchestration**: Asynchronous AI segmentation jobs using Celery
- **Report Generation**: Medical report creation and management
- **RESTful API**: Comprehensive REST API with DRF (Django REST Framework)
- **Admin Interface**: Django admin panel for system management
- **Health Checks**: API health monitoring endpoints
- **Database**: PostgreSQL support with SQLite fallback for development
- **File Storage**: Local and cloud storage (S3/GCS) support

## Prerequisites

- Python 3.12+
- PostgreSQL 14+ (optional, SQLite works for development)
- Redis (for Celery task queue)
- CUDA-compatible GPU (optional, for AI inference)

## Installation

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```

   Key environment variables:
   ```env
   # Django Settings
   DEBUG=True
   SECRET_KEY=your-secret-key-here
   ALLOWED_HOSTS=localhost,127.0.0.1
   
   # Database
   DATABASE_URL=sqlite:///db.sqlite3
   # Or for PostgreSQL: postgresql://user:password@localhost:5432/medical_ai_db
   
   # JWT Settings
   JWT_ACCESS_TOKEN_LIFETIME=60
   JWT_REFRESH_TOKEN_LIFETIME=1440
   
   # Celery
   CELERY_BROKER_URL=redis://localhost:6379/0
   CELERY_RESULT_BACKEND=redis://localhost:6379/0
   
   # File Upload
   MEDIA_ROOT=./media
   MAX_UPLOAD_SIZE=1073741824
   
   # AI Model Paths
   SEGMENTATION_MODEL_PATH=./models/mome_plus.pth
   LLM_MODEL_PATH=./models/medalpaca-7b
   MODEL_DEVICE=cuda
   ```

5. **Run database migrations:**
   ```bash
   python manage.py migrate
   ```

6. **Create a superuser (admin):**
   ```bash
   python manage.py createsuperuser
   ```

7. **Collect static files (for production):**
   ```bash
   python manage.py collectstatic --noinput
   ```

## Running the Application

### Development Server
Start the Django development server:
```bash
python manage.py runserver
```

The API will be available at `http://localhost:8000/api/`

### Celery Worker (for async tasks)
In a separate terminal, start the Celery worker:
```bash
celery -A medical_ai_backend worker -l info
```

### Celery Beat (for scheduled tasks)
For periodic tasks, start Celery Beat:
```bash
celery -A medical_ai_backend beat -l info
```

### Redis Server
Ensure Redis is running:
```bash
# On Windows (using WSL or native Redis)
redis-server

# On macOS (via Homebrew)
brew services start redis

# On Linux
sudo systemctl start redis
```

## Project Structure

```
backend/
├── medical_ai_backend/     # Main project configuration
│   ├── settings.py         # Django settings
│   ├── urls.py             # Root URL configuration
│   ├── celery.py           # Celery configuration
│   ├── exceptions.py       # Custom exception handlers
│   ├── health_views.py     # Health check endpoints
│   └── wsgi.py/asgi.py     # WSGI/ASGI entry points
├── users/                  # User management app
│   ├── models.py           # User model with roles
│   ├── serializers.py      # User serializers
│   ├── views.py            # Auth and profile endpoints
│   └── urls.py             # User routes
├── cases/                  # Case management app
│   ├── models.py           # Case and MRIImage models
│   ├── serializers.py      # Case serializers
│   ├── mri_serializers.py  # MRI image serializers
│   ├── views.py            # Case CRUD endpoints
│   ├── urls.py             # Case routes
│   └── management/         # Django management commands
│       └── commands/
│           ├── populate_sample_data.py
│           └── add_user_cases.py
├── inference/              # AI inference app
│   ├── models.py           # Inference job tracking
│   ├── views.py            # Inference endpoints
│   └── urls.py             # Inference routes
├── reports/                # Report generation app
│   ├── models.py           # Report model
│   ├── views.py            # Report endpoints
│   └── urls.py             # Report routes
├── media/                  # Uploaded files (MRI scans)
├── logs/                   # Application logs
├── requirements.txt        # Python dependencies
├── manage.py               # Django management script
└── .env                    # Environment variables (not in git)
```

## Database Models

### User Model
- Email-based authentication
- Roles: Doctor, Radiologist, Admin
- Profile fields: specialty, institution, phone number

### Case Model
- Patient information (anonymized ID, age, gender, diagnosis)
- Status tracking (Pending, In Progress, Completed)
- Related user (created_by)
- Timestamps

### MRIImage Model
- Multi-modal support (T1, T2, T1CE, FLAIR)
- File storage with metadata
- Linked to cases

### Inference Model
- Segmentation job tracking
- Status monitoring
- Results storage

### Report Model
- Generated medical reports
- Linked to cases and inferences

## API Endpoints

### Authentication
- `POST /api/users/register/` - User registration
- `POST /api/users/login/` - User login (returns JWT tokens)
- `POST /api/users/logout/` - User logout
- `POST /api/users/refresh/` - Refresh access token

### User Profile
- `GET /api/users/profile/` - Get current user profile
- `PATCH /api/users/profile/update/` - Update profile
- `POST /api/users/profile/change-password/` - Change password

### Admin
- `GET /api/users/users/` - List all users (admin only)

### Cases
- `GET /api/cases/` - List all cases
- `POST /api/cases/` - Create new case
- `GET /api/cases/{id}/` - Get case details
- `PATCH /api/cases/{id}/update/` - Update case
- `DELETE /api/cases/{id}/delete/` - Delete case

### MRI Images
- `POST /api/cases/{id}/upload/` - Upload MRI image
- `GET /api/cases/{id}/images/` - List case MRI images

### Inference (Placeholder)
- `POST /api/inference/` - Start segmentation job
- `GET /api/inference/{id}/` - Get inference status
- `GET /api/inference/` - List all inferences

### Reports (Placeholder)
- `POST /api/reports/` - Generate report
- `GET /api/reports/{id}/` - Get report details
- `GET /api/reports/` - List all reports

### Health
- `GET /api/health/` - API health check

## Authentication

The API uses JWT (JSON Web Tokens) for authentication:

1. **Login**: Send credentials to `/api/users/login/` to receive `access` and `refresh` tokens
2. **Protected Requests**: Include access token in header: `Authorization: Bearer <access_token>`
3. **Token Refresh**: When access token expires, use refresh token at `/api/users/refresh/`
4. **Logout**: Send refresh token to `/api/users/logout/` to blacklist it

## Configuration

### Key Settings (in `settings.py`)

- **CORS**: Configured for frontend at `http://localhost:5173`
- **JWT Expiration**: Access token (60 min), Refresh token (24 hours)
- **File Upload**: Max 1GB per file
- **Celery**: Redis broker for async tasks
- **REST Framework**: Pagination, authentication, and permissions

### Management Commands

Populate sample data:
```bash
python manage.py populate_sample_data
```

Add cases for specific user:
```bash
python manage.py add_user_cases <user_id>
```

## Development

### Running Tests
```bash
python manage.py test
```

### Django Shell
Access Django shell for debugging:
```bash
python manage.py shell
```

### Database Shell
```bash
python manage.py dbshell
```

### Create Migrations
After model changes:
```bash
python manage.py makemigrations
python manage.py migrate
```

## Docker Deployment

Build and run with Docker:
```bash
docker build -t medical-ai-backend .
docker run -p 8000:8000 --env-file .env medical-ai-backend
```

Or use Docker Compose (if `docker-compose.yml` exists):
```bash
docker-compose up --build
```

## Monitoring & Logging

Logs are stored in `logs/` directory:
- `app.log` - Application logs
- `celery.log` - Celery task logs

View logs in real-time:
```bash
tail -f logs/app.log
```

## Future Enhancements

- **AI Inference Pipeline**: Complete integration with MoME+ segmentation model
  - Celery task for async inference
  - Model loading and caching
  - Result storage and visualization data export
  
- **LLM Report Generation**: Integration with fine-tuned MedAlpaca/MedGemma
  - JSON schema-based input from segmentation results
  - Structured medical report generation
  - Factual consistency verification

- **Advanced Image Processing**: 
  - DICOM format support
  - Image preprocessing pipelines
  - Multi-sequence registration

- **Feedback Loop**: 
  - Clinician feedback collection
  - Model retraining with expert annotations
  - Continual learning integration

- **Cloud Storage Integration**:
  - AWS S3 / Google Cloud Storage for MRI files
  - CDN for fast image delivery

- **GraphQL API**: Alternative to REST for complex queries

- **WebSocket Support**: Real-time updates for inference progress

- **Advanced Analytics**:
  - Case statistics and trends
  - Model performance metrics
  - User activity tracking

- **Audit Logging**: Complete audit trail for compliance

- **Multi-tenancy**: Support for multiple institutions

- **DICOM Viewer Integration**: Server-side DICOM processing

- **Automated Testing**: Comprehensive test suite with 80%+ coverage

## License

Part of the Expert-Guided Multimodal AI Ecosystem project.

## Contributing

This is an academic project. For contributions, please contact the project maintainers.

## Troubleshooting

### Port already in use
```bash
# Find and kill process using port 8000
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:8000 | xargs kill -9
```

### Database connection errors
- Ensure PostgreSQL is running (if using PostgreSQL)
- Verify `DATABASE_URL` in `.env`
- Try SQLite for development: `DATABASE_URL=sqlite:///db.sqlite3`

### Celery connection errors
- Ensure Redis is running
- Verify `CELERY_BROKER_URL` in `.env`

### Import errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
