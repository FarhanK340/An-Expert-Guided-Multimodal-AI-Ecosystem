# Running & Testing the MedicalAI Dashboard

## Prerequisites

| Dependency | Version |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ |
| SQLite | (bundled, no setup needed) |

---

## 1. Start the Database (Docker)

PostgreSQL and Redis are configured in `docker-compose.yml`. For development,
run **only the DB services** in Docker (keeps hot-reload and avoids building
the heavy ML image):

```powershell
docker-compose up -d db redis
```

This starts:
- **PostgreSQL 15** on `localhost:5432` (user/pass: `postgres/postgres`, db: `medical_ai_db`)
- **Redis 7** on `localhost:6379`

Then create `backend/.env`:

```env
SECRET_KEY=some-random-secret-key
DEBUG=True
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/medical_ai_db
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
GEMINI_API_KEY=your-gemini-key   # optional
```

> **SQLite alternative** (no Docker needed): set `DATABASE_URL=sqlite:///db.sqlite3`

---

## 2. Start the Backend (Django)

```powershell
cd backend
.venv\Scripts\activate

# First-time only
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser   # admin account

# Start server
python manage.py runserver
```

Backend runs at **http://localhost:8000**  
API docs (Swagger): **http://localhost:8000/api/docs/**

---

## 2. Start the Frontend (React + Vite)

```powershell
cd frontend
npm install   # first time only
npm run dev
```

Frontend runs at **http://localhost:5173**

---

## 3. Role Definitions (Quick Reference)

| Role | Can Do |
|---|---|
| **Doctor / Radiologist** | Full workflow: create case → upload scans → run inference → generate & edit report → export PDF |
| **Patient** | View own cases and reports (read-only) |
| **Researcher** | View all cases/results (read-only); submit feedback |
| **Admin** | Manage users; view system stats; no clinical workflow |

---

## 4. Testing the Full Pipeline

### 4.1 Create Accounts

Go to **http://localhost:5173/signup** and create:

| Account | Email | Role |
|---|---|---|
| Doctor | `doctor@test.com` | Doctor |
| Patient | `patient@test.com` | Patient |

### 4.2 Doctor Workflow

1. **Login** as `doctor@test.com` → lands on Dashboard
2. **Cases → New Case** — fill in Patient ID (e.g. `PAT-001`), age, sex
3. On the case detail page, **upload all 4 MRI modalities** (T1, T1ce, T2, FLAIR)  
   - Use any `.nii.gz` test file (e.g. `BraTS-GLI-00006-100-seg.nii.gz` in repo root)
4. Once all 4 are uploaded, click **Run Prediction**  
   ⚠️ This requires the segmentation model checkpoint at `backend/models/checkpoints/mome_segmenter.pth`  
   Without it, the backend returns a `FileNotFoundError` — results page will show the error.
5. After inference completes → **View Results** shows volumes + confidence scores
6. Click **Generate Report** → AI report appears
7. **Reports** page lists the report
8. Click **View** → edit the report text inline → **Save**
9. Click **Export PDF** → downloads a PDF

### 4.3 Patient Workflow

1. **Login** as `patient@test.com`
2. Sidebar shows only: Dashboard, My Cases, My Reports, Settings
3. Patient sees **only cases where patient_id matches** their ID — they cannot create cases
4. Report is visible in read-only mode (no Edit button)

### 4.4 Admin Workflow

1. Go to **http://localhost:8000/admin** (Django admin panel)
2. Login with the superuser credentials
3. Manage users: change roles, deactivate accounts

---

## 5. Running Backend Tests

```powershell
cd backend
.venv\Scripts\activate
python -m pytest backend/tests/ -v
```

Or with Django test runner:

```powershell
python manage.py test users cases reports inference
```

### Test Coverage

| Test File | What It Tests |
|---|---|
| `backend/tests/test_auth.py` | Register, login, token, role guards |
| `backend/tests/test_pipeline.py` | Create case, upload, inference (mocked), report generation, PDF |

---

## 6. API Endpoints Quick Reference

```
POST  /api/users/register/          Register new user
POST  /api/users/login/             Login (returns JWT)
GET   /api/users/profile/           Get current user
PATCH /api/users/profile/update/    Update profile

GET   /api/cases/                   List cases
POST  /api/cases/                   Create case
POST  /api/cases/<id>/upload/       Upload MRI (modality=t1|t1ce|t2|flair)
DELETE /api/cases/<id>/delete/      Delete case

POST  /api/inference/predict/<id>/  Run segmentation
GET   /api/inference/result/<id>/   Get segmentation result

POST  /api/reports/generate/<id>/   Generate report from segmentation
GET   /api/reports/                 List reports
GET   /api/reports/<id>/            Get report
PATCH /api/reports/<id>/update/     Edit report
POST  /api/reports/<id>/export/     Download as PDF
```

---

## 7. Environment Variables (`.env` in `backend/`)

```env
SECRET_KEY=your-secret-key
DEBUG=True
DATABASE_URL=sqlite:///db.sqlite3
GEMINI_API_KEY=your-gemini-key   # optional — template fallback used if absent
```

If `GEMINI_API_KEY` is not set, report generation uses a built-in rule-based template. The pipeline still works end-to-end.

---

## 8. Common Issues

| Issue | Fix |
|---|---|
| `FileNotFoundError: mome_segmenter.pth` | Place trained model at `backend/models/checkpoints/mome_segmenter.pth` |
| `ModuleNotFoundError: scipy` | `pip install scipy` |
| `ModuleNotFoundError: reportlab` | `pip install reportlab` |
| CORS error in browser | Ensure backend is on port 8000, frontend on 5173 |
| `401 Unauthorized` on API calls | Token expired — logout and login again |
| Migration error | `python manage.py migrate --run-syncdb` |
