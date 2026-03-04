# 🚀 Quick Start: Medical AI Dashboard

## ✅ Prerequisites (Already Done)

- ✅ Docker installed and running
- ✅ PostgreSQL running in Docker (port 5432)
- ✅ Redis running in Docker (port 6379)
- ✅ Backend and Frontend files in place
- ✅ Environment variables configured

## 📋 Step-by-Step Guide to Start the Dashboard

### **Step 1: Start the Backend (Django API)**

Open a **PowerShell terminal** and run:

```powershell
# Navigate to backend directory
cd "c:\Users\Farhan\Desktop\FYP\An-Expert-Guided-Multimodal-AI-Ecosystem\backend"

# Activate virtual environment (if exists)
.\.venv\Scripts\Activate.ps1

# If venv doesn't exist, create it first:
# python -m venv .venv
# Then activate it

# Install dependencies (first time only)
pip install -r requirements.txt

# Run database migrations (first time only)
python manage.py makemigrations
python manage.py migrate

# Create a superuser (first time only - optional but recommended)
python manage.py createsuperuser
# Follow prompts: username, email, password

# Start the Django development server
python manage.py runserver
```

**Expected Output:**
```
Starting development server at http://127.0.0.1:8000/
Quit the server with CTRL-BREAK.
```

✅ **Backend is now running on:** http://localhost:8000

---

### **Step 2: Start Celery Worker (Optional - For Async Tasks)**

Open a **new PowerShell terminal** (keep the backend running):

```powershell
# Navigate to backend directory
cd "c:\Users\Farhan\Desktop\FYP\An-Expert-Guided-Multimodal-AI-Ecosystem\backend"

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Start Celery worker
celery -A medical_ai_backend worker -l info --pool=solo
```

> **Note:** On Windows, use `--pool=solo` due to Windows limitations with Celery.

✅ **Celery worker is now processing async tasks**

---

### **Step 3: Start the Frontend (React Dashboard)**

Open **another new PowerShell terminal** (keep backend and celery running):

```powershell
# Navigate to frontend directory
cd "c:\Users\Farhan\Desktop\FYP\An-Expert-Guided-Multimodal-AI-Ecosystem\frontend"

# Install dependencies (first time only)
npm install

# Start the development server
npm run dev
```

**Expected Output:**
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

✅ **Frontend is now running on:** http://localhost:5173

---

## 🎯 Access the Dashboard

1. **Open your browser** and go to: **http://localhost:5173**
2. You should see the Medical AI Dashboard login/home page
3. **API Documentation** is available at: **http://localhost:8000/api/docs/**

---

## 🔑 Quick Commands Reference

### Check if Docker containers are running:
```powershell
docker ps
```

You should see:
- `medical_ai_db` (PostgreSQL on port 5432)
- `medical_ai_redis` (Redis on port 6379)

### Stop Docker containers (when done):
```powershell
docker stop medical_ai_db medical_ai_redis
```

### Restart Docker containers:
```powershell
docker start medical_ai_db medical_ai_redis
```

---

## 🛠️ Troubleshooting

### Problem: "Port already in use"
```powershell
# Find what's using the port (e.g., 8000)
netstat -ano | findstr :8000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

### Problem: Database connection error
```powershell
# Check if PostgreSQL container is running
docker ps | findstr medical_ai_db

# If not running, start it:
docker start medical_ai_db

# Check container logs:
docker logs medical_ai_db
```

### Problem: Redis connection error
```powershell
# Check if Redis container is running
docker ps | findstr medical_ai_redis

# If not running, start it:
docker start medical_ai_redis
```

### Problem: Frontend not connecting to backend
- Check that backend is running on port 8000
- Verify CORS settings in `backend/.env`:
  ```
  CORS_ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
  ```

---

## 📊 System Architecture

```
┌─────────────────────────────────────┐
│   Browser: localhost:5173           │
│   (React Frontend)                  │
└──────────────┬──────────────────────┘
               │ HTTP Requests
               ▼
┌─────────────────────────────────────┐
│   Django API: localhost:8000        │
│   (Backend REST API)                │
└──────┬──────────────┬───────────────┘
       │              │
       ▼              ▼
  ┌─────────┐   ┌──────────┐
  │ PostgreSQL│  │  Redis   │
  │  :5432   │   │  :6379   │
  └─────────┘   └──────────┘
       │              │
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │ Celery Worker│
       └──────────────┘
```

---

## 🎨 Dashboard Features

Once running, you can:

1. **Upload MRI scans** (T1, T1ce, T2, FLAIR)
2. **Run AI segmentation** using the MoME+ model
3. **View 3D visualizations** of brain tumors
4. **Generate clinical reports** with LLM
5. **Edit and export reports** to PDF
6. **Provide feedback** for continual learning
7. **Manage cases** and patient data

---

## 📝 Development Workflow

### Typical Development Session:

**Terminal 1 - Backend:**
```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python manage.py runserver
```

**Terminal 2 - Celery Worker:**
```powershell
cd backend
.\.venv\Scripts\Activate.ps1
celery -A medical_ai_backend worker -l info --pool=solo
```

**Terminal 3 - Frontend:**
```powershell
cd frontend
npm run dev
```

**Terminal 4 - Additional Commands:**
```powershell
# Run tests, manage database, etc.
```

---

## 🔒 Default Admin Access

After creating a superuser, access the Django admin panel:
- **URL:** http://localhost:8000/admin/
- **Username:** (what you set during `createsuperuser`)
- **Password:** (what you set during `createsuperuser`)

---

## 💡 Next Steps

1. **Test the API:** Visit http://localhost:8000/api/docs/
2. **Create a test user:** Register through the frontend
3. **Upload sample MRI:** Test the segmentation pipeline
4. **View generated reports:** Test LLM report generation
5. **Explore feedback system:** Submit clinician feedback

---

## 📞 Need Help?

- **Backend Issues:** Check `backend/medical_ai_backend/settings.py`
- **Frontend Issues:** Check `frontend/src/services/api.ts`
- **Database Issues:** Check `.env` file database settings
- **API Documentation:** http://localhost:8000/api/docs/

---

**⚡ Quick Start Command (One-liner for future use):**

Create a PowerShell script `start-all.ps1`:

```powershell
# Check Docker containers
docker start medical_ai_db medical_ai_redis

# Start backend in background
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; .\.venv\Scripts\Activate.ps1; python manage.py runserver"

# Start celery in background
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; .\.venv\Scripts\Activate.ps1; celery -A medical_ai_backend worker -l info --pool=solo"

# Start frontend in background
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev"

Write-Host "All services starting..."
Write-Host "Frontend: http://localhost:5173"
Write-Host "Backend: http://localhost:8000"
Write-Host "API Docs: http://localhost:8000/api/docs/"
```

Then just run: `.\start-all.ps1`

---

**Good luck with your Medical AI Dashboard! 🚀**
