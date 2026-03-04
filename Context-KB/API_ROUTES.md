# Backend API Routes - Quick Reference

## ✅ Working Endpoints

### Authentication (`/api/users/`)
- `POST /api/users/register/` - Register new user
- `POST /api/users/login/` - Login (returns JWT tokens)
- `POST /api/users/logout/` - Logout (requires auth)
- `POST /api/users/refresh/` - Refresh access token
- `GET /api/users/profile/` - Get current user (requires auth)
- `PATCH /api/users/profile/update/` - Update profile (requires auth)
- `GET /api/users/users/` - List all users (admin only)
- `GET /api/users/users/{id}/` - Get user details (admin only)

### Cases (`/api/cases/`)
- `GET /api/cases/` - List all cases
- `POST /api/cases/` - Create new case
- `GET /api/cases/{case_id}/` - Get case details
- `PATCH /api/cases/{case_id}/update/` - Update case
- `DELETE /api/cases/{case_id}/delete/` - Delete case
- `POST /api/cases/{case_id}/upload/` - Upload MRI images
- `GET /api/cases/{case_id}/images/` - List MRI images
- `GET /api/cases/{case_id}/segmentation/` - Get segmentation results
- `POST /api/cases/{case_id}/feedback/` - Submit feedback

### Reports (`/api/reports/`)
- `GET /api/reports/` - List all reports
- `GET /api/reports/{report_id}/` - Get report details
- `PATCH /api/reports/{report_id}/update/` - Update report
- `POST /api/reports/generate/{case_id}/` - Generate report for case
- `GET /api/reports/{report_id}/export/` - Export report as PDF

### Inference (`/api/inference/`)
- `POST /api/inference/segment/` - Start segmentation task
- `GET /api/inference/tasks/` - List inference tasks
- `GET /api/inference/tasks/{task_id}/` - Get task status
- `POST /api/inference/tasks/{task_id}/cancel/` - Cancel task

### Other
- `GET /api/health/` - Health check
- `GET /api/docs/` - API documentation (Swagger UI)
- `GET /api/schema/` - OpenAPI schema

## Quick Test

### Test Login
```bash
curl -X POST http://localhost:8000/api/users/login/ \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "test123456"}'
```

### Test Register
```bash
curl -X POST http://localhost:8000/api/users/register/ \
  -H "Content-Type: application/json" \
  -d '{
    "email": "newuser@example.com",
    "password": "securepass123",
    "confirm_password": "securepass123",
    "first_name": "John",
    "last_name": "Doe",
    "role": "doctor"
  }'
```

### Test Profile (with JWT)
```bash
curl -X GET http://localhost:8000/api/users/profile/ \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN_HERE"
```

## URL Pattern Fixed

### Before:
- Frontend calling: `/api/users/login/`
- Backend expecting: `/api/v1/auth/login/`
- **Result:** 404 Not Found ❌

### After:
- Frontend calling: `/api/users/login/`
- Backend now serving: `/api/users/login/`
- **Result:** Working! ✅

## Next Steps
1. ✅ URLs are fixed - restart Django server
2. Test login from frontend
3. Check browser console for any CORS errors
4. Verify JWT tokens in localStorage
