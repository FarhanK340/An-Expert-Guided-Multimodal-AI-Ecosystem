"""
Locust load testing configuration for the Medical AI Backend.

Run with:
    cd backend
    .venv\\Scripts\\locust.exe -f tests/locustfile.py --host=http://localhost:8000

Or headless mode:
    .venv\\Scripts\\locust.exe -f tests/locustfile.py --host=http://localhost:8000 --headless -u 50 -r 10 -t 60s
"""

import json
import uuid
from locust import HttpUser, task, between, events


class MedicalAIUser(HttpUser):
    """Simulates a doctor user interacting with the platform."""

    wait_time = between(1, 3)

    def on_start(self):
        """Register and login to get JWT token."""
        self.email = f"locust_{uuid.uuid4().hex[:8]}@test.com"
        self.password = "LocustTest123!"

        # Register
        reg_resp = self.client.post("/api/users/register/", json={
            "email": self.email,
            "password": self.password,
            "confirm_password": self.password,
            "first_name": "Locust",
            "last_name": "User",
            "role": "doctor"
        })

        if reg_resp.status_code == 201:
            data = reg_resp.json()
            self.token = data["access"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            # Fallback: try login
            login_resp = self.client.post("/api/users/login/", json={
                "email": self.email,
                "password": self.password
            })
            if login_resp.status_code == 200:
                data = login_resp.json()
                self.token = data["access"]
                self.headers = {"Authorization": f"Bearer {self.token}"}
            else:
                self.headers = {}

        self.case_ids = []

    @task(5)
    def list_cases(self):
        """GET /api/cases/ — most frequent operation."""
        self.client.get("/api/cases/", headers=self.headers)

    @task(3)
    def get_profile(self):
        """GET /api/users/profile/"""
        self.client.get("/api/users/profile/", headers=self.headers)

    @task(2)
    def create_case(self):
        """POST /api/cases/"""
        resp = self.client.post("/api/cases/", json={
            "patientId": f"PAT-{uuid.uuid4().hex[:6]}"
        }, headers=self.headers)

        if resp.status_code == 201:
            data = resp.json()
            self.case_ids.append(data.get("caseId"))

    @task(2)
    def list_reports(self):
        """GET /api/reports/"""
        self.client.get("/api/reports/", headers=self.headers)

    @task(1)
    def view_case_detail(self):
        """GET /api/cases/<id>/ — pick a random created case."""
        if self.case_ids:
            case_id = self.case_ids[-1]
            self.client.get(f"/api/cases/{case_id}/", headers=self.headers)

    @task(1)
    def health_check(self):
        """GET /api/health/"""
        self.client.get("/api/health/")


class PatientUser(HttpUser):
    """Simulates a patient user (read-heavy)."""

    wait_time = between(2, 5)
    weight = 2  # fewer patients than doctors

    def on_start(self):
        self.email = f"locust_pat_{uuid.uuid4().hex[:8]}@test.com"
        self.password = "LocustTest123!"

        reg_resp = self.client.post("/api/users/register/", json={
            "email": self.email,
            "password": self.password,
            "confirm_password": self.password,
            "first_name": "Patient",
            "last_name": "Locust",
            "role": "patient"
        })

        if reg_resp.status_code == 201:
            self.token = reg_resp.json()["access"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.headers = {}

    @task(5)
    def list_cases(self):
        self.client.get("/api/cases/", headers=self.headers)

    @task(3)
    def get_profile(self):
        self.client.get("/api/users/profile/", headers=self.headers)

    @task(2)
    def list_reports(self):
        self.client.get("/api/reports/", headers=self.headers)
