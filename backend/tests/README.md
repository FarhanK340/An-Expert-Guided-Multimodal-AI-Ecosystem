# Medical AI Backend Testing Infrastructure

This directory contains the comprehensive testing infrastructure for the Medical AI Backend. The suite is designed to ensure the reliability, security, and performance of the core business logic, ML inference pipelines, report generation, and role-based access controls.

## Prerequisites

Before running any tests, ensure your virtual environment is activated and you have installed the testing dependencies:

```powershell
# Navigate to the backend directory
cd backend

# Activate your virtual environment (Windows)
.\.venv\Scripts\Activate.ps1

# Ensure dependencies are installed
pip install -r requirements.txt
```

---

## Overview of Test Suites

The testing infrastructure is broken down into several modular tiers:

1. **Unit Tests**: Fast, isolated tests for database models (`test_models.py`), serializers (`test_serializers.py`), and simple endpoints (`test_health.py`).
2. **Integration Tests**: Tests that verify the interaction between components, specifically covering authentication pipelines (`test_auth.py`), the inference/report generation logic (`test_pipeline.py`), and the comprehensive Role-Based Access Control matrix (`test_rbac_matrix.py`).
3. **End-to-End (E2E) System Tests**: Tests that mimic full user journeys across the application, validating end-to-end flows like patient registration and doctor case management (`test_e2e_workflow.py`).
4. **Stress Tests**: Pytest-based concurrency tests evaluating the database and API behavior under multi-threaded load (`test_stress.py`), as well as a specialized script to test the physical ML inference engine concurrency without mocking (`test_inference_stress.py`).
5. **Load Tests**: Locust-based performance tests generating continuous HTTP traffic to simulate real-world usage (`locustfile.py`).

*Note: The ML inference engine and LLM generation endpoints are heavily mocked during tests to prevent long execution times and unnecessary disk/API usage.*

---

## Running the Tests

Django's test runner (`pytest-django`) automatically spins up a temporary, isolated, in-memory SQLite database specifically for testing. **You do NOT need to have the backend server running to run the standard Pytest suites.**

### Running All Standard Tests
To run the entire suite (excluding the intensive stress tests):
```powershell
python -m pytest tests/ -v --tb=short -m "not stress"
```

### Running Tests by Category
You can target specific modules to speed up development workflows:

**Run only Unit Tests:**
```powershell
python -m pytest tests/test_models.py tests/test_serializers.py tests/test_health.py -v
```

**Run only Integration Tests:**
```powershell
python -m pytest tests/test_auth.py tests/test_pipeline.py tests/test_rbac_matrix.py -v
```

**Run only System / E2E Tests:**
```powershell
python -m pytest tests/test_e2e_workflow.py -v
```

### Running Stress & Load Tests

**1. Concurrency Stress Tests (Pytest)**
These tests use Python threading to hammer specific endpoints and check for race conditions. No external servers needed.
```powershell
python -m pytest tests/test_stress.py -v -m stress
```

**2. Unmocked ML Inference Stress Tests**
This specialized script uploads 40 NIfTI files (4 per simulated user) and triggers the actual PyTorch inference engine simultaneously across 10 threads to monitor VRAM and model load behavior.

To run with *fake data* (tests concurrency/deadlocking without heavy GPU load):
```powershell
python -m pytest tests/test_inference_stress.py -v -s
```

To run with *real data* (tests physical GPU memory utilization and processing speed):
```powershell
python -m pytest tests/test_inference_stress.py -v -s --real-data
```
*(Ensure `REAL_DATA_DIR` inside the script points to a valid BraTS dataset directory before running).*

**3. API Load Tests (Locust)**
Locust generates continuous HTTP traffic to test your server's capacity. **You MUST have the backend development server running in a separate terminal before starting Locust.**

First, in Terminal A:
```powershell
python manage.py runserver
```

Then, in Terminal B (Locust Headless Mode):
```powershell
locust -f tests/locustfile.py --host=http://localhost:8000 --headless -u 50 -r 10 -t 30s
```
*(This simulates 50 concurrent users, spawning 10 new users per second, for 30 seconds).*

If you prefer to use the interactive Locust Web UI, drop the headless flags:
```powershell
locust -f tests/locustfile.py
```
Then navigate to `http://localhost:8089` in your browser to configure and launch the swarm manually.
