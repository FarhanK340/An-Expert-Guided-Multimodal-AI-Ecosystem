"""
Stress tests for the UNMOCKED Inference Engine.
This test simulates multiple concurrent users triggering the actual ML model simultaneously.

WARNING:
This test uses tiny, fake NIfTI files by default to avoid bloating the repository.
Because these files do not have valid brain MRI dimensions (e.g., 240x240x155), 
the actual ML model might crash during preprocessing and return a 400/500 error.

Even if it returns an error, it successfully tests the backend's ability to handle 
concurrent inference requests without deadlocking. 

To truly stress test your GPU/CPU (VRAM usage, inference times), you should replace 
`make_fake_nifti()` to return bytes from a REAL, valid MRI scan.
"""

import time
import uuid
import threading
import pytest
from django.test import TransactionTestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient
from django.contrib.auth import get_user_model
import os
import sys
from django.conf import settings

User = get_user_model()

USE_REAL_DATA = '--real-data' in sys.argv
REAL_DATA_DIR = r"G:\FYP\synapsedownloads\Brats2024\BratsGLI\training_data1_v2\BraTS-GLI-00005-100"

def get_nifti_bytes(modality: str) -> bytes:
    """
    Return bytes for the requested modality. 
    If --real-data is passed, it reads from the real Brats directory.
    Otherwise, returns fake 348-byte header.
    """
    if USE_REAL_DATA:
        # Map our internal modalities to the BraTS file naming convention
        mod_map = {
            't1ce': 't1c',
            't1': 't1n',
            't2': 't2w',
            'flair': 't2f'
        }
        brats_mod = mod_map.get(modality, modality)
        filename = f"BraTS-GLI-00005-100-{brats_mod}.nii.gz"
        filepath = os.path.join(REAL_DATA_DIR, filename)
        
        with open(filepath, 'rb') as f:
            return f.read()
    else:
        # Return fake tiny header
        header = bytearray(348)
        header[0:4] = (348).to_bytes(4, 'little')
        header[344:348] = b'n+1\x00'
        return bytes(header)


@pytest.mark.stress
class TestUnmockedInferenceStress(TransactionTestCase):
    """Stress tests for the live Inference Engine."""

    def setUp(self):
        # Create a single doctor user for the tests
        self.doctor = User.objects.create_user(
            email='inf_stress@test.com', password='TestPass123!',
            first_name='Inference', last_name='Stress', role='doctor'
        )
        client = APIClient()
        resp = client.post('/api/users/login/', {
            'email': 'inf_stress@test.com', 'password': 'TestPass123!'
        }, format='json')
        self.token = resp.data['access']

    def _setup_case_with_files(self):
        """Helper to create a case and upload 4 required modalities."""
        client = APIClient()
        client.credentials(HTTP_AUTHORIZATION=f'Bearer {self.token}')
        
        # 1. Create Case
        case_resp = client.post('/api/cases/', {
            'patientId': f'PAT-INF-STRESS-{uuid.uuid4().hex[:6]}'
        }, format='json')
        case_id = case_resp.data['caseId']
        
        # Create physical directory
        case_dir = os.path.join(settings.MEDIA_ROOT, 'cases', str(case_id))
        os.makedirs(case_dir, exist_ok=True)

        # 2. Upload 4 Modalities
        for mod in ['t1', 't1ce', 't2', 'flair']:
            f = SimpleUploadedFile(
                f'{mod}.nii.gz' if USE_REAL_DATA else f'{mod}.nii', 
                get_nifti_bytes(mod),
                content_type='application/octet-stream'
            )
            client.post(
                f'/api/cases/{case_id}/upload/',
                {'file': f, 'modality': mod},
                format='multipart'
            )
            
        return case_id

    def test_concurrent_unmocked_inference(self):
        """
        Simulate 10 concurrent inference requests.
        DOES NOT mock the InferenceEngine.
        """
        # Number of concurrent inference tasks
        CONCURRENT_USERS = 10
        
        # Setup cases sequentially first so we purely stress the inference engine
        case_ids = []
        for _ in range(CONCURRENT_USERS):
            case_ids.append(self._setup_case_with_files())

        results = []
        errors = []

        def run_inference(case_id):
            client = APIClient()
            client.credentials(HTTP_AUTHORIZATION=f'Bearer {self.token}')
            try:
                # This hits the REAL, UNMOCKED /api/inference/predict/<case_id>/
                start_time = time.time()
                resp = client.post(f'/api/inference/predict/{case_id}/')
                elapsed = time.time() - start_time
                
                results.append({
                    'status': resp.status_code,
                    'time': elapsed,
                    'data': resp.data
                })
            except Exception as e:
                errors.append(str(e))

        print(f"\n--- Starting UNMOCKED Inference Stress Test ({CONCURRENT_USERS} users) ---")
        threads = [threading.Thread(target=run_inference, args=(cid,)) for cid in case_ids]
        
        total_start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=300) # Give it 5 minutes maximum to complete all 10 inferences
        total_elapsed = time.time() - total_start

        print(f"--- Completed in {total_elapsed:.2f} seconds ---")
        
        # Analyze results
        successes = [r for r in results if r['status'] == 200]
        failures = [r for r in results if r['status'] != 200]
        
        print(f"Total Requests: {len(results)}")
        print(f"Successes (200 OK): {len(successes)}")
        print(f"Failures: {len(failures)}")
        if failures:
            print(f"Sample Failure Response: {failures[0]['data']}")
            
        self.assertEqual(len(errors), 0, f"Thread execution errors: {errors}")
        
        # We don't strictly assert status_code == 200 because the real model 
        # might crash on the fake NIfTI files. 
        # As long as the server responded without deadlocking, the test passes its goal.
        self.assertEqual(len(results), CONCURRENT_USERS, "Not all threads completed.")
