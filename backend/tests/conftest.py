"""
Shared pytest fixtures for the Medical AI test suite.
Provides pre-created users, authenticated API clients, sample cases,
segmentation results, and report objects for reuse across tests.
"""

import uuid
import pytest
from io import BytesIO
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient
from django.contrib.auth import get_user_model

User = get_user_model()


def pytest_addoption(parser):
    parser.addoption(
        "--real-data", action="store_true", default=False, help="Run stress test with real MRI data"
    )


# ── Helper utilities ──────────────────────────────────────────────────

def make_fake_nifti() -> bytes:
    """Return a minimal NIfTI-1 file header (348 bytes) for upload tests."""
    header = bytearray(348)
    header[0:4] = (348).to_bytes(4, 'little')
    header[344:348] = b'n+1\x00'
    return bytes(header)


# ── User fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def doctor_user(db):
    """A doctor user."""
    return User.objects.create_user(
        email='doctor@fixture.com', password='TestPass123!',
        first_name='Alice', last_name='Smith', role='doctor',
        specialty='Neurology', institution='City Hospital'
    )


@pytest.fixture
def patient_user(db):
    """A patient user."""
    return User.objects.create_user(
        email='patient@fixture.com', password='TestPass123!',
        first_name='Bob', last_name='Jones', role='patient'
    )


@pytest.fixture
def admin_user(db):
    """An admin / staff user."""
    return User.objects.create_user(
        email='admin@fixture.com', password='TestPass123!',
        first_name='Carol', last_name='Admin', role='admin',
        is_staff=True
    )


@pytest.fixture
def researcher_user(db):
    """A researcher user."""
    return User.objects.create_user(
        email='researcher@fixture.com', password='TestPass123!',
        first_name='Dave', last_name='Research', role='researcher'
    )


# ── Authenticated client factory ──────────────────────────────────────

@pytest.fixture
def api_client():
    """An unauthenticated DRF API client."""
    return APIClient()


@pytest.fixture
def auth_client_factory(api_client):
    """
    Factory fixture: call with a user to get an APIClient
    authenticated as that user via JWT.
    """
    def _make(user):
        client = APIClient()
        resp = client.post('/api/users/login/', {
            'email': user.email, 'password': 'TestPass123!'
        }, format='json')
        assert resp.status_code == 200, f"Login failed for {user.email}: {resp.data}"
        token = resp.data['access']
        client.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')
        return client
    return _make


# ── Case fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def sample_case(db, doctor_user):
    """A basic Case created by the doctor."""
    from cases.models import Case
    return Case.objects.create(
        patient_id='PAT-FIXTURE',
        created_by=doctor_user,
        age=55,
        sex='M',
        clinical_history='Headaches for 2 months',
        indication='Rule out brain tumor'
    )


@pytest.fixture
def sample_case_with_mri(sample_case):
    """A Case with all 4 MRI modality records attached."""
    from cases.models import MRIImage
    for mod in ['t1', 't1ce', 't2', 'flair']:
        MRIImage.objects.create(
            case=sample_case,
            modality=mod,
            file_path=f'cases/{sample_case.case_id}/{mod}.nii',
            file_size=1024,
            original_filename=f'{mod}.nii'
        )
    return sample_case


@pytest.fixture
def sample_case_with_segmentation(sample_case):
    """A Case with a SegmentationResult attached."""
    from cases.models import SegmentationResult
    seg = SegmentationResult.objects.create(
        case=sample_case,
        whole_tumor_mask='cases/fake/wt.nii.gz',
        tumor_core_mask='cases/fake/tc.nii.gz',
        enhancing_tumor_mask='cases/fake/et.nii.gz',
        whole_tumor_volume=18000.0,
        tumor_core_volume=6000.0,
        enhancing_tumor_volume=1500.0,
        whole_tumor_confidence=0.91,
        tumor_core_confidence=0.87,
        enhancing_tumor_confidence=0.74,
        structured_findings={
            'volumes': {
                'whole_tumor': 18000.0,
                'tumor_core': 6000.0,
                'enhancing_tumor': 1500.0
            },
            'confidence_scores': {
                'whole_tumor': 0.91,
                'tumor_core': 0.87,
                'enhancing_tumor': 0.74
            },
            'full_segmentation_mask': 'cases/fake/full_seg.nii.gz',
            'timestamp': '2026-03-01T23:00:00',
            'model_version': 'MoME+ v1.0',
            'device': 'cpu'
        }
    )
    return sample_case, seg


@pytest.fixture
def sample_report(sample_case_with_segmentation, doctor_user):
    """A Report linked to a case with segmentation."""
    from reports.models import Report
    case, seg = sample_case_with_segmentation
    report = Report.objects.create(
        case=case,
        ai_generated_text='AI generated report text for testing purposes.',
        finalized_text='Finalized report text for testing purposes.',
        findings_json={
            'patient_info': {'case_id': str(case.case_id), 'age': 55, 'sex': 'M'},
            'tumor_metrics': {
                'volumes': {'whole_tumor': 18000.0, 'tumor_core': 6000.0, 'enhancing_tumor': 1500.0},
                'confidence_scores': {'whole_tumor': 0.91, 'tumor_core': 0.87, 'enhancing_tumor': 0.74}
            }
        },
        status='draft',
        last_edited_by=doctor_user
    )
    return report


@pytest.fixture
def fake_nifti_file():
    """A SimpleUploadedFile containing a minimal NIfTI-1 header."""
    return SimpleUploadedFile(
        'test.nii', make_fake_nifti(),
        content_type='application/octet-stream'
    )
