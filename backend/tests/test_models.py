"""
Unit tests for all Django models.
Tests model creation, methods, properties, constraints, and string representations.
"""

import pytest
from django.test import TestCase
from django.utils import timezone
from django.db import IntegrityError
from django.contrib.auth import get_user_model
from datetime import timedelta

User = get_user_model()


# ═══════════════════════════════════════════════════════════════════════
# User Model Tests
# ═══════════════════════════════════════════════════════════════════════

class TestUserModel(TestCase):
    """Tests for the custom User model."""

    def test_create_user_with_email(self):
        user = User.objects.create_user(
            email='test@example.com', password='Pass1234!',
            first_name='Test', last_name='User'
        )
        self.assertEqual(user.email, 'test@example.com')
        self.assertTrue(user.check_password('Pass1234!'))
        self.assertFalse(user.is_staff)
        self.assertFalse(user.is_superuser)
        self.assertTrue(user.is_active)

    def test_create_user_normalizes_email(self):
        user = User.objects.create_user(
            email='Test@EXAMPLE.COM', password='Pass1234!',
            first_name='A', last_name='B'
        )
        self.assertEqual(user.email, 'Test@example.com')

    def test_create_user_without_email_raises(self):
        with self.assertRaises(ValueError):
            User.objects.create_user(
                email='', password='Pass1234!',
                first_name='A', last_name='B'
            )

    def test_create_superuser(self):
        admin = User.objects.create_superuser(
            email='super@test.com', password='Admin1234!',
            first_name='Super', last_name='Admin'
        )
        self.assertTrue(admin.is_staff)
        self.assertTrue(admin.is_superuser)
        self.assertEqual(admin.role, 'admin')

    def test_create_superuser_without_staff_raises(self):
        with self.assertRaises(ValueError):
            User.objects.create_superuser(
                email='s@t.com', password='Pass1234!',
                first_name='A', last_name='B', is_staff=False
            )

    def test_create_superuser_without_superuser_raises(self):
        with self.assertRaises(ValueError):
            User.objects.create_superuser(
                email='s2@t.com', password='Pass1234!',
                first_name='A', last_name='B', is_superuser=False
            )

    def test_role_properties(self):
        doctor = User.objects.create_user(
            email='doc@r.com', password='P', first_name='D', last_name='R', role='doctor'
        )
        patient = User.objects.create_user(
            email='pat@r.com', password='P', first_name='P', last_name='R', role='patient'
        )
        researcher = User.objects.create_user(
            email='res@r.com', password='P', first_name='R', last_name='R', role='researcher'
        )
        admin = User.objects.create_user(
            email='adm@r.com', password='P', first_name='A', last_name='R', role='admin'
        )

        # Doctor
        self.assertTrue(doctor.is_doctor)
        self.assertTrue(doctor.is_clinician)
        self.assertFalse(doctor.is_patient)
        self.assertFalse(doctor.is_admin)

        # Patient
        self.assertTrue(patient.is_patient)
        self.assertFalse(patient.is_clinician)

        # Researcher
        self.assertTrue(researcher.is_researcher)
        self.assertTrue(researcher.is_clinician)

        # Admin
        self.assertTrue(admin.is_admin)
        self.assertFalse(admin.is_clinician)

    def test_get_full_name(self):
        user = User.objects.create_user(
            email='fn@t.com', password='P',
            first_name='John', last_name='Doe'
        )
        self.assertEqual(user.get_full_name(), 'John Doe')

    def test_get_short_name(self):
        user = User.objects.create_user(
            email='sn@t.com', password='P',
            first_name='Jane', last_name='Doe'
        )
        self.assertEqual(user.get_short_name(), 'Jane')

    def test_str_representation(self):
        user = User.objects.create_user(
            email='str@t.com', password='P',
            first_name='Alice', last_name='Smith'
        )
        self.assertEqual(str(user), 'Alice Smith (str@t.com)')

    def test_default_role_is_doctor(self):
        user = User.objects.create_user(
            email='def@t.com', password='P',
            first_name='D', last_name='R'
        )
        self.assertEqual(user.role, 'doctor')

    def test_duplicate_email_raises(self):
        User.objects.create_user(
            email='dup@t.com', password='P',
            first_name='A', last_name='B'
        )
        with self.assertRaises(IntegrityError):
            User.objects.create_user(
                email='dup@t.com', password='P',
                first_name='C', last_name='D'
            )


# ═══════════════════════════════════════════════════════════════════════
# Case Model Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCaseModel(TestCase):
    """Tests for the Case model."""

    def setUp(self):
        self.doctor = User.objects.create_user(
            email='doc_case@t.com', password='P',
            first_name='Dr', last_name='Case', role='doctor'
        )

    def test_case_creation_defaults(self):
        from cases.models import Case
        case = Case.objects.create(patient_id='PAT-001', created_by=self.doctor)
        self.assertEqual(case.status, 'created')
        self.assertIsNotNone(case.case_id)
        self.assertEqual(case.priority, 0)
        self.assertFalse(case.is_training_data)

    def test_case_uuid_is_unique(self):
        from cases.models import Case
        c1 = Case.objects.create(patient_id='PAT-A', created_by=self.doctor)
        c2 = Case.objects.create(patient_id='PAT-B', created_by=self.doctor)
        self.assertNotEqual(c1.case_id, c2.case_id)

    def test_complete_method(self):
        from cases.models import Case
        case = Case.objects.create(patient_id='PAT-CMP', created_by=self.doctor)
        case.complete()
        case.refresh_from_db()
        self.assertEqual(case.status, 'completed')
        self.assertIsNotNone(case.completed_at)

    def test_fail_method(self):
        from cases.models import Case
        case = Case.objects.create(patient_id='PAT-FAIL', created_by=self.doctor)
        case.fail('Something went wrong')
        case.refresh_from_db()
        self.assertEqual(case.status, 'failed')
        self.assertEqual(case.error_message, 'Something went wrong')
        self.assertIsNotNone(case.processing_ended_at)

    def test_processing_duration(self):
        from cases.models import Case
        case = Case.objects.create(patient_id='PAT-DUR', created_by=self.doctor)
        now = timezone.now()
        case.processing_started_at = now - timedelta(seconds=120)
        case.processing_ended_at = now
        case.save()
        self.assertAlmostEqual(case.processing_duration, 120.0, places=0)

    def test_processing_duration_none_when_incomplete(self):
        from cases.models import Case
        case = Case.objects.create(patient_id='PAT-INC', created_by=self.doctor)
        self.assertIsNone(case.processing_duration)

    def test_str_representation(self):
        from cases.models import Case
        case = Case.objects.create(patient_id='PAT-STR', created_by=self.doctor)
        self.assertIn('PAT-STR', str(case))

    def test_cascade_delete_on_user(self):
        """When a user is deleted, their cases are cascade-deleted."""
        from cases.models import Case
        case = Case.objects.create(patient_id='PAT-CASCADE', created_by=self.doctor)
        case_id = case.case_id
        self.doctor.delete()
        self.assertFalse(Case.objects.filter(case_id=case_id).exists())


# ═══════════════════════════════════════════════════════════════════════
# MRIImage Model Tests
# ═══════════════════════════════════════════════════════════════════════

class TestMRIImageModel(TestCase):
    """Tests for the MRIImage model."""

    def setUp(self):
        self.doctor = User.objects.create_user(
            email='doc_mri@t.com', password='P',
            first_name='Dr', last_name='MRI', role='doctor'
        )
        from cases.models import Case
        self.case = Case.objects.create(patient_id='PAT-MRI', created_by=self.doctor)

    def test_create_mri_image(self):
        from cases.models import MRIImage
        img = MRIImage.objects.create(
            case=self.case, modality='t1',
            file_path='cases/test/t1.nii',
            file_size=2048, original_filename='t1.nii'
        )
        self.assertEqual(img.modality, 't1')
        self.assertTrue(img.is_valid)

    def test_unique_together_constraint(self):
        from cases.models import MRIImage
        MRIImage.objects.create(
            case=self.case, modality='t1',
            file_path='cases/test/t1.nii',
            file_size=1024, original_filename='t1.nii'
        )
        with self.assertRaises(IntegrityError):
            MRIImage.objects.create(
                case=self.case, modality='t1',
                file_path='cases/test/t1_dup.nii',
                file_size=1024, original_filename='t1_dup.nii'
            )

    def test_cascade_delete_with_case(self):
        from cases.models import MRIImage
        MRIImage.objects.create(
            case=self.case, modality='flair',
            file_path='cases/test/flair.nii',
            file_size=1024, original_filename='flair.nii'
        )
        self.case.delete()
        self.assertEqual(MRIImage.objects.count(), 0)


# ═══════════════════════════════════════════════════════════════════════
# SegmentationResult Model Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSegmentationResultModel(TestCase):
    """Tests for the SegmentationResult model."""

    def setUp(self):
        self.doctor = User.objects.create_user(
            email='doc_seg@t.com', password='P',
            first_name='Dr', last_name='Seg', role='doctor'
        )
        from cases.models import Case
        self.case = Case.objects.create(patient_id='PAT-SEG', created_by=self.doctor)

    def test_create_segmentation_result(self):
        from cases.models import SegmentationResult
        seg = SegmentationResult.objects.create(
            case=self.case,
            whole_tumor_mask='wt.nii.gz',
            tumor_core_mask='tc.nii.gz',
            enhancing_tumor_mask='et.nii.gz',
            whole_tumor_volume=15000.0,
            tumor_core_volume=5000.0,
            enhancing_tumor_volume=1200.0,
            structured_findings={'test': True}
        )
        self.assertEqual(seg.case, self.case)
        self.assertEqual(seg.whole_tumor_volume, 15000.0)

    def test_str_representation(self):
        from cases.models import SegmentationResult
        seg = SegmentationResult.objects.create(
            case=self.case,
            whole_tumor_mask='wt.nii.gz',
            tumor_core_mask='tc.nii.gz',
            enhancing_tumor_mask='et.nii.gz',
            whole_tumor_volume=15000.0,
            tumor_core_volume=5000.0,
            enhancing_tumor_volume=1200.0,
            structured_findings={}
        )
        self.assertIn(str(self.case.case_id), str(seg))


# ═══════════════════════════════════════════════════════════════════════
# InferenceTask Model Tests
# ═══════════════════════════════════════════════════════════════════════

class TestInferenceTaskModel(TestCase):
    """Tests for the InferenceTask model."""

    def setUp(self):
        self.doctor = User.objects.create_user(
            email='doc_inf@t.com', password='P',
            first_name='Dr', last_name='Inf', role='doctor'
        )
        from cases.models import Case
        self.case = Case.objects.create(patient_id='PAT-INF', created_by=self.doctor)

    def test_start_method(self):
        from inference.models import InferenceTask
        task = InferenceTask.objects.create(
            case=self.case, task_type='segmentation',
            initiated_by=self.doctor
        )
        task.start()
        task.refresh_from_db()
        self.assertEqual(task.status, 'running')
        self.assertIsNotNone(task.started_at)

    def test_complete_method(self):
        from inference.models import InferenceTask
        task = InferenceTask.objects.create(
            case=self.case, task_type='segmentation',
            initiated_by=self.doctor
        )
        task.start()
        task.complete(result_data={'volumes': {'wt': 1000}})
        task.refresh_from_db()
        self.assertEqual(task.status, 'completed')
        self.assertEqual(task.progress_percentage, 100)
        self.assertIsNotNone(task.duration_seconds)
        self.assertIn('wt', task.result_data['volumes'])

    def test_fail_method(self):
        from inference.models import InferenceTask
        task = InferenceTask.objects.create(
            case=self.case, task_type='segmentation',
            initiated_by=self.doctor
        )
        task.start()
        task.fail('OOM error', traceback='Traceback ...')
        task.refresh_from_db()
        self.assertEqual(task.status, 'failed')
        self.assertEqual(task.error_message, 'OOM error')
        self.assertIn('Traceback', task.error_traceback)

    def test_update_progress(self):
        from inference.models import InferenceTask
        task = InferenceTask.objects.create(
            case=self.case, task_type='segmentation',
            initiated_by=self.doctor
        )
        task.update_progress(50, 'Preprocessing')
        task.refresh_from_db()
        self.assertEqual(task.progress_percentage, 50)
        self.assertEqual(task.current_step, 'Preprocessing')

    def test_progress_capped_at_100(self):
        from inference.models import InferenceTask
        task = InferenceTask.objects.create(
            case=self.case, task_type='segmentation',
            initiated_by=self.doctor
        )
        task.update_progress(150)
        task.refresh_from_db()
        self.assertEqual(task.progress_percentage, 100)


# ═══════════════════════════════════════════════════════════════════════
# ModelVersion Tests
# ═══════════════════════════════════════════════════════════════════════

class TestModelVersionModel(TestCase):
    """Tests for the ModelVersion model."""

    def test_activate_deactivates_siblings(self):
        from inference.models import ModelVersion
        v1 = ModelVersion.objects.create(
            name='MoME', version='1.0', model_type='segmentation',
            file_path='/models/v1.pth', file_size=100000,
            checksum='abc123', status='active'
        )
        v2 = ModelVersion.objects.create(
            name='MoME', version='2.0', model_type='segmentation',
            file_path='/models/v2.pth', file_size=200000,
            checksum='def456', status='testing'
        )
        v2.activate()
        v1.refresh_from_db()
        v2.refresh_from_db()
        self.assertEqual(v1.status, 'deprecated')
        self.assertEqual(v2.status, 'active')
        self.assertTrue(v2.is_default)


# ═══════════════════════════════════════════════════════════════════════
# Report Model Tests
# ═══════════════════════════════════════════════════════════════════════

class TestReportModel(TestCase):
    """Tests for the Report model."""

    def setUp(self):
        self.doctor = User.objects.create_user(
            email='doc_rpt@t.com', password='P',
            first_name='Dr', last_name='Rpt', role='doctor'
        )
        from cases.models import Case
        self.case = Case.objects.create(patient_id='PAT-RPT', created_by=self.doctor)

    def test_finalize(self):
        from reports.models import Report
        report = Report.objects.create(
            case=self.case,
            ai_generated_text='AI text',
            finalized_text='Final text',
            findings_json={'test': True}
        )
        report.finalize(self.doctor)
        report.refresh_from_db()
        self.assertEqual(report.status, 'finalized')
        self.assertIsNotNone(report.finalized_at)
        self.assertEqual(report.last_edited_by, self.doctor)

    def test_increment_edit_count(self):
        from reports.models import Report
        report = Report.objects.create(
            case=self.case,
            ai_generated_text='AI text',
            finalized_text='Final text',
            findings_json={'test': True}
        )
        self.assertEqual(report.edit_count, 0)
        report.increment_edit_count()
        report.refresh_from_db()
        self.assertEqual(report.edit_count, 1)
        report.increment_edit_count()
        report.refresh_from_db()
        self.assertEqual(report.edit_count, 2)

    def test_str_representation(self):
        from reports.models import Report
        report = Report.objects.create(
            case=self.case,
            ai_generated_text='AI text',
            finalized_text='Final text',
            findings_json={}
        )
        self.assertIn(str(report.report_id), str(report))
        self.assertIn(str(self.case.case_id), str(report))

    def test_cascade_delete_with_case(self):
        from reports.models import Report
        Report.objects.create(
            case=self.case,
            ai_generated_text='AI text',
            finalized_text='Final text',
            findings_json={}
        )
        self.case.delete()
        self.assertEqual(Report.objects.count(), 0)
