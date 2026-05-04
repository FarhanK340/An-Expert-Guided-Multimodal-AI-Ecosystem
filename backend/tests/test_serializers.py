"""
Unit tests for all serializers.
Tests camelCase field mapping, validation logic, and data transformations.
"""

from django.test import TestCase
from django.contrib.auth import get_user_model
from rest_framework.test import APIRequestFactory

User = get_user_model()


# ═══════════════════════════════════════════════════════════════════════
# UserSerializer Tests
# ═══════════════════════════════════════════════════════════════════════

class TestUserSerializer(TestCase):
    """Tests for UserSerializer camelCase output."""

    def setUp(self):
        self.user = User.objects.create_user(
            email='serial@test.com', password='Pass1234!',
            first_name='John', last_name='Doe', role='doctor',
            specialty='Neurology', institution='Hospital A',
            phone_number='+1234567890'
        )

    def test_camel_case_fields(self):
        from users.serializers import UserSerializer
        data = UserSerializer(self.user).data
        self.assertEqual(data['firstName'], 'John')
        self.assertEqual(data['lastName'], 'Doe')
        self.assertEqual(data['phoneNumber'], '+1234567890')
        self.assertEqual(data['role'], 'doctor')

    def test_username_from_email(self):
        from users.serializers import UserSerializer
        data = UserSerializer(self.user).data
        self.assertEqual(data['username'], 'serial')

    def test_read_only_fields(self):
        from users.serializers import UserSerializer
        data = UserSerializer(self.user).data
        self.assertIn('id', data)
        self.assertIn('createdAt', data)
        self.assertIn('updatedAt', data)
        self.assertIn('isEmailVerified', data)


# ═══════════════════════════════════════════════════════════════════════
# RegisterSerializer Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRegisterSerializer(TestCase):
    """Tests for RegisterSerializer validation and user creation."""

    def test_valid_registration(self):
        from users.serializers import RegisterSerializer
        data = {
            'email': 'newuser@test.com',
            'password': 'StrongPass123!',
            'confirm_password': 'StrongPass123!',
            'first_name': 'New',
            'last_name': 'User',
            'role': 'doctor'
        }
        serializer = RegisterSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        user = serializer.save()
        self.assertEqual(user.email, 'newuser@test.com')
        self.assertTrue(user.check_password('StrongPass123!'))

    def test_password_mismatch(self):
        from users.serializers import RegisterSerializer
        data = {
            'email': 'mismatch@test.com',
            'password': 'Pass1234!',
            'confirm_password': 'Different!',
            'first_name': 'A', 'last_name': 'B', 'role': 'doctor'
        }
        serializer = RegisterSerializer(data=data)
        self.assertFalse(serializer.is_valid())

    def test_duplicate_email_rejected(self):
        from users.serializers import RegisterSerializer
        User.objects.create_user(
            email='existing@test.com', password='P',
            first_name='E', last_name='X'
        )
        data = {
            'email': 'existing@test.com',
            'password': 'StrongPass123!',
            'confirm_password': 'StrongPass123!',
            'first_name': 'D', 'last_name': 'U', 'role': 'doctor'
        }
        serializer = RegisterSerializer(data=data)
        self.assertFalse(serializer.is_valid())

    def test_short_password_rejected(self):
        from users.serializers import RegisterSerializer
        data = {
            'email': 'short@test.com',
            'password': 'Ab1!',
            'confirm_password': 'Ab1!',
            'first_name': 'S', 'last_name': 'P', 'role': 'doctor'
        }
        serializer = RegisterSerializer(data=data)
        self.assertFalse(serializer.is_valid())


# ═══════════════════════════════════════════════════════════════════════
# UpdateProfileSerializer Tests
# ═══════════════════════════════════════════════════════════════════════

class TestUpdateProfileSerializer(TestCase):
    """Tests for UpdateProfileSerializer with role escalation prevention."""

    def setUp(self):
        self.doctor = User.objects.create_user(
            email='update@test.com', password='Pass1234!',
            first_name='Up', last_name='Date', role='doctor'
        )
        self.factory = APIRequestFactory()

    def test_admin_role_escalation_blocked(self):
        from users.serializers import UpdateProfileSerializer
        request = self.factory.patch('/')
        request.user = self.doctor
        serializer = UpdateProfileSerializer(
            self.doctor, data={'role': 'admin'}, partial=True,
            context={'request': request}
        )
        self.assertFalse(serializer.is_valid())

    def test_valid_partial_update(self):
        from users.serializers import UpdateProfileSerializer
        request = self.factory.patch('/')
        request.user = self.doctor
        serializer = UpdateProfileSerializer(
            self.doctor,
            data={'firstName': 'Updated', 'specialty': 'Oncology'},
            partial=True,
            context={'request': request}
        )
        self.assertTrue(serializer.is_valid(), serializer.errors)
        serializer.save()
        self.doctor.refresh_from_db()
        self.assertEqual(self.doctor.first_name, 'Updated')
        self.assertEqual(self.doctor.specialty, 'Oncology')


# ═══════════════════════════════════════════════════════════════════════
# ChangePasswordSerializer Tests
# ═══════════════════════════════════════════════════════════════════════

class TestChangePasswordSerializer(TestCase):
    """Tests for ChangePasswordSerializer."""

    def setUp(self):
        self.user = User.objects.create_user(
            email='chpw@test.com', password='OldPass123!',
            first_name='Ch', last_name='Pw'
        )
        self.factory = APIRequestFactory()

    def test_wrong_current_password(self):
        from users.serializers import ChangePasswordSerializer
        request = self.factory.post('/')
        request.user = self.user
        serializer = ChangePasswordSerializer(
            data={
                'currentPassword': 'WrongPassword!',
                'newPassword': 'NewPass123!',
                'confirmPassword': 'NewPass123!'
            },
            context={'request': request}
        )
        self.assertFalse(serializer.is_valid())

    def test_new_password_mismatch(self):
        from users.serializers import ChangePasswordSerializer
        request = self.factory.post('/')
        request.user = self.user
        serializer = ChangePasswordSerializer(
            data={
                'currentPassword': 'OldPass123!',
                'newPassword': 'NewPass123!',
                'confirmPassword': 'DifferentPass!'
            },
            context={'request': request}
        )
        self.assertFalse(serializer.is_valid())

    def test_valid_password_change(self):
        from users.serializers import ChangePasswordSerializer
        request = self.factory.post('/')
        request.user = self.user
        serializer = ChangePasswordSerializer(
            data={
                'currentPassword': 'OldPass123!',
                'newPassword': 'NewPass123!',
                'confirmPassword': 'NewPass123!'
            },
            context={'request': request}
        )
        self.assertTrue(serializer.is_valid(), serializer.errors)


# ═══════════════════════════════════════════════════════════════════════
# CaseSerializer Tests
# ═══════════════════════════════════════════════════════════════════════

class TestCaseSerializer(TestCase):
    """Tests for CaseSerializer camelCase fields and patient linking."""

    def setUp(self):
        self.doctor = User.objects.create_user(
            email='doc_cs@test.com', password='P',
            first_name='Dr', last_name='Case', role='doctor'
        )
        self.patient = User.objects.create_user(
            email='pat_cs@test.com', password='P',
            first_name='Pat', last_name='Ient', role='patient'
        )

    def test_camel_case_output(self):
        from cases.models import Case
        from cases.serializers import CaseSerializer
        case = Case.objects.create(patient_id='PAT-CS', created_by=self.doctor, age=40, sex='F')
        data = CaseSerializer(case).data
        self.assertEqual(data['patientId'], 'PAT-CS')
        self.assertIn('caseId', data)
        self.assertIn('createdAt', data)
        self.assertIn('createdBy', data)
        self.assertEqual(data['createdBy'], 'Dr Case')

    def test_patient_email_links_patient(self):
        from cases.models import Case
        from cases.serializers import CaseSerializer
        data = {
            'patientId': 'PAT-LINK',
            'patientEmail': 'pat_cs@test.com'
        }
        serializer = CaseSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        case = serializer.save(created_by=self.doctor)
        case.refresh_from_db()
        self.assertEqual(case.patient_user, self.patient)

    def test_patient_email_nonexistent_ignored(self):
        from cases.serializers import CaseSerializer
        data = {
            'patientId': 'PAT-NOLINK',
            'patientEmail': 'nobody@test.com'
        }
        serializer = CaseSerializer(data=data)
        self.assertTrue(serializer.is_valid(), serializer.errors)
        case = serializer.save(created_by=self.doctor)
        case.refresh_from_db()
        self.assertIsNone(case.patient_user)

    def test_patient_user_email_read(self):
        from cases.models import Case
        from cases.serializers import CaseSerializer
        case = Case.objects.create(
            patient_id='PAT-REMAIL', created_by=self.doctor,
            patient_user=self.patient
        )
        data = CaseSerializer(case).data
        self.assertEqual(data['patientUserEmail'], 'pat_cs@test.com')


# ═══════════════════════════════════════════════════════════════════════
# ReportSerializer Tests
# ═══════════════════════════════════════════════════════════════════════

class TestReportSerializer(TestCase):
    """Tests for ReportSerializer camelCase output."""

    def setUp(self):
        self.doctor = User.objects.create_user(
            email='doc_rs@test.com', password='P',
            first_name='Dr', last_name='Rpt', role='doctor'
        )

    def test_camel_case_output(self):
        from cases.models import Case
        from reports.models import Report
        from reports.serializers import ReportSerializer
        case = Case.objects.create(patient_id='PAT-RS', created_by=self.doctor)
        report = Report.objects.create(
            case=case,
            ai_generated_text='AI text',
            finalized_text='Final text',
            findings_json={'test': True},
            last_edited_by=self.doctor
        )
        data = ReportSerializer(report).data
        self.assertIn('reportId', data)
        self.assertIn('caseId', data)
        self.assertIn('aiGeneratedText', data)
        self.assertIn('finalizedText', data)
        self.assertIn('findingsJson', data)
        self.assertIn('editCount', data)
        self.assertEqual(data['lastEditedBy'], 'Dr Rpt')

    def test_last_edited_by_none(self):
        from cases.models import Case
        from reports.models import Report
        from reports.serializers import ReportSerializer
        case = Case.objects.create(patient_id='PAT-RSNULL', created_by=self.doctor)
        report = Report.objects.create(
            case=case,
            ai_generated_text='AI text',
            finalized_text='Final text',
            findings_json={}
        )
        data = ReportSerializer(report).data
        self.assertIsNone(data['lastEditedBy'])


# ═══════════════════════════════════════════════════════════════════════
# ReportUpdateSerializer Tests
# ═══════════════════════════════════════════════════════════════════════

class TestReportUpdateSerializer(TestCase):
    """Tests for ReportUpdateSerializer validation."""

    def test_valid_update(self):
        from reports.serializers import ReportUpdateSerializer
        serializer = ReportUpdateSerializer(data={
            'finalizedText': 'Updated report text',
            'editReason': 'Corrected findings'
        })
        self.assertTrue(serializer.is_valid())

    def test_missing_text_invalid(self):
        from reports.serializers import ReportUpdateSerializer
        serializer = ReportUpdateSerializer(data={'editReason': 'No text'})
        self.assertFalse(serializer.is_valid())

    def test_edit_reason_optional(self):
        from reports.serializers import ReportUpdateSerializer
        serializer = ReportUpdateSerializer(data={'finalizedText': 'Text only'})
        self.assertTrue(serializer.is_valid())
