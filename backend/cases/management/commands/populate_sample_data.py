"""
Management command to populate sample cases for testing and demonstration.
"""

from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from cases.models import Case
from datetime import date, timedelta
from django.utils import timezone

User = get_user_model()


class Command(BaseCommand):
    help = 'Populate database with sample case data'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Creating sample cases...'))

        # Get or create a user to own the cases
        user, created = User.objects.get_or_create(
            email='demo@hospital.com',
            defaults={
                'first_name': 'Demo',
                'last_name': 'Doctor',
                'role': 'doctor',
                'specialty': 'Neurology',
                'institution': 'Demo Hospital',
            }
        )
        
        if created:
            user.set_password('demo123456')
            user.save()
            self.stdout.write(self.style.SUCCESS(f'Created demo user: {user.email}'))

        # Clear existing sample cases
        Case.objects.filter(patient_id__startswith='DEMO-').delete()

        # Sample cases data
        sample_cases = [
            {
                'patient_id': 'DEMO-001',
                'age': 58,
                'sex': 'M',
                'status': 'completed',
                'scan_date': date.today() - timedelta(days=1),
                'field_strength': '3T',
                'clinical_history': 'Chronic headaches, recent onset of seizures',
                'indication': 'Rule out intracranial mass',
                'completed_at': timezone.now() - timedelta(days=1),
            },
            {
                'patient_id': 'DEMO-002',
                'age': 42,
                'sex': 'F',
                'status': 'processing',
                'scan_date': date.today(),
                'field_strength': '3T',
                'clinical_history': 'Progressive neurological symptoms',
                'indication': 'Evaluation for brain tumor',
            },
            {
                'patient_id': 'DEMO-003',
                'age': 65,
                'sex': 'M',
                'status': 'completed',
                'scan_date': date.today() - timedelta(days=2),
                'field_strength': '1.5T',
                'clinical_history': 'Follow-up post-surgery',
                'indication': 'Surveillance imaging',
                'completed_at': timezone.now() - timedelta(days=2),
            },
            {
                'patient_id': 'DEMO-004',
                'age': 51,
                'sex': 'F',
                'status': 'pending',
                'scan_date': date.today() - timedelta(days=2),
                'field_strength': '3T',
                'clinical_history': 'Memory loss, cognitive decline',
                'indication': 'Assessment for dementia vs mass lesion',
            },
            {
                'patient_id': 'DEMO-005',
                'age': 36,
                'sex': 'M',
                'status': 'completed',
                'scan_date': date.today() - timedelta(days=3),
                'field_strength': '3T',
                'clinical_history': 'New onset seizures',
                'indication': 'Rule out structural abnormality',
                'completed_at': timezone.now() - timedelta(hours=12),
            },
        ]

        created_count = 0
        for case_data in sample_cases:
            case = Case.objects.create(
                created_by=user,
                **case_data
            )
            created_count += 1
            self.stdout.write(f'  Created case: {case.patient_id} ({case.status})')

        self.stdout.write(
            self.style.SUCCESS(f'\nSuccessfully created {created_count} sample cases!')
        )
        self.stdout.write(f'Demo user credentials: demo@hospital.com / demo123456')
