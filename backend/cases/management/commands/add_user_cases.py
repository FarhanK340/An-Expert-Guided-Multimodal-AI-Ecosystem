"""
Management command to populate sample cases for a specific user.
"""

from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from cases.models import Case
from datetime import date, timedelta
from django.utils import timezone

User = get_user_model()


class Command(BaseCommand):
    help = 'Populate database with sample case data for current logged-in user'

    def add_arguments(self, parser):
        parser.add_argument(
            '--email',
            type=str,
            help='Email of the user to create cases for',
        )

    def handle(self, *args, **options):
        email = options.get('email', 'fkashif.bese22seecs@seecs.edu.pk')
        
        self.stdout.write(self.style.SUCCESS(f'Creating sample cases for {email}...'))

        # Get or create the user
        try:
            user = User.objects.get(email=email)
            self.stdout.write(self.style.SUCCESS(f'Found user: {user.get_full_name()}'))
        except User.DoesNotExist:
            self.stdout.write(self.style.ERROR(f'User with email {email} not found!'))
            self.stdout.write('Please provide an existing user email.')
            return

        # Sample cases data
        sample_cases = [
            {
                'patient_id': 'CASE-001',
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
                'patient_id': 'CASE-002',
                'age': 42,
                'sex': 'F',
                'status': 'processing',
                'scan_date': date.today(),
                'field_strength': '3T',
                'clinical_history': 'Progressive neurological symptoms',
                'indication': 'Evaluation for brain tumor',
            },
            {
                'patient_id': 'CASE-003',
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
                'patient_id': 'CASE-004',
                'age': 51,
                'sex': 'F',
                'status': 'pending',
                'scan_date': date.today() - timedelta(days=2),
                'field_strength': '3T',
                'clinical_history': 'Memory loss, cognitive decline',
                'indication': 'Assessment for dementia vs mass lesion',
            },
            {
                'patient_id': 'CASE-005',
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
            # Check if case already exists
            if Case.objects.filter(patient_id=case_data['patient_id'], created_by=user).exists():
                self.stdout.write(f'  Skipping {case_data["patient_id"]} (already exists)')
                continue
                
            case = Case.objects.create(
                created_by=user,
                **case_data
            )
            created_count += 1
            self.stdout.write(f'  Created case: {case.patient_id} ({case.status})')

        self.stdout.write(
            self.style.SUCCESS(f'\nSuccessfully created {created_count} sample cases for {user.email}!')
        )
