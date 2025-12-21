"""
Custom User model for the Medical AI System.
Supports clinicians and administrators with role-based access.
"""

from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, BaseUserManager
from django.db import models
from django.utils import timezone
from django.core.validators import EmailValidator


class UserManager(BaseUserManager):
    """Custom user manager for email-based authentication."""
    
    def create_user(self, email, password=None, **extra_fields):
        """Create and return a regular user."""
        if not email:
            raise ValueError('Users must have an email address')
        
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user
    
    def create_superuser(self, email, password=None, **extra_fields):
        """Create and return a superuser."""
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('role', 'admin')
        
        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')
        
        return self.create_user(email, password, **extra_fields)


class User(AbstractBaseUser, PermissionsMixin):
    """
    Custom User model for medical professionals.
    Extends AbstractBaseUser to use email instead of username.
    """
    
    ROLE_CHOICES = [
        ('doctor', 'Doctor'),
        ('radiologist', 'Radiologist'),
        ('researcher', 'Researcher'),
        ('admin', 'Administrator'),
    ]
    
    # Basic Information
    email = models.EmailField(
        unique=True,
        validators=[EmailValidator()],
        help_text='Email address for authentication'
    )
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    
    # Role and Permissions
    role = models.CharField(
        max_length=20,
        choices=ROLE_CHOICES,
        default='doctor',
        help_text='User role in the system'
    )
    
    # Professional Information
    institution = models.CharField(max_length=255, blank=True, help_text='Medical institution/hospital')
    specialty = models.CharField(max_length=100, blank=True, help_text='Medical specialty (e.g., Radiology, Neurology)')
    license_number = models.CharField(max_length=100, blank=True, help_text='Medical license number')
    phone_number = models.CharField(max_length=20, blank=True, help_text='Contact phone number')
    
    # Account Status
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_verified = models.BooleanField(default=False, help_text='Email verification status')
    
    # Timestamps
    date_joined = models.DateTimeField(default=timezone.now)
    last_login = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Preferences
    receive_notifications = models.BooleanField(default=True)
    theme_preference = models.CharField(
        max_length=10,
        choices=[('light', 'Light'), ('dark', 'Dark'), ('auto', 'Auto')],
        default='auto'
    )
    
    objects = UserManager()
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name', 'last_name']
    
    class Meta:
        db_table = 'users'
        verbose_name = 'User'
        verbose_name_plural = 'Users'
        ordering = ['-date_joined']
        indexes = [
            models.Index(fields=['email']),
            models.Index(fields=['role']),
        ]
    
    def __str__(self):
        return f"{self.get_full_name()} ({self.email})"
    
    def get_full_name(self):
        """Return the user's full name."""
        return f"{self.first_name} {self.last_name}".strip()
    
    def get_short_name(self):
        """Return the user's first name."""
        return self.first_name
    
    @property
    def is_clinician(self):
        """Check if user is a clinician."""
        return self.role == 'clinician'
    
    @property
    def is_admin(self):
        """Check if user is an administrator."""
        return self.role == 'admin'
    
    @property
    def is_researcher(self):
        """Check if user is a researcher."""
        return self.role == 'researcher'


class UserSession(models.Model):
    """Track user sessions for security and analytics."""
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sessions')
    session_key = models.CharField(max_length=40, unique=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    login_time = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'user_sessions'
        verbose_name = 'User Session'
        verbose_name_plural = 'User Sessions'
        ordering = ['-login_time']
    
    def __str__(self):
        return f"{self.user.email} - {self.login_time}"
