from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    """Admin panel for custom User model."""

    list_display  = ('email', 'first_name', 'last_name', 'role', 'is_active', 'date_joined')
    list_filter   = ('role', 'is_active', 'is_staff', 'is_verified')
    search_fields = ('email', 'first_name', 'last_name', 'institution')
    ordering      = ('-date_joined',)

    fieldsets = (
        (None,              {'fields': ('email', 'password')}),
        ('Personal Info',   {'fields': ('first_name', 'last_name', 'phone_number')}),
        ('Professional',    {'fields': ('role', 'institution', 'specialty', 'license_number')}),
        ('Permissions',     {'fields': ('is_active', 'is_staff', 'is_superuser', 'is_verified', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )

    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'first_name', 'last_name', 'role', 'password1', 'password2'),
        }),
    )

    # These override BaseUserAdmin which uses 'username'
    filter_horizontal = ('groups', 'user_permissions')
