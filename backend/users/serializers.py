"""
Serializers for User authentication and management.
"""

from rest_framework import serializers
from django.contrib.auth import get_user_model
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

User = get_user_model()


class UserSerializer(serializers.ModelSerializer):
    """Serializer for User model with camelCase fields for frontend."""
    
    username = serializers.SerializerMethodField()
    firstName = serializers.CharField(source='first_name')
    lastName = serializers.CharField(source='last_name')
    phoneNumber = serializers.CharField(source='phone_number', required=False, allow_blank=True)
    isEmailVerified = serializers.BooleanField(source='is_verified', read_only=True)
    createdAt = serializers.DateTimeField(source='date_joined', read_only=True)
    updatedAt = serializers.DateTimeField(source='updated_at', read_only=True)
    
    class Meta:
        model = User
        fields = [
            'id', 'username', 'email', 'firstName', 'lastName', 'role',
            'institution', 'specialty', 'phoneNumber', 'isEmailVerified', 
            'createdAt', 'updatedAt'
        ]
        read_only_fields = ['id', 'username', 'isEmailVerified', 'createdAt', 'updatedAt']
    
    def get_username(self, obj):
        """Generate username from email."""
        return obj.email.split('@')[0]


class RegisterSerializer(serializers.ModelSerializer):
    """Serializer for user registration."""
    
    password = serializers.CharField(write_only=True, min_length=8)
    confirm_password = serializers.CharField(write_only=True)
    
    class Meta:
        model = User
        fields = [
            'email', 'password', 'confirm_password', 
            'first_name', 'last_name', 'role',
            'institution', 'specialty'
        ]
    
    def validate(self, data):
        """Validate password match."""
        if data.get('password') != data.get('confirm_password'):
            raise serializers.ValidationError("Passwords do not match")
        return data
    
    def create(self, validated_data):
        """Create new user."""
        validated_data.pop('confirm_password')
        password = validated_data.pop('password')
        
        user = User.objects.create_user(
            password=password,
            **validated_data
        )
        return user


class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    """Custom JWT serializer to include user data."""
    
    def validate(self, attrs):
        data = super().validate(attrs)
        
        # Add user data to response
        data['user'] = UserSerializer(self.user).data
        
        return data


class UpdateProfileSerializer(serializers.ModelSerializer):
    """Serializer for updating user profile with camelCase fields."""
    
    firstName = serializers.CharField(source='first_name', required=False)
    lastName = serializers.CharField(source='last_name', required=False)
    phoneNumber = serializers.CharField(source='phone_number', required=False, allow_blank=True)
    
    class Meta:
        model = User
        fields = [
            'firstName', 'lastName', 'email', 'role',
            'institution', 'specialty', 'phoneNumber'
        ]
    
    def validate_role(self, value):
        """Prevent users from changing to admin role unless they're already admin."""
        request = self.context.get('request')
        if value == 'admin' and not request.user.is_admin:
            raise serializers.ValidationError("Cannot change role to admin")
        return value


class ChangePasswordSerializer(serializers.Serializer):
    """Serializer for changing user password."""
    
    currentPassword = serializers.CharField(write_only=True, required=True)
    newPassword = serializers.CharField(write_only=True, required=True, min_length=8)
    confirmPassword = serializers.CharField(write_only=True, required=True)
    
    def validate(self, data):
        """Validate passwords match."""
        if data['newPassword'] != data['confirmPassword']:
            raise serializers.ValidationError({"confirmPassword": "Passwords do not match"})
        return data
    
    def validate_currentPassword(self, value):
        """Validate current password is correct."""
        user = self.context['request'].user
        if not user.check_password(value):
            raise serializers.ValidationError("Current password is incorrect")
        return value
