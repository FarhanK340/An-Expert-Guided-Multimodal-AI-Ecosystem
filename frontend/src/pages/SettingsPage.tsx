import { useState, useEffect } from 'react';
import { User, Mail, Lock, Building2, Stethoscope, Save, Phone } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { useNotification } from '../contexts/NotificationContext';
import { apiService } from '../services/api';
import './SettingsPage.css';

interface FormErrors {
    firstName?: string;
    lastName?: string;
    email?: string;
    phoneNumber?: string;
    specialty?: string;
    institution?: string;
    currentPassword?: string;
    newPassword?: string;
    confirmPassword?: string;
}

export default function SettingsPage() {
    const { user, updateUser } = useAuth();
    const { success, error: showError } = useNotification();

    const [profile, setProfile] = useState({
        firstName: '',
        lastName: '',
        email: '',
        role: 'doctor',
        specialty: '',
        institution: '',
        phoneNumber: '',
    });

    const [passwords, setPasswords] = useState({
        current: '',
        new: '',
        confirm: '',
    });

    const [profileErrors, setProfileErrors] = useState<FormErrors>({});
    const [passwordErrors, setPasswordErrors] = useState<FormErrors>({});
    const [isSubmittingProfile, setIsSubmittingProfile] = useState(false);
    const [isSubmittingPassword, setIsSubmittingPassword] = useState(false);

    // Load user data when component mounts
    useEffect(() => {
        if (user) {
            setProfile({
                firstName: user.firstName || '',
                lastName: user.lastName || '',
                email: user.email || '',
                role: user.role || 'doctor',
                specialty: user.specialty || '',
                institution: user.institution || '',
                phoneNumber: user.phoneNumber || '',
            });
        }
    }, [user]);

    const validateProfile = (): boolean => {
        const errors: FormErrors = {};

        if (!profile.firstName.trim()) {
            errors.firstName = 'First name is required';
        }

        if (!profile.lastName.trim()) {
            errors.lastName = 'Last name is required';
        }

        if (!profile.email.trim()) {
            errors.email = 'Email is required';
        } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(profile.email)) {
            errors.email = 'Invalid email format';
        }

        if (profile.phoneNumber && !/^\+?[\d\s\-()]+$/.test(profile.phoneNumber)) {
            errors.phoneNumber = 'Invalid phone number format';
        }

        setProfileErrors(errors);
        return Object.keys(errors).length === 0;
    };

    const validatePassword = (): boolean => {
        const errors: FormErrors = {};

        if (!passwords.current.trim()) {
            errors.currentPassword = 'Current password is required';
        }

        if (!passwords.new.trim()) {
            errors.newPassword = 'New password is required';
        } else if (passwords.new.length < 8) {
            errors.newPassword = 'Password must be at least 8 characters';
        }

        if (passwords.new !== passwords.confirm) {
            errors.confirmPassword = 'Passwords do not match';
        }

        setPasswordErrors(errors);
        return Object.keys(errors).length === 0;
    };

    const handleSaveProfile = async (e: React.FormEvent) => {
        e.preventDefault();

        if (!validateProfile()) {
            showError('Please fix the errors in the form');
            return;
        }

        setIsSubmittingProfile(true);
        setProfileErrors({});

        try {
            // Update user in context (cast role to proper type)
            await updateUser({
                ...profile,
                role: profile.role as 'doctor' | 'radiologist' | 'researcher' | 'admin',
            });

            success('Profile updated successfully!');
        } catch (err: any) {
            showError(err.message || 'Failed to update profile');
        } finally {
            setIsSubmittingProfile(false);
        }
    };

    const handleChangePassword = async (e: React.FormEvent) => {
        e.preventDefault();

        if (!validatePassword()) {
            showError('Please fix the errors in the form');
            return;
        }

        setIsSubmittingPassword(true);
        setPasswordErrors({});

        try {
            // Call API to change password
            await apiService.changePassword(passwords.current, passwords.new, passwords.confirm);

            success('Password changed successfully!');
            setPasswords({ current: '', new: '', confirm: '' });
        } catch (err: any) {
            // Try to parse backend validation errors
            try {
                const errorData = JSON.parse(err.message);

                // Map backend errors to form fields
                const errors: FormErrors = {};
                if (errorData.currentPassword) {
                    errors.currentPassword = Array.isArray(errorData.currentPassword)
                        ? errorData.currentPassword[0]
                        : errorData.currentPassword;
                }
                if (errorData.newPassword) {
                    errors.newPassword = Array.isArray(errorData.newPassword)
                        ? errorData.newPassword[0]
                        : errorData.newPassword;
                }
                if (errorData.confirmPassword) {
                    errors.confirmPassword = Array.isArray(errorData.confirmPassword)
                        ? errorData.confirmPassword[0]
                        : errorData.confirmPassword;
                }

                setPasswordErrors(errors);
                showError('Please fix the errors in the form');
            } catch {
                // If not JSON, show generic error
                showError(err.message || 'Failed to change password');
            }
        } finally {
            setIsSubmittingPassword(false);
        }
    };

    return (
        <div className="settings-page">
            <div className="page-header">
                <div>
                    <h1 className="page-title">Settings</h1>
                    <p className="page-subtitle">Manage your account and preferences</p>
                </div>
            </div>

            <div className="settings-grid">
                {/* Profile Settings */}
                <div className="card">
                    <div className="card-header">
                        <h3>Profile Information</h3>
                    </div>
                    <div className="card-body">
                        <form onSubmit={handleSaveProfile} className="settings-form">
                            <div className="form-row">
                                <div className="form-group">
                                    <label className="form-label">
                                        First Name <span className="required">*</span>
                                    </label>
                                    <div className="input-wrapper">
                                        <User size={18} className="input-icon" />
                                        <input
                                            type="text"
                                            value={profile.firstName}
                                            onChange={(e) => setProfile({ ...profile, firstName: e.target.value })}
                                            className={`input-with-icon ${profileErrors.firstName ? 'input-error' : ''}`}
                                        />
                                    </div>
                                    {profileErrors.firstName && (
                                        <span className="error-message">{profileErrors.firstName}</span>
                                    )}
                                </div>

                                <div className="form-group">
                                    <label className="form-label">
                                        Last Name <span className="required">*</span>
                                    </label>
                                    <div className="input-wrapper">
                                        <User size={18} className="input-icon" />
                                        <input
                                            type="text"
                                            value={profile.lastName}
                                            onChange={(e) => setProfile({ ...profile, lastName: e.target.value })}
                                            className={`input-with-icon ${profileErrors.lastName ? 'input-error' : ''}`}
                                        />
                                    </div>
                                    {profileErrors.lastName && (
                                        <span className="error-message">{profileErrors.lastName}</span>
                                    )}
                                </div>
                            </div>

                            <div className="form-group">
                                <label className="form-label">
                                    Email Address <span className="required">*</span>
                                </label>
                                <div className="input-wrapper">
                                    <Mail size={18} className="input-icon" />
                                    <input
                                        type="email"
                                        value={profile.email}
                                        onChange={(e) => setProfile({ ...profile, email: e.target.value })}
                                        className={`input-with-icon ${profileErrors.email ? 'input-error' : ''}`}
                                    />
                                </div>
                                {profileErrors.email && (
                                    <span className="error-message">{profileErrors.email}</span>
                                )}
                                {user?.isEmailVerified && (
                                    <span className="badge badge-success" style={{ marginTop: '0.5rem', display: 'inline-block' }}>
                                        Verified
                                    </span>
                                )}
                            </div>

                            <div className="form-group">
                                <label className="form-label">Phone Number</label>
                                <div className="input-wrapper">
                                    <Phone size={18} className="input-icon" />
                                    <input
                                        type="tel"
                                        value={profile.phoneNumber}
                                        onChange={(e) => setProfile({ ...profile, phoneNumber: e.target.value })}
                                        className={`input-with-icon ${profileErrors.phoneNumber ? 'input-error' : ''}`}
                                        placeholder="+1 234 567 8900"
                                    />
                                </div>
                                {profileErrors.phoneNumber && (
                                    <span className="error-message">{profileErrors.phoneNumber}</span>
                                )}
                            </div>

                            <div className="form-row">
                                <div className="form-group">
                                    <label className="form-label">Role</label>
                                    <div className="input-wrapper">
                                        <Stethoscope size={18} className="input-icon" />
                                        <select
                                            value={profile.role}
                                            onChange={(e) => setProfile({ ...profile, role: e.target.value })}
                                            className="input-with-icon"
                                        >
                                            <option value="doctor">Doctor</option>
                                            <option value="radiologist">Radiologist</option>
                                            <option value="researcher">Researcher</option>
                                        </select>
                                    </div>
                                </div>

                                <div className="form-group">
                                    <label className="form-label">Specialty</label>
                                    <input
                                        type="text"
                                        value={profile.specialty}
                                        onChange={(e) => setProfile({ ...profile, specialty: e.target.value })}
                                        placeholder="e.g., Neurology"
                                    />
                                </div>
                            </div>

                            <div className="form-group">
                                <label className="form-label">Institution</label>
                                <div className="input-wrapper">
                                    <Building2 size={18} className="input-icon" />
                                    <input
                                        type="text"
                                        value={profile.institution}
                                        onChange={(e) => setProfile({ ...profile, institution: e.target.value })}
                                        className="input-with-icon"
                                        placeholder="e.g., City Hospital"
                                    />
                                </div>
                            </div>

                            <button
                                type="submit"
                                className="btn btn-primary"
                                disabled={isSubmittingProfile}
                            >
                                <Save size={18} />
                                {isSubmittingProfile ? 'Saving...' : 'Save Changes'}
                            </button>
                        </form>
                    </div>
                </div>

                {/* Password Change */}
                <div className="card">
                    <div className="card-header">
                        <h3>Change Password</h3>
                    </div>
                    <div className="card-body">
                        <form onSubmit={handleChangePassword} className="settings-form">
                            <div className="form-group">
                                <label className="form-label">
                                    Current Password <span className="required">*</span>
                                </label>
                                <div className="input-wrapper">
                                    <Lock size={18} className="input-icon" />
                                    <input
                                        type="password"
                                        value={passwords.current}
                                        onChange={(e) => setPasswords({ ...passwords, current: e.target.value })}
                                        className={`input-with-icon ${passwordErrors.currentPassword ? 'input-error' : ''}`}
                                    />
                                </div>
                                {passwordErrors.currentPassword && (
                                    <span className="error-message">{passwordErrors.currentPassword}</span>
                                )}
                            </div>

                            <div className="form-group">
                                <label className="form-label">
                                    New Password <span className="required">*</span>
                                </label>
                                <div className="input-wrapper">
                                    <Lock size={18} className="input-icon" />
                                    <input
                                        type="password"
                                        value={passwords.new}
                                        onChange={(e) => setPasswords({ ...passwords, new: e.target.value })}
                                        className={`input-with-icon ${passwordErrors.newPassword ? 'input-error' : ''}`}
                                        minLength={8}
                                    />
                                </div>
                                {passwordErrors.newPassword ? (
                                    <span className="error-message">{passwordErrors.newPassword}</span>
                                ) : (
                                    <p className="form-hint">Minimum 8 characters</p>
                                )}
                            </div>

                            <div className="form-group">
                                <label className="form-label">
                                    Confirm New Password <span className="required">*</span>
                                </label>
                                <div className="input-wrapper">
                                    <Lock size={18} className="input-icon" />
                                    <input
                                        type="password"
                                        value={passwords.confirm}
                                        onChange={(e) => setPasswords({ ...passwords, confirm: e.target.value })}
                                        className={`input-with-icon ${passwordErrors.confirmPassword ? 'input-error' : ''}`}
                                    />
                                </div>
                                {passwordErrors.confirmPassword && (
                                    <span className="error-message">{passwordErrors.confirmPassword}</span>
                                )}
                            </div>

                            <button
                                type="submit"
                                className="btn btn-primary"
                                disabled={isSubmittingPassword}
                            >
                                <Lock size={18} />
                                {isSubmittingPassword ? 'Updating...' : 'Update Password'}
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    );
}
