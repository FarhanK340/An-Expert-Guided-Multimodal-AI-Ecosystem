import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Brain, Mail, Lock, User, Building2, Stethoscope } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import type { SignUpData } from '../types';
import './SignUpPage.css';

export default function SignUpPage() {
    const [formData, setFormData] = useState<SignUpData>({
        email: '',
        password: '',
        confirmPassword: '',
        firstName: '',
        lastName: '',
        role: 'doctor',
        specialty: '',
        institution: '',
    });
    const [isLoading, setIsLoading] = useState(false);
    const navigate = useNavigate();
    const { signup } = useAuth();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        if (formData.password !== formData.confirmPassword) {
            alert('Passwords do not match');
            return;
        }

        setIsLoading(true);

        try {
            await signup(formData);
            // Navigation handled by signup function
        } catch (error) {
            alert('Sign up failed. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    const updateField = (field: keyof SignUpData, value: string) => {
        setFormData(prev => ({ ...prev, [field]: value }));
    };

    return (
        <div className="signup-page">
            <div className="signup-container">
                <div className="signup-card card">
                    <div className="signup-header">
                        <div className="signup-logo">
                            <Brain size={40} className="signup-logo-icon" />
                        </div>
                        <h1 className="signup-title">Create Your Account</h1>
                        <p className="signup-subtitle">Join the AI-powered medical analysis platform</p>
                    </div>

                    <form onSubmit={handleSubmit} className="signup-form">
                        {/* Personal Information */}
                        <div className="form-section">
                            <h3 className="form-section-title">Personal Information</h3>

                            <div className="form-row">
                                <div className="form-group">
                                    <label htmlFor="firstName" className="form-label">First Name</label>
                                    <div className="input-wrapper">
                                        <User size={18} className="input-icon" />
                                        <input
                                            id="firstName"
                                            type="text"
                                            value={formData.firstName}
                                            onChange={(e) => updateField('firstName', e.target.value)}
                                            placeholder="John"
                                            required
                                            className="input-with-icon"
                                        />
                                    </div>
                                </div>

                                <div className="form-group">
                                    <label htmlFor="lastName" className="form-label">Last Name</label>
                                    <div className="input-wrapper">
                                        <User size={18} className="input-icon" />
                                        <input
                                            id="lastName"
                                            type="text"
                                            value={formData.lastName}
                                            onChange={(e) => updateField('lastName', e.target.value)}
                                            placeholder="Doe"
                                            required
                                            className="input-with-icon"
                                        />
                                    </div>
                                </div>
                            </div>

                            <div className="form-group">
                                <label htmlFor="email" className="form-label">Email Address</label>
                                <div className="input-wrapper">
                                    <Mail size={18} className="input-icon" />
                                    <input
                                        id="email"
                                        type="email"
                                        value={formData.email}
                                        onChange={(e) => updateField('email', e.target.value)}
                                        placeholder="doctor@hospital.com"
                                        required
                                        className="input-with-icon"
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Professional Information */}
                        <div className="form-section">
                            <h3 className="form-section-title">Professional Information</h3>

                            <div className="form-group">
                                <label htmlFor="role" className="form-label">Role</label>
                                <div className="input-wrapper">
                                    <Stethoscope size={18} className="input-icon" />
                                    <select
                                        id="role"
                                        value={formData.role}
                                        onChange={(e) => updateField('role', e.target.value as any)}
                                        required
                                        className="input-with-icon"
                                    >
                                        <option value="doctor">Doctor</option>
                                        <option value="radiologist">Radiologist</option>
                                        <option value="researcher">Researcher</option>
                                    </select>
                                </div>
                            </div>

                            <div className="form-row">
                                <div className="form-group">
                                    <label htmlFor="specialty" className="form-label">Specialty (Optional)</label>
                                    <input
                                        id="specialty"
                                        type="text"
                                        value={formData.specialty}
                                        onChange={(e) => updateField('specialty', e.target.value)}
                                        placeholder="Neurology, Oncology, etc."
                                    />
                                </div>

                                <div className="form-group">
                                    <label htmlFor="institution" className="form-label">Institution (Optional)</label>
                                    <div className="input-wrapper">
                                        <Building2 size={18} className="input-icon" />
                                        <input
                                            id="institution"
                                            type="text"
                                            value={formData.institution}
                                            onChange={(e) => updateField('institution', e.target.value)}
                                            placeholder="Hospital or University"
                                            className="input-with-icon"
                                        />
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Security */}
                        <div className="form-section">
                            <h3 className="form-section-title">Security</h3>

                            <div className="form-group">
                                <label htmlFor="password" className="form-label">Password</label>
                                <div className="input-wrapper">
                                    <Lock size={18} className="input-icon" />
                                    <input
                                        id="password"
                                        type="password"
                                        value={formData.password}
                                        onChange={(e) => updateField('password', e.target.value)}
                                        placeholder="••••••••"
                                        required
                                        minLength={8}
                                        className="input-with-icon"
                                    />
                                </div>
                                <p className="form-hint">Minimum 8 characters</p>
                            </div>

                            <div className="form-group">
                                <label htmlFor="confirmPassword" className="form-label">Confirm Password</label>
                                <div className="input-wrapper">
                                    <Lock size={18} className="input-icon" />
                                    <input
                                        id="confirmPassword"
                                        type="password"
                                        value={formData.confirmPassword}
                                        onChange={(e) => updateField('confirmPassword', e.target.value)}
                                        placeholder="••••••••"
                                        required
                                        className="input-with-icon"
                                    />
                                </div>
                            </div>
                        </div>

                        <button
                            type="submit"
                            className="btn btn-primary btn-lg signup-button"
                            disabled={isLoading}
                        >
                            {isLoading ? 'Creating Account...' : 'Create Account'}
                        </button>
                    </form>

                    <div className="signup-footer">
                        <p className="footer-text">
                            Already have an account?{' '}
                            <button onClick={() => navigate('/login')} className="link-button">
                                Sign In
                            </button>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
