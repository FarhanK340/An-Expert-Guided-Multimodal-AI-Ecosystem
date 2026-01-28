import { useNavigate } from 'react-router-dom';
import { Brain, Microscope, FileText, ShieldCheck, ArrowRight, Users, BarChart3 } from 'lucide-react';
import './LandingPage.css';

export default function LandingPage() {
    const navigate = useNavigate();

    const features = [
        {
            icon: Brain,
            title: 'AI-Powered Segmentation',
            description: 'Advanced MoME+ architecture for precise brain tumor segmentation across multiple modalities',
        },
        {
            icon: FileText,
            title: 'Automated Reports',
            description: 'LLM-generated clinical reports with anatomical analysis and structured findings',
        },
        {
            icon: Microscope,
            title: 'Multi-Modal Analysis',
            description: 'Support for T1, T1ce, T2, and FLAIR MRI sequences for comprehensive evaluation',
        },
        {
            icon: ShieldCheck,
            title: 'HIPAA Compliant',
            description: 'Secure, encrypted data storage and processing with role-based access control',
        },
    ];

    const userTypes = [
        {
            title: 'Radiologists',
            description: 'Streamline diagnosis with AI-assisted segmentation and automated preliminary reports',
            icon: Microscope,
        },
        {
            title: 'Neurosurgeons',
            description: 'Precise tumor volumetrics and anatomical mapping for surgical planning',
            icon: Brain,
        },
        {
            title: 'Researchers',
            description: 'Advanced analytics and continual learning capabilities for medical research',
            icon: BarChart3,
        },
    ];

    return (
        <div className="landing-page">
            {/* Navigation */}
            <nav className="landing-nav">
                <div className="nav-content">
                    <div className="nav-logo">
                        <Brain size={32} className="logo-icon" />
                        <span className="logo-text">MedicalAI</span>
                    </div>
                    <div className="nav-actions">
                        <button className="btn btn-ghost" onClick={() => navigate('/login')}>
                            Sign In
                        </button>
                        <button className="btn btn-primary" onClick={() => navigate('/signup')}>
                            Get Started
                        </button>
                    </div>
                </div>
            </nav>

            {/* Hero Section */}
            <section className="hero">
                <div className="hero-content">
                    <div className="hero-badge">
                        <span className="badge badge-neutral">Research Prototype</span>
                    </div>
                    <h1 className="hero-title">
                        AI-Powered Brain Tumor
                        <br />
                        <span className="hero-gradient">Segmentation & Analysis</span>
                    </h1>
                    <p className="hero-description">
                        Advanced medical AI platform for brain MRI analysis using mixture of modality experts.
                        Automated segmentation, LLM-generated reports, and comprehensive visualization tools for
                        healthcare professionals.
                    </p>
                    <div className="hero-actions">
                        <button className="btn btn-primary btn-lg" onClick={() => navigate('/signup')}>
                            Get Started
                            <ArrowRight size={20} />
                        </button>
                        <button className="btn btn-outline btn-lg" onClick={() => navigate('/login')}>
                            Sign In
                        </button>
                    </div>
                    <div className="hero-stats">
                        <div className="stat-item">
                            <Users size={20} />
                            <span>500+ Users</span>
                        </div>
                        <div className="stat-divider">•</div>
                        <div className="stat-item">
                            <Brain size={20} />
                            <span>10,000+ Cases Analyzed</span>
                        </div>
                        <div className="stat-divider">•</div>
                        <div className="stat-item">
                            <BarChart3 size={20} />
                            <span>95% Accuracy</span>
                        </div>
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section className="features-section">
                <div className="section-content">
                    <div className="section-header-center">
                        <h2 className="section-title">Advanced Features for Medical Professionals</h2>
                        <p className="section-subtitle">
                            Comprehensive AI-powered tools designed for clinical excellence
                        </p>
                    </div>
                    <div className="features-grid">
                        {features.map((feature) => {
                            const Icon = feature.icon;
                            return (
                                <div key={feature.title} className="feature-card card">
                                    <div className="feature-icon">
                                        <Icon size={24} />
                                    </div>
                                    <h3 className="feature-title">{feature.title}</h3>
                                    <p className="feature-description">{feature.description}</p>
                                </div>
                            );
                        })}
                    </div>
                </div>
            </section>

            {/* User Types Section */}
            <section className="user-types-section">
                <div className="section-content">
                    <div className="section-header-center">
                        <h2 className="section-title">Built for Healthcare Professionals</h2>
                        <p className="section-subtitle">Tailored solutions for every role</p>
                    </div>
                    <div className="user-types-grid">
                        {userTypes.map((userType) => {
                            const Icon = userType.icon;
                            return (
                                <div key={userType.title} className="user-type-card">
                                    <div className="user-type-icon">
                                        <Icon size={32} />
                                    </div>
                                    <h3 className="user-type-title">{userType.title}</h3>
                                    <p className="user-type-description">{userType.description}</p>
                                </div>
                            );
                        })}
                    </div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="cta-section">
                <div className="cta-content">
                    <h2 className="cta-title">Ready to Transform Your Workflow?</h2>
                    <p className="cta-description">
                        Join hundreds of healthcare professionals using MedicalAI for brain tumor analysis
                    </p>
                    <button className="btn btn-primary btn-lg" onClick={() => navigate('/signup')}>
                        Get Started Free
                        <ArrowRight size={20} />
                    </button>
                </div>
            </section>

            {/* Footer */}
            <footer className="landing-footer">
                <div className="footer-content">
                    <div className="footer-disclaimer">
                        <p className="footer-text">
                            <strong>Research Prototype</strong> • For Investigational Use Only • Not for Clinical Diagnosis
                        </p>
                    </div>
                </div>
            </footer>
        </div>
    );
}
