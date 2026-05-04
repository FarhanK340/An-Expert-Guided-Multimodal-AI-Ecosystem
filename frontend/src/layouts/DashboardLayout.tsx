import { type ReactNode } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Home, FolderOpen, FileText, Settings, LogOut, Brain, ShieldCheck } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import './DashboardLayout.css';

interface DashboardLayoutProps {
    children: ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
    const location = useLocation();
    const navigate = useNavigate();
    const { user, logout, isAdmin, isPatient, isAuthenticated, isLoading } = useAuth();

    // Redirect to login if not authenticated (after loading)
    if (!isLoading && !isAuthenticated) {
        navigate('/login', { replace: true });
        return null;
    }

    // Navigation items filtered by role
    const getNavItems = () => {
        const base = [
            { name: 'Dashboard', href: '/dashboard', icon: Home, roles: ['doctor', 'researcher', 'patient', 'admin'] },
            { name: 'Cases', href: '/cases', icon: FolderOpen, roles: ['doctor', 'researcher', 'admin'] },
            { name: 'Reports', href: '/reports', icon: FileText, roles: ['doctor', 'researcher', 'patient', 'admin'] },
            { name: 'Settings', href: '/settings', icon: Settings, roles: ['doctor', 'researcher', 'patient', 'admin'] },
            { name: 'Admin Panel', href: '/admin', icon: ShieldCheck, roles: ['admin'] },
        ];

        const role = user?.role || 'doctor';
        return base.filter(item => item.roles.includes(role));
    };

    const navigation = getNavItems();
    const isActive = (path: string) => location.pathname === path || location.pathname.startsWith(path + '/');

    const getUserInitials = () => {
        if (!user) return 'U';
        return `${(user.firstName || 'U')[0]}${(user.lastName || 'U')[0]}`.toUpperCase();
    };

    const getRoleDisplay = () => {
        if (!user) return 'User';
        const roleMap: Record<string, string> = {
            doctor: 'Doctor',
            researcher: 'Researcher',
            patient: 'Patient',
            admin: 'Administrator',
        };
        return roleMap[user.role] || user.role;
    };

    if (isLoading) {
        return (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', color: '#6B7280' }}>
                Loading...
            </div>
        );
    }

    return (
        <div className="dashboard-layout">
            {/* Sidebar */}
            <aside className="sidebar">
                <div className="sidebar-header">
                    <div className="logo">
                        <Brain className="logo-icon" size={32} />
                        <div>
                            <h1 className="logo-title">MedicalAI</h1>
                            <p className="logo-subtitle">Brain Segmentation</p>
                        </div>
                    </div>
                </div>

                <nav className="sidebar-nav">
                    {navigation.map((item) => {
                        const Icon = item.icon;
                        return (
                            <Link
                                key={item.name}
                                to={item.href}
                                className={`nav-item ${isActive(item.href) ? 'active' : ''}`}
                            >
                                <Icon size={20} />
                                <span>{item.name}</span>
                            </Link>
                        );
                    })}
                </nav>

                <div className="sidebar-footer">
                    <button className="nav-item logout-btn" onClick={logout}>
                        <LogOut size={20} />
                        <span>Logout</span>
                    </button>
                </div>
            </aside>

            {/* Main Content */}
            <div className="main-content">
                <header className="top-bar">
                    <div className="top-bar-content">
                        <div className="user-info">
                            <div className={`user-avatar ${isAdmin ? 'avatar-admin' : isPatient ? 'avatar-patient' : ''}`}>
                                {getUserInitials()}
                            </div>
                            <div>
                                <p className="user-name">
                                    {user ? `${user.firstName} ${user.lastName}` : 'User'}
                                </p>
                                <p className="user-role">{getRoleDisplay()}</p>
                            </div>
                        </div>
                    </div>
                </header>

                <main className="content-area">
                    {children}
                </main>
            </div>
        </div>
    );
}
