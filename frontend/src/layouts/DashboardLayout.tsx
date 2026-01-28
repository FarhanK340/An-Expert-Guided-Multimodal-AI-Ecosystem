import { type ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Home, FolderOpen, FileText, Settings, LogOut, Brain } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import './DashboardLayout.css';

interface DashboardLayoutProps {
    children: ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
    const location = useLocation();
    const { user, logout } = useAuth();

    const navigation = [
        { name: 'Dashboard', href: '/dashboard', icon: Home },
        { name: 'Cases', href: '/cases', icon: FolderOpen },
        { name: 'Reports', href: '/reports', icon: FileText },
        { name: 'Settings', href: '/settings', icon: Settings },
    ];

    const isActive = (path: string) => location.pathname === path;

    // Get user initials
    const getUserInitials = () => {
        if (!user) return 'U';
        return `${user.firstName[0]}${user.lastName[0]}`.toUpperCase();
    };

    // Get role display name
    const getRoleDisplay = () => {
        if (!user) return 'User';
        const roleMap: Record<string, string> = {
            doctor: 'Doctor',
            radiologist: 'Radiologist',
            researcher: 'Researcher',
            admin: 'Administrator',
        };
        return roleMap[user.role] || user.role;
    };

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
                            <div className="user-avatar">{getUserInitials()}</div>
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
