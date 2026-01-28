import { Users, FolderOpen, FileText, TrendingUp, Search } from 'lucide-react';
import { useState } from 'react';
import './AdminDashboardPage.css';

export default function AdminDashboardPage() {
    const [searchQuery, setSearchQuery] = useState('');

    const stats = [
        { label: 'Total Users', value: '156', icon: Users, color: 'primary' },
        { label: 'Total Cases', value: '1,234', icon: FolderOpen, color: 'success' },
        { label: 'Reports Generated', value: '987', icon: FileText, color: 'warning' },
        { label: 'This Month', value: '+45', icon: TrendingUp, color: 'neutral' },
    ];

    const topUsers = [
        { name: 'Dr. Sarah Johnson', role: 'Radiologist', casesCount: 145, reportsCount: 132 },
        { name: 'Dr. Michael Chen', role: 'Doctor', casesCount: 98, reportsCount: 89 },
        { name: 'Dr. Emily Rodriguez', role: 'Researcher', casesCount: 76, reportsCount: 54 },
        { name: 'Dr. James Wilson', role: 'Radiologist', casesCount: 65, reportsCount: 61 },
        { name: 'Dr. Lisa Anderson', role: 'Doctor', casesCount: 52, reportsCount: 48 },
    ];

    const recentUsers = [
        { name: 'Dr. John Smith', email: 'john.smith@hospital.com', role: 'Doctor', date: '2024-12-18', verified: true },
        { name: 'Dr. Maria Garcia', email: 'maria.garcia@clinic.com', role: 'Radiologist', date: '2024-12-17', verified: true },
        { name: 'Dr. David Lee', email: 'david.lee@research.edu', role: 'Researcher', date: '2024-12-17', verified: false },
        { name: 'Dr. Anna Kowalski', email: 'anna.k@hospital.com', role: 'Doctor', date: '2024-12-16', verified: true },
    ];

    const roleDistribution = [
        { role: 'Doctor', count: 78 },
        { role: 'Radiologist', count: 54 },
        { role: 'Researcher', count: 24 },
    ];

    return (
        <div className="admin-dashboard-page">
            <div className="page-header">
                <div>
                    <h1 className="page-title">Admin Dashboard</h1>
                    <p className="page-subtitle">System overview and user management</p>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="stats-grid">
                {stats.map((stat) => {
                    const Icon = stat.icon;
                    return (
                        <div key={stat.label} className={`stat-card card stat-${stat.color}`}>
                            <div className="stat-icon-wrapper">
                                <Icon className="stat-icon" size={24} />
                            </div>
                            <div className="stat-content">
                                <p className="stat-label">{stat.label}</p>
                                <p className="stat-value">{stat.value}</p>
                            </div>
                        </div>
                    );
                })}
            </div>

            <div className="admin-content-grid">
                {/* Top Users */}
                <div className="card">
                    <div className="card-header">
                        <h3>Top Users by Activity</h3>
                    </div>
                    <div className="card-body">
                        <div className="table-wrapper">
                            <table className="cases-table">
                                <thead>
                                    <tr>
                                        <th>User</th>
                                        <th>Role</th>
                                        <th>Cases</th>
                                        <th>Reports</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {topUsers.map((user, idx) => (
                                        <tr key={idx}>
                                            <td>
                                                <div className="user-cell">
                                                    <div className="user-avatar-small">
                                                        {user.name.split(' ').map(n => n[0]).join('')}
                                                    </div>
                                                    <span className="font-medium">{user.name}</span>
                                                </div>
                                            </td>
                                            <td>
                                                <span className="badge badge-neutral">{user.role}</span>
                                            </td>
                                            <td className="font-semibold">{user.casesCount}</td>
                                            <td className="font-semibold">{user.reportsCount}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                {/* Role Distribution */}
                <div className="card">
                    <div className="card-header">
                        <h3>Users by Role</h3>
                    </div>
                    <div className="card-body">
                        <div className="role-distribution">
                            {roleDistribution.map((item) => (
                                <div key={item.role} className="role-item">
                                    <div className="role-header">
                                        <span className="role-name">{item.role}</span>
                                        <span className="role-count">{item.count} users</span>
                                    </div>
                                    <div className="role-bar">
                                        <div
                                            className="role-bar-fill"
                                            style={{ width: `${(item.count / 156) * 100}%` }}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* All Users List */}
            <div className="card">
                <div className="card-header">
                    <h3>All  Users</h3>
                </div>
                <div className="card-body">
                    {/* Search */}
                    <div className="search-wrapper mb-3">
                        <div className="search-input-wrapper">
                            <Search size={18} className="search-icon" />
                            <input
                                type="text"
                                placeholder="Search users by name, email, or role..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="search-input"
                            />
                        </div>
                    </div>

                    <div className="table-wrapper">
                        <table className="cases-table">
                            <thead>
                                <tr>
                                    <th>User</th>
                                    <th>Email</th>
                                    <th>Role</th>
                                    <th>Joined</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {recentUsers.map((user, idx) => (
                                    <tr key={idx}>
                                        <td>
                                            <div className="user-cell">
                                                <div className="user-avatar-small">
                                                    {user.name.split(' ').map(n => n[0]).join('')}
                                                </div>
                                                <span className="font-medium">{user.name}</span>
                                            </div>
                                        </td>
                                        <td className="text-neutral">{user.email}</td>
                                        <td>
                                            <span className="badge badge-neutral">{user.role}</span>
                                        </td>
                                        <td className="text-neutral">{user.date}</td>
                                        <td>
                                            <span className={`badge ${user.verified ? 'badge-success' : 'badge-warning'}`}>
                                                {user.verified ? 'Verified' : 'Pending'}
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
}
