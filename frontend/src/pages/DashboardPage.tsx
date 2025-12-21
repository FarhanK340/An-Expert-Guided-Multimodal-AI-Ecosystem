import { useState, useEffect } from 'react';
import { Activity, FolderOpen, CheckCircle, Clock } from 'lucide-react';
import { apiService } from '../services/api';
import './DashboardPage.css';

export default function DashboardPage() {
    const [cases, setCases] = useState<any[]>([]);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const fetchCases = async () => {
            try {
                const data = await apiService.getCases();
                setCases(data);
            } catch (error) {
                console.error('Failed to fetch cases:', error);
            } finally {
                setIsLoading(false);
            }
        };

        fetchCases();
    }, []);

    // Calculate stats from actual data
    const stats = [
        {
            label: 'Total Cases',
            value: cases.length.toString(),
            icon: FolderOpen,
            color: 'primary'
        },
        {
            label: 'Completed',
            value: cases.filter(c => c.status === 'completed').length.toString(),
            icon: CheckCircle,
            color: 'success'
        },
        {
            label: 'Processing',
            value: cases.filter(c => c.status === 'processing').length.toString(),
            icon: Activity,
            color: 'warning'
        },
        {
            label: 'Pending',
            value: cases.filter(c => c.status === 'pending').length.toString(),
            icon: Clock,
            color: 'neutral'
        },
    ];

    // Get recent 5 cases
    const recentCases = cases.slice(0, 5).map(case_ => ({
        id: case_.patientId,
        age: case_.age || 'N/A',
        sex: case_.sex || 'N/A',
        status: case_.status,
        date: case_.createdAt ? new Date(case_.createdAt).toISOString().split('T')[0] : 'N/A',
    }));

    const getStatusBadge = (status: string) => {
        const badgeMap: Record<string, string> = {
            completed: 'badge-success',
            processing: 'badge-warning',
            pending: 'badge-neutral',
            created: 'badge-neutral',
            uploading: 'badge-warning',
            uploaded: 'badge-neutral',
            failed: 'badge-error',
        };
        return `badge ${badgeMap[status] || 'badge-neutral'}`;
    };

    return (
        <div className="dashboard-page">
            <div className="page-header">
                <div>
                    <h1 className="page-title">Dashboard</h1>
                    <p className="page-subtitle">Overview of your medical AI analysis</p>
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

            {/* Recent Cases */}
            <div className="recent-section">
                <div className="section-header">
                    <h2 className="section-title">Recent Cases</h2>
                    <a href="/cases" className="section-link">
                        View All →
                    </a>
                </div>

                <div className="card">
                    <div className="table-wrapper">
                        {isLoading ? (
                            <div style={{ padding: '2rem', textAlign: 'center', color: '#6B7280' }}>
                                Loading cases...
                            </div>
                        ) : recentCases.length === 0 ? (
                            <div style={{ padding: '2rem', textAlign: 'center', color: '#6B7280' }}>
                                No cases found. Create your first case to get started.
                            </div>
                        ) : (
                            <table className="cases-table">
                                <thead>
                                    <tr>
                                        <th>Case ID</th>
                                        <th>Age</th>
                                        <th>Sex</th>
                                        <th>Status</th>
                                        <th>Date</th>
                                        <th></th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {recentCases.map((case_) => (
                                        <tr key={case_.id}>
                                            <td>
                                                <span className="case-id">{case_.id}</span>
                                            </td>
                                            <td>{case_.age}</td>
                                            <td>{case_.sex}</td>
                                            <td>
                                                <span className={getStatusBadge(case_.status)}>
                                                    {case_.status}
                                                </span>
                                            </td>
                                            <td className="text-neutral">{case_.date}</td>
                                            <td>
                                                <button className="btn btn-sm btn-ghost">View</button>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
