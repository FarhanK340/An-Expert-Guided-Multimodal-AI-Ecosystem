import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FileText, Search, Eye, Calendar, Activity } from 'lucide-react';
import { apiService } from '../services/api';
import { useNotification } from '../contexts/NotificationContext';
import './ReportsPage.css';

export default function ReportsPage() {
    const navigate = useNavigate();
    const { error: showError } = useNotification();
    const [reports, setReports] = useState<any[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [search, setSearch] = useState('');

    useEffect(() => {
        fetchReports();
    }, []);

    const fetchReports = async () => {
        try {
            setIsLoading(true);
            const data = await apiService.getReports();
            setReports(Array.isArray(data) ? data : []);
        } catch (err: any) {
            showError(err.message || 'Failed to load reports');
        } finally {
            setIsLoading(false);
        }
    };

    const filtered = reports.filter(r =>
        r.patientId?.toLowerCase().includes(search.toLowerCase()) ||
        r.status?.toLowerCase().includes(search.toLowerCase())
    );

    const getStatusBadge = (status: string) => {
        const map: Record<string, string> = {
            draft: 'badge badge-neutral',
            reviewed: 'badge badge-warning',
            finalized: 'badge badge-success',
            exported: 'badge badge-success',
        };
        return map[status] || 'badge badge-neutral';
    };

    return (
        <div className="reports-page">
            <div className="page-header">
                <div>
                    <h1 className="page-title">Reports</h1>
                    <p className="page-subtitle">AI-generated diagnostic reports</p>
                </div>
            </div>

            {/* Search */}
            <div className="reports-toolbar">
                <div className="search-wrapper">
                    <Search size={18} className="search-icon" />
                    <input
                        type="text"
                        className="search-input"
                        placeholder="Search by patient ID or status…"
                        value={search}
                        onChange={e => setSearch(e.target.value)}
                    />
                </div>
            </div>

            {isLoading ? (
                <div className="loading-state">Loading reports…</div>
            ) : filtered.length === 0 ? (
                <div className="empty-state">
                    <FileText size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
                    <p>No reports found.</p>
                    <p style={{ fontSize: '0.875rem', color: '#9CA3AF' }}>
                        Generate a report from a completed case to see it here.
                    </p>
                </div>
            ) : (
                <div className="card">
                    <table className="reports-table">
                        <thead>
                            <tr>
                                <th>Patient ID</th>
                                <th>Status</th>
                                <th>Edits</th>
                                <th>Generated</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filtered.map(report => (
                                <tr key={report.reportId}>
                                    <td>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                            <FileText size={16} style={{ color: '#6366F1' }} />
                                            <span className="patient-id">{report.patientId}</span>
                                        </div>
                                    </td>
                                    <td>
                                        <span className={getStatusBadge(report.status)}>
                                            {report.status}
                                        </span>
                                    </td>
                                    <td>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                                            <Activity size={14} style={{ color: '#9CA3AF' }} />
                                            {report.editCount ?? 0}
                                        </div>
                                    </td>
                                    <td>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                                            <Calendar size={14} style={{ color: '#9CA3AF' }} />
                                            {report.generatedAt
                                                ? new Date(report.generatedAt).toLocaleDateString()
                                                : 'N/A'}
                                        </div>
                                    </td>
                                    <td>
                                        <button
                                            className="btn btn-primary btn-sm"
                                            onClick={() => navigate(`/reports/${report.reportId}`)}
                                        >
                                            <Eye size={14} />
                                            View
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
}
