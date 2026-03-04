import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FileText, Clock, CheckCircle, Download, ExternalLink } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { apiService } from '../services/api';
import { useNotification } from '../contexts/NotificationContext';
import './PatientDashboardPage.css';

export default function PatientDashboardPage() {
    const { user } = useAuth();
    const navigate = useNavigate();
    const { error: showError } = useNotification();

    const [reports, setReports] = useState<any[]>([]);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        fetchReports();
    }, []);

    const fetchReports = async () => {
        try {
            setIsLoading(true);
            const data = await apiService.getReports();
            setReports(data);
        } catch (err: any) {
            showError(err.message || 'Failed to load reports');
        } finally {
            setIsLoading(false);
        }
    };

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'final':
            case 'approved':
                return <CheckCircle size={16} className="status-icon status-done" />;
            case 'draft':
            case 'pending':
                return <Clock size={16} className="status-icon status-pending" />;
            default:
                return <Clock size={16} className="status-icon status-pending" />;
        }
    };

    const handleExportPDF = async (reportId: string) => {
        try {
            await apiService.exportReportPDF(reportId);
        } catch (err: any) {
            showError('Failed to export PDF');
        }
    };

    const stats = [
        {
            label: 'Total Reports',
            value: reports.length,
            icon: FileText,
            color: 'primary',
        },
        {
            label: 'Finalised',
            value: reports.filter(r => r.status === 'final' || r.status === 'approved').length,
            icon: CheckCircle,
            color: 'success',
        },
        {
            label: 'Pending Review',
            value: reports.filter(r => r.status === 'draft' || r.status === 'pending').length,
            icon: Clock,
            color: 'warning',
        },
    ];

    return (
        <div className="patient-dashboard">
            {/* Welcome */}
            <div className="page-header">
                <div>
                    <h1 className="page-title">My Health Reports</h1>
                    <p className="page-subtitle">
                        Reports prepared by your care team for {user?.firstName} {user?.lastName}
                    </p>
                </div>
            </div>

            {/* Summary Stats */}
            <div className="patient-stats-grid">
                {stats.map(stat => {
                    const Icon = stat.icon;
                    return (
                        <div key={stat.label} className={`stat-card card stat-${stat.color}`}>
                            <div className="stat-icon-wrapper">
                                <Icon size={22} className="stat-icon" />
                            </div>
                            <div className="stat-content">
                                <p className="stat-label">{stat.label}</p>
                                <p className="stat-value">{stat.value}</p>
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Reports List */}
            <div className="card">
                <div className="card-header">
                    <h3>Your Reports</h3>
                </div>
                <div className="card-body" style={{ padding: 0 }}>
                    {isLoading ? (
                        <div style={{ padding: '3rem', textAlign: 'center', color: '#6B7280' }}>
                            Loading your reports…
                        </div>
                    ) : reports.length === 0 ? (
                        <div className="empty-reports">
                            <FileText size={48} style={{ color: '#D1D5DB', marginBottom: '1rem' }} />
                            <p>No reports yet.</p>
                            <p style={{ fontSize: '0.85rem', color: '#9CA3AF' }}>
                                Your doctor will generate a report once your scan analysis is complete.
                            </p>
                        </div>
                    ) : (
                        <div className="reports-list">
                            {reports.map(report => (
                                <div key={report.reportId || report.id} className="report-row">
                                    <div className="report-row-left">
                                        <div className="report-status-badge">
                                            {getStatusIcon(report.status)}
                                            <span className={`status-label status-${report.status}`}>
                                                {report.status?.charAt(0).toUpperCase() + report.status?.slice(1)}
                                            </span>
                                        </div>
                                        <div>
                                            <p className="report-title">
                                                Brain MRI Analysis Report
                                            </p>
                                            <p className="report-meta">
                                                {report.createdAt
                                                    ? new Date(report.createdAt).toLocaleDateString('en-US', {
                                                        year: 'numeric', month: 'long', day: 'numeric'
                                                    })
                                                    : 'Date unavailable'
                                                }
                                                {report.createdByName && ` · Dr. ${report.createdByName}`}
                                            </p>
                                        </div>
                                    </div>
                                    <div className="report-row-actions">
                                        <button
                                            className="btn btn-outline btn-sm"
                                            onClick={() => navigate(`/reports/${report.reportId || report.id}`)}
                                        >
                                            <ExternalLink size={14} />
                                            View
                                        </button>
                                        <button
                                            className="btn btn-primary btn-sm"
                                            onClick={() => handleExportPDF(report.reportId || report.id)}
                                        >
                                            <Download size={14} />
                                            PDF
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {/* Information note */}
            <div className="patient-info-note">
                <p>
                    <strong>Note:</strong> These reports are generated using AI-assisted brain MRI analysis.
                    Please discuss any findings with your treating physician.
                    This tool is for informational purposes only and is not a substitute for clinical diagnosis.
                </p>
            </div>
        </div>
    );
}
