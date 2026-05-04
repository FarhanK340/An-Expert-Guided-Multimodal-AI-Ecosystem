import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Edit3, Save, X, Download, FileText } from 'lucide-react';
import { apiService } from '../services/api';
import { useNotification } from '../contexts/NotificationContext';
import { useAuth } from '../contexts/AuthContext';
import './ReportDetailPage.css';

export default function ReportDetailPage() {
    const { id } = useParams<{ id: string }>();
    const navigate = useNavigate();
    const { error: showError, success } = useNotification();
    const { isPatient } = useAuth();

    const [report, setReport] = useState<any>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isEditing, setIsEditing] = useState(false);
    const [editedText, setEditedText] = useState('');
    const [editReason, setEditReason] = useState('');
    const [isSaving, setIsSaving] = useState(false);
    const [isExporting, setIsExporting] = useState(false);

    useEffect(() => {
        if (id) fetchReport();
    }, [id]);

    const fetchReport = async () => {
        try {
            setIsLoading(true);
            const data = await apiService.getReport(id!);
            setReport(data);
            setEditedText(data.finalizedText || data.aiGeneratedText || '');
        } catch (err: any) {
            showError(err.message || 'Failed to load report');
        } finally {
            setIsLoading(false);
        }
    };

    const handleSaveEdit = async () => {
        if (!id) return;
        setIsSaving(true);
        try {
            const updated = await apiService.updateReport(id, {
                finalizedText: editedText,
                editReason,
            });
            setReport(updated.report);
            setIsEditing(false);
            setEditReason('');
            success('Report updated successfully');
        } catch (err: any) {
            showError(err.message || 'Failed to save changes');
        } finally {
            setIsSaving(false);
        }
    };

    const handleExportPDF = async () => {
        if (!id) return;
        setIsExporting(true);
        try {
            await apiService.exportReportPDF(id);
            success('PDF exported successfully');
        } catch (err: any) {
            showError(err.message || 'Failed to export PDF');
        } finally {
            setIsExporting(false);
        }
    };

    const getStatusBadge = (status: string) => {
        const map: Record<string, string> = {
            draft: 'badge badge-neutral',
            reviewed: 'badge badge-warning',
            finalized: 'badge badge-success',
            exported: 'badge badge-success',
        };
        return map[status] || 'badge badge-neutral';
    };

    if (isLoading) return <div className="report-detail-page"><div className="loading-state">Loading report…</div></div>;
    if (!report) return <div className="report-detail-page"><div className="loading-state">Report not found.</div></div>;

    const metrics = report.findingsJson?.tumor_metrics || {};
    const vols = metrics.volumes || {};
    const confs = metrics.confidence_scores || {};

    return (
        <div className="report-detail-page">
            {/* Header */}
            <div className="page-header">
                <button className="btn btn-ghost" onClick={() => navigate('/reports')}>
                    <ArrowLeft size={18} /> Back to Reports
                </button>
                <div style={{ flex: 1 }}>
                    <h1 className="page-title">Report — {report.patientId}</h1>
                    <p className="page-subtitle">Case ID: {report.caseId}</p>
                </div>
                <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                    <span className={getStatusBadge(report.status)}>{report.status}</span>
                    {!isPatient && !isEditing && (
                        <button className="btn btn-outline" onClick={() => setIsEditing(true)}>
                            <Edit3 size={16} /> Edit
                        </button>
                    )}
                    {isEditing && (
                        <>
                            <button className="btn btn-ghost" onClick={() => { setIsEditing(false); setEditedText(report.finalizedText || report.aiGeneratedText); }}>
                                <X size={16} /> Cancel
                            </button>
                            <button className="btn btn-primary" onClick={handleSaveEdit} disabled={isSaving}>
                                <Save size={16} /> {isSaving ? 'Saving…' : 'Save'}
                            </button>
                        </>
                    )}
                    <button className="btn btn-primary" onClick={handleExportPDF} disabled={isExporting}>
                        <Download size={16} /> {isExporting ? 'Exporting…' : 'Export PDF'}
                    </button>
                </div>
            </div>

            <div className="report-grid">
                {/* Segmentation Metrics */}
                {Object.keys(vols).length > 0 && (
                    <div className="card metrics-card">
                        <div className="card-header"><h3>Segmentation Metrics</h3></div>
                        <div className="card-body">
                            {[
                                { key: 'whole_tumor', label: 'Whole Tumor', color: '#6366F1' },
                                { key: 'tumor_core', label: 'Tumor Core', color: '#F59E0B' },
                                { key: 'enhancing_tumor', label: 'Enhancing Tumor', color: '#EF4444' },
                            ].map(({ key, label, color }) => (
                                <div key={key} className="metric-row">
                                    <div className="metric-header">
                                        <span className="metric-label" style={{ color }}>{label}</span>
                                        <span className="metric-volume">{(vols[key] || 0).toFixed(1)} mm³</span>
                                    </div>
                                    <div className="confidence-bar-bg">
                                        <div
                                            className="confidence-bar-fill"
                                            style={{
                                                width: `${((confs[key] || 0) * 100).toFixed(1)}%`,
                                                background: color,
                                            }}
                                        />
                                    </div>
                                    <span className="confidence-label">Confidence: {((confs[key] || 0) * 100).toFixed(1)}%</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Report Text */}
                <div className="card report-text-card">
                    <div className="card-header">
                        <h3>
                            <FileText size={18} style={{ marginRight: '0.5rem' }} />
                            Report Narrative
                        </h3>
                        {report.editCount > 0 && (
                            <span style={{ fontSize: '0.75rem', color: '#9CA3AF' }}>
                                {report.editCount} edit{report.editCount !== 1 ? 's' : ''}
                            </span>
                        )}
                    </div>
                    <div className="card-body">
                        {isEditing ? (
                            <>
                                <textarea
                                    className="report-textarea"
                                    value={editedText}
                                    onChange={e => setEditedText(e.target.value)}
                                    rows={24}
                                />
                                <div className="edit-reason-group">
                                    <label className="form-label">Reason for edit (optional)</label>
                                    <input
                                        type="text"
                                        className="edit-reason-input"
                                        placeholder="Clinical correction, typo fix, etc."
                                        value={editReason}
                                        onChange={e => setEditReason(e.target.value)}
                                    />
                                </div>
                            </>
                        ) : (
                            <div className="report-text" style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit' }}>
                                {(report.finalizedText || report.aiGeneratedText || '').split('\n').map((line: string, i: number) => {
                                    // Split by **text** and capture the match
                                    const parts = line.split(/(\*\*.*?\*\*)/g);
                                    return (
                                        <div key={i} style={{ minHeight: '1.5em' }}>
                                            {parts.map((part, j) => {
                                                if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
                                                    return <strong key={j} style={{ fontWeight: 600 }}>{part.slice(2, -2)}</strong>;
                                                }
                                                return <span key={j}>{part}</span>;
                                            })}
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Footer metadata */}
            <div className="report-meta">
                <span>Generated: {report.generatedAt ? new Date(report.generatedAt).toLocaleString() : 'N/A'}</span>
                <span>Last updated: {report.updatedAt ? new Date(report.updatedAt).toLocaleString() : 'N/A'}</span>
                {report.lastEditedBy && <span>Last edited by: {report.lastEditedBy}</span>}
            </div>
        </div>
    );
}
