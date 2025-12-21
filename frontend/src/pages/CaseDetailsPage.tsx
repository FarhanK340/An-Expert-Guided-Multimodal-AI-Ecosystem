import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Calendar, User, Activity, Eye, Download, ArrowLeft, Trash2 } from 'lucide-react';
import { apiService } from '../services/api';
import { useNotification } from '../contexts/NotificationContext';
import MRIViewer from '../components/MRIViewer';
import './CaseDetailsPage.css';

export default function CaseDetailsPage() {
    const { id } = useParams<{ id: string }>();
    const navigate = useNavigate();
    const { error: showError, success } = useNotification();

    const [caseData, setCaseData] = useState<any>(null);
    const [mriImages, setMriImages] = useState<any[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [viewerImage, setViewerImage] = useState<{ url: string; modality: string } | null>(null);
    const [showDeleteDialog, setShowDeleteDialog] = useState(false);
    const [isDeleting, setIsDeleting] = useState(false);

    useEffect(() => {
        if (id) {
            fetchCaseData();
        }
    }, [id]);

    const fetchCaseData = async () => {
        try {
            setIsLoading(true);
            const [fetchedCase, images] = await Promise.all([
                apiService.getCase(id!),
                apiService.getMRIImages(id!),
            ]);
            setCaseData(fetchedCase);
            setMriImages(images);
        } catch (err: any) {
            showError(err.message || 'Failed to load case');
        } finally {
            setIsLoading(false);
        }
    };

    const handleDeleteCase = async () => {
        if (!id) return;

        setIsDeleting(true);
        try {
            await apiService.deleteCase(id);
            success('Case deleted successfully');
            navigate('/cases');
        } catch (err: any) {
            showError(err.message || 'Failed to delete case');
            setIsDeleting(false);
            setShowDeleteDialog(false);
        }
    };

    const getStatusBadge = (status: string) => {
        const badgeMap: Record<string, string> = {
            completed: 'badge-success',
            processing: 'badge-warning',
            pending: 'badge-neutral',
            uploaded: 'badge-neutral',
            created: 'badge-neutral',
            uploading: 'badge-warning',
            failed: 'badge-error',
        };
        return `badge ${badgeMap[status] || 'badge-neutral'}`;
    };

    const modalityLabels: Record<string, string> = {
        t1: 'T1-weighted',
        t1ce: 'T1 Contrast Enhanced',
        t2: 'T2-weighted',
        flair: 'FLAIR',
    };

    if (isLoading) {
        return (
            <div className="case-details-page">
                <div style={{ padding: '3rem', textAlign: 'center', color: '#6B7280' }}>
                    Loading case details...
                </div>
            </div>
        );
    }

    if (!caseData) {
        return (
            <div className="case-details-page">
                <div style={{ padding: '3rem', textAlign: 'center', color: '#6B7280' }}>
                    Case not found
                </div>
            </div>
        );
    }

    return (
        <div className="case-details-page">
            {/* Header */}
            <div className="page-header">
                <button className="btn btn-ghost" onClick={() => navigate('/cases')}>
                    <ArrowLeft size={18} />
                    Back to Cases
                </button>
                <div>
                    <h1 className="page-title">{caseData.patientId}</h1>
                    <p className="page-subtitle">Case Details & MRI Analysis</p>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <span className={getStatusBadge(caseData.status)}>
                        {caseData.status}
                    </span>
                    <button
                        className="btn btn-outline"
                        onClick={() => setShowDeleteDialog(true)}
                        style={{ color: '#EF4444', borderColor: '#EF4444' }}
                    >
                        <Trash2 size={18} />
                        Delete Case
                    </button>
                </div>
            </div>

            <div className="case-details-grid">
                {/* Patient Information */}
                <div className="card">
                    <div className="card-header">
                        <h3>Patient Information</h3>
                    </div>
                    <div className="card-body">
                        <div className="info-grid">
                            <div className="info-item">
                                <User size={18} className="info-icon" />
                                <div>
                                    <span className="info-label">Patient ID</span>
                                    <span className="info-value">{caseData.patientId}</span>
                                </div>
                            </div>
                            <div className="info-item">
                                <User size={18} className="info-icon" />
                                <div>
                                    <span className="info-label">Age</span>
                                    <span className="info-value">{caseData.age || 'N/A'} years</span>
                                </div>
                            </div>
                            <div className="info-item">
                                <User size={18} className="info-icon" />
                                <div>
                                    <span className="info-label">Sex</span>
                                    <span className="info-value">
                                        {caseData.sex === 'M' ? 'Male' : caseData.sex === 'F' ? 'Female' : 'N/A'}
                                    </span>
                                </div>
                            </div>
                            <div className="info-item">
                                <Calendar size={18} className="info-icon" />
                                <div>
                                    <span className="info-label">Created</span>
                                    <span className="info-value">
                                        {caseData.createdAt ? new Date(caseData.createdAt).toLocaleDateString() : 'N/A'}
                                    </span>
                                </div>
                            </div>
                            <div className="info-item">
                                <Activity size={18} className="info-icon" />
                                <div>
                                    <span className="info-label">Status</span>
                                    <span className="info-value">{caseData.status}</span>
                                </div>
                            </div>
                            <div className="info-item">
                                <User size={18} className="info-icon" />
                                <div>
                                    <span className="info-label">Created By</span>
                                    <span className="info-value">{caseData.createdBy || 'Unknown'}</span>
                                </div>
                            </div>
                        </div>

                        {caseData.clinicalHistory && (
                            <div style={{ marginTop: '1.5rem' }}>
                                <div className="info-label" style={{ marginBottom: '0.5rem' }}>Clinical History</div>
                                <p className="info-value">{caseData.clinicalHistory}</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* MRI Images */}
                <div className="card">
                    <div className="card-header">
                        <h3>MRI Scans ({mriImages.length}/4)</h3>
                    </div>
                    <div className="card-body">
                        {mriImages.length === 0 ? (
                            <div style={{ padding: '2rem', textAlign: 'center', color: '#6B7280' }}>
                                No MRI scans uploaded yet
                            </div>
                        ) : (
                            <div className="mri-images-grid">
                                {mriImages.map((image) => (
                                    <div key={image.id} className="mri-image-card">
                                        <div className="mri-image-header">
                                            <span className="mri-modality-badge">
                                                {modalityLabels[image.modality] || image.modality.toUpperCase()}
                                            </span>
                                            <span className="mri-file-size">
                                                {(image.fileSize / 1024 / 1024).toFixed(2)} MB
                                            </span>
                                        </div>
                                        <div className="mri-image-info">
                                            <p className="mri-filename">{image.originalFilename}</p>
                                            <p className="mri-upload-date">
                                                Uploaded: {new Date(image.uploadedAt).toLocaleDateString()}
                                            </p>
                                        </div>
                                        <div className="mri-image-actions">
                                            <button
                                                className="btn btn-primary btn-sm"
                                                onClick={() => setViewerImage({
                                                    url: image.filePath.startsWith('http')
                                                        ? image.filePath
                                                        : `http://localhost:8000${image.filePath}`,
                                                    modality: image.modality
                                                })}
                                            >
                                                <Eye size={16} />
                                                View 3D
                                            </button>
                                            <a
                                                href={image.filePath.startsWith('http')
                                                    ? image.filePath
                                                    : `http://localhost:8000${image.filePath}`}
                                                download
                                                className="btn btn-outline btn-sm"
                                            >
                                                <Download size={16} />
                                                Download
                                            </a>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* MRI Viewer Modal */}
            {viewerImage && (
                <MRIViewer
                    imageUrl={viewerImage.url}
                    modality={viewerImage.modality}
                    onClose={() => setViewerImage(null)}
                />
            )}

            {/* Delete Confirmation Dialog */}
            {showDeleteDialog && (
                <div className="mri-viewer-overlay" style={{ zIndex: 1000 }}>
                    <div className="card" style={{
                        maxWidth: '500px',
                        margin: 'auto',
                        marginTop: '20vh',
                        padding: '2rem'
                    }}>
                        <div className="card-header">
                            <h3 style={{ color: '#EF4444' }}>Delete Case</h3>
                        </div>
                        <div className="card-body">
                            <p style={{ marginBottom: '1.5rem', color: '#6B7280' }}>
                                Are you sure you want to delete case <strong>{caseData.patientId}</strong>?
                                This will permanently remove all associated MRI scans and data. This action cannot be undone.
                            </p>
                            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'flex-end' }}>
                                <button
                                    className="btn btn-outline"
                                    onClick={() => setShowDeleteDialog(false)}
                                    disabled={isDeleting}
                                >
                                    Cancel
                                </button>
                                <button
                                    className="btn btn-primary"
                                    onClick={handleDeleteCase}
                                    disabled={isDeleting}
                                    style={{
                                        backgroundColor: '#EF4444',
                                        borderColor: '#EF4444'
                                    }}
                                >
                                    {isDeleting ? 'Deleting...' : 'Delete Case'}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
