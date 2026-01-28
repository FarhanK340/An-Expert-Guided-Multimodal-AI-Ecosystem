import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Upload, Download, Eye, BarChart3 } from 'lucide-react';
import { apiService } from '../services/api';
import { useNotification } from '../contexts/NotificationContext';
import MRIViewer from '../components/MRIViewer';
import './ResultsPage.css';

export default function ResultsPage() {
    const { id } = useParams<{ id: string }>();
    const navigate = useNavigate();
    const { error: showError, success } = useNotification();

    const [caseData, setCaseData] = useState<any>(null);
    const [resultData, setResultData] = useState<any>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [viewerImage, setViewerImage] = useState<{ url: string; modality: string } | null>(null);
    const [isUploadingGT, setIsUploadingGT] = useState(false);
    const [hasGroundTruth, setHasGroundTruth] = useState(false);

    useEffect(() => {
        if (id) {
            fetchResults();
        }
    }, [id]);

    const fetchResults = async () => {
        try {
            setIsLoading(true);
            const [fetchedCase, fetchedResult] = await Promise.all([
                apiService.getCase(id!),
                apiService.getSegmentationResult(id!),
            ]);
            setCaseData(fetchedCase);
            setResultData(fetchedResult);

            // Check if ground truth exists
            if (fetchedResult.structured_findings?.ground_truth_mask) {
                setHasGroundTruth(true);
            }
        } catch (err: any) {
            showError(err.message || 'Failed to load results');
        } finally {
            setIsLoading(false);
        }
    };

    const handleGroundTruthUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file || !id) return;

        // Validate file type
        if (!file.name.endsWith('.nii') && !file.name.endsWith('.nii.gz')) {
            showError('Please upload a NIfTI file (.nii or .nii.gz)');
            return;
        }

        setIsUploadingGT(true);
        try {
            await apiService.uploadGroundTruth(id, file);
            success('Ground truth uploaded successfully');
            setHasGroundTruth(true);
            // Refresh results to get updated data
            await fetchResults();
        } catch (err: any) {
            showError(err.message || 'Failed to upload ground truth');
        } finally {
            setIsUploadingGT(false);
        }
    };

    const formatVolume = (volume: number): string => {
        if (volume < 1000) {
            return `${volume.toFixed(0)} mm³`;
        } else if (volume < 1000000) {
            return `${(volume / 1000).toFixed(2)} cm³`;
        } else {
            return `${(volume / 1000000).toFixed(2)} L`;
        }
    };

    const formatConfidence = (confidence: number): string => {
        return `${(confidence * 100).toFixed(1)}%`;
    };

    if (isLoading) {
        return (
            <div className="results-page">
                <div style={{ padding: '3rem', textAlign: 'center', color: '#6B7280' }}>
                    Loading results...
                </div>
            </div>
        );
    }

    if (!resultData) {
        return (
            <div className="results-page">
                <div style={{ padding: '3rem', textAlign: 'center', color: '#6B7280' }}>
                    No results found for this case
                </div>
            </div>
        );
    }

    return (
        <div className="results-page">
            {/* Header */}
            <div className="page-header">
                <button className="btn btn-ghost" onClick={() => navigate(`/cases/${id}`)}>
                    <ArrowLeft size={18} />
                    Back to Case Details
                </button>
                <div>
                    <h1 className="page-title">Segmentation Results</h1>
                    <p className="page-subtitle">{caseData?.patientId || 'Patient'} - Analysis Report</p>
                </div>
            </div>

            <div className="results-grid">
                {/* Metrics Card */}
                <div className="card">
                    <div className="card-header">
                        <h3>
                            <BarChart3 size={20} style={{ marginRight: '0.5rem' }} />
                            Tumor Metrics
                        </h3>
                    </div>
                    <div className="card-body">
                        <div className="metrics-grid">
                            <div className="metric-item">
                                <div className="metric-label">Whole Tumor (WT)</div>
                                <div className="metric-value">
                                    {formatVolume(resultData.volumes.whole_tumor)}
                                </div>
                                <div className="metric-confidence">
                                    Confidence: {formatConfidence(resultData.confidence_scores.whole_tumor)}
                                </div>
                            </div>
                            <div className="metric-item">
                                <div className="metric-label">Tumor Core (TC)</div>
                                <div className="metric-value">
                                    {formatVolume(resultData.volumes.tumor_core)}
                                </div>
                                <div className="metric-confidence">
                                    Confidence: {formatConfidence(resultData.confidence_scores.tumor_core)}
                                </div>
                            </div>
                            <div className="metric-item">
                                <div className="metric-label">Enhancing Tumor (ET)</div>
                                <div className="metric-value">
                                    {formatVolume(resultData.volumes.enhancing_tumor)}
                                </div>
                                <div className="metric-confidence">
                                    Confidence: {formatConfidence(resultData.confidence_scores.enhancing_tumor)}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Visualization Card */}
                <div className="card">
                    <div className="card-header">
                        <h3>Segmentation Masks</h3>
                    </div>
                    <div className="card-body">
                        <div className="visualization-grid">
                            {resultData.mask_files && Object.entries(resultData.mask_files).map(([key, path]: [string, any]) => {
                                if (!path) return null;
                                const label = key.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase());
                                return (
                                    <div key={key} className="visualization-item">
                                        <div className="visualization-label">{label}</div>
                                        <button
                                            className="btn btn-primary btn-sm"
                                            onClick={() => setViewerImage({
                                                url: `http://localhost:8000${path}`,
                                                modality: label
                                            })}
                                        >
                                            <Eye size={16} />
                                            View 3D
                                        </button>
                                        <a
                                            href={`http://localhost:8000${path}`}
                                            download
                                            className="btn btn-outline btn-sm"
                                        >
                                            <Download size={16} />
                                            Download
                                        </a>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </div>

                {/* Ground Truth Upload Card */}
                <div className="card">
                    <div className="card-header">
                        <h3>Ground Truth Comparison</h3>
                    </div>
                    <div className="card-body">
                        {hasGroundTruth ? (
                            <div className="ground-truth-info">
                                <div className="success-message">
                                    ✓ Ground truth mask uploaded
                                </div>
                                <p className="info-text">
                                    Ground truth mask is available for comparison. You can view it in 3D by opening the viewer.
                                </p>
                                <div className="upload-actions">
                                    <label className="btn btn-outline">
                                        <Upload size={16} />
                                        Replace Ground Truth
                                        <input
                                            type="file"
                                            accept=".nii,.nii.gz"
                                            onChange={handleGroundTruthUpload}
                                            disabled={isUploadingGT}
                                            style={{ display: 'none' }}
                                        />
                                    </label>
                                </div>
                            </div>
                        ) : (
                            <div className="upload-section">
                                <p className="info-text">
                                    Upload a ground truth segmentation mask to compare with the predicted results.
                                </p>
                                <label className="btn btn-primary upload-btn">
                                    <Upload size={18} />
                                    {isUploadingGT ? 'Uploading...' : 'Upload Ground Truth (.nii/.nii.gz)'}
                                    <input
                                        type="file"
                                        accept=".nii,.nii.gz"
                                        onChange={handleGroundTruthUpload}
                                        disabled={isUploadingGT}
                                        style={{ display: 'none' }}
                                    />
                                </label>
                            </div>
                        )}
                    </div>
                </div>

                {/* Model Info Card */}
                {resultData.structured_findings && (
                    <div className="card">
                        <div className="card-header">
                            <h3>Model Information</h3>
                        </div>
                        <div className="card-body">
                            <div className="info-grid">
                                <div className="info-item">
                                    <span className="info-label">Model Version</span>
                                    <span className="info-value">
                                        {resultData.structured_findings.model_version || 'MoME+ v1.0'}
                                    </span>
                                </div>
                                <div className="info-item">
                                    <span className="info-label">Device</span>
                                    <span className="info-value">
                                        {resultData.structured_findings.device || 'N/A'}
                                    </span>
                                </div>
                                <div className="info-item">
                                    <span className="info-label">Processed</span>
                                    <span className="info-value">
                                        {new Date(resultData.created_at).toLocaleString()}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* MRI Viewer Modal */}
            {viewerImage && (
                <MRIViewer
                    imageUrl={viewerImage.url}
                    modality={viewerImage.modality}
                    onClose={() => setViewerImage(null)}
                />
            )}
        </div>
    );
}
