import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Upload, Download, Eye, BarChart3, FileText, Loader2 } from 'lucide-react';
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
    const [groundTruthUrl, setGroundTruthUrl] = useState<string | null>(null);
    const [isGeneratingReport, setIsGeneratingReport] = useState(false);

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

            // Check if ground truth exists and grab its URL
            const gtPath = fetchedResult.structured_findings?.ground_truth_mask;
            if (gtPath) {
                setHasGroundTruth(true);
                // Build absolute URL — gtPath may be relative or absolute
                const url = gtPath.startsWith('http') ? gtPath : `http://localhost:8000/media/${gtPath.replace(/^\/media\//, '')}`;
                setGroundTruthUrl(url);
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

        if (!file.name.endsWith('.nii') && !file.name.endsWith('.nii.gz')) {
            showError('Please upload a NIfTI file (.nii or .nii.gz)');
            return;
        }

        setIsUploadingGT(true);
        try {
            const response = await apiService.uploadGroundTruth(id, file);
            success('Ground truth uploaded successfully');
            setHasGroundTruth(true);
            // Grab URL from response if available, otherwise re-fetch
            if (response?.url) {
                const url = response.url.startsWith('http') ? response.url : `http://localhost:8000${response.url}`;
                setGroundTruthUrl(url);
            }
            await fetchResults();
        } catch (err: any) {
            showError(err.message || 'Failed to upload ground truth');
        } finally {
            setIsUploadingGT(false);
        }
    };

    const handleGenerateReport = async () => {
        if (!id) return;
        setIsGeneratingReport(true);
        try {
            const result = await apiService.generateReport(id);
            success('Report generated successfully!');
            navigate(`/reports/${result.report?.reportId}`);
        } catch (err: any) {
            showError(err.message || 'Failed to generate report');
        } finally {
            setIsGeneratingReport(false);
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
                <div style={{ flex: 1 }}>
                    <h1 className="page-title">Segmentation Results</h1>
                    <p className="page-subtitle">{caseData?.patientId || 'Patient'} - Analysis Report</p>
                </div>
                <button
                    className="btn btn-primary"
                    onClick={handleGenerateReport}
                    disabled={isGeneratingReport}
                >
                    {isGeneratingReport
                        ? <><Loader2 size={16} className="spin" style={{ marginRight: '0.4rem', animation: 'spin 1s linear infinite' }} /> Generating&hellip;</>
                        : <><FileText size={16} style={{ marginRight: '0.4rem' }} /> Generate Report</>
                    }
                </button>
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

                {/* Expert Gating Weights Card */}
                {resultData.gating_weights && Object.keys(resultData.gating_weights).length > 0 && (
                    <div className="card">
                        <div className="card-header">
                            <h3>Expert Contributions (Gating Weights)</h3>
                        </div>
                        <div className="card-body">
                            <p style={{ fontSize: '0.8rem', color: '#9CA3AF', marginBottom: '1rem' }}>
                                How much each modality expert contributed to the final segmentation.
                            </p>
                            {Object.entries(resultData.gating_weights as Record<string, number>).map(([mod, w]) => (
                                <div key={mod} style={{ marginBottom: '0.75rem' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.25rem' }}>
                                        <span style={{ fontWeight: 600 }}>{mod}</span>
                                        <span>{(w * 100).toFixed(1)}%</span>
                                    </div>
                                    <div style={{ height: 8, borderRadius: 999, background: 'var(--bg-secondary)', overflow: 'hidden' }}>
                                        <div style={{ height: '100%', borderRadius: 999, background: 'var(--color-primary)', width: `${(w * 100).toFixed(1)}%`, transition: 'width 0.6s ease' }} />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Visualization Card */}
                <div className="card">
                    <div className="card-header">
                        <h3>Segmentation Masks</h3>
                    </div>
                    <div className="card-body">
                        <div className="visualization-grid">
                            {resultData.mask_files && Object.entries(resultData.mask_files).map(([key, path]: [string, any]) => {
                                if (!path) return null;
                                let label = key.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase());
                                if (key === 'full_segmentation') {
                                    label = 'Full Segmentation (Combined)';
                                }
                                return (
                                    <div key={key} className="visualization-item">
                                        <div className="visualization-label">{label}</div>
                                        <button
                                            className="btn btn-primary btn-sm"
                                            onClick={() => setViewerImage({
                                                url: path.startsWith('http') ? path : `http://localhost:8000${path}`,
                                                modality: label
                                            })}
                                        >
                                            <Eye size={16} />
                                            View 3D
                                        </button>
                                        <a
                                            href={path.startsWith('http') ? path : `http://localhost:8000${path}`}
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
                                <div className="upload-actions" style={{ gap: '0.75rem', marginTop: '1rem' }}>
                                    {groundTruthUrl && (
                                        <>
                                            <button
                                                className="btn btn-primary btn-sm"
                                                onClick={() => setViewerImage({ url: groundTruthUrl, modality: 'Ground Truth' })}
                                            >
                                                <Eye size={16} />
                                                View 3D
                                            </button>
                                            <a
                                                href={groundTruthUrl}
                                                download
                                                className="btn btn-outline btn-sm"
                                            >
                                                <Download size={16} />
                                                Download
                                            </a>
                                        </>
                                    )}
                                    <label className="btn btn-outline btn-sm">
                                        <Upload size={16} />
                                        Replace
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

                {/* Voxel Difference Comparison Card */}
                {
                    resultData.structured_findings?.ground_truth_comparison && (
                        <div className="card">
                            <div className="card-header">
                                <h3>Voxel Difference Comparison</h3>
                            </div>
                            <div className="card-body">
                                <p style={{ fontSize: '0.85rem', color: '#6B7280', marginBottom: '1rem' }}>
                                    Direct comparison between the model's prediction and the uploaded ground truth mask.
                                </p>
                                <div style={{ overflowX: 'auto' }}>
                                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
                                        <thead>
                                            <tr style={{ borderBottom: '2px solid var(--border-color)', textAlign: 'left' }}>
                                                <th style={{ padding: '0.75rem' }}>Region</th>
                                                <th style={{ padding: '0.75rem' }}>Dice Score (DSC)</th>
                                                <th style={{ padding: '0.75rem' }}>IoU</th>
                                                <th style={{ padding: '0.75rem' }}>GT Volume</th>
                                                <th style={{ padding: '0.75rem' }}>Pred Volume</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {['whole_tumor', 'tumor_core', 'enhancing_tumor'].map((region) => {
                                                const comp = resultData.structured_findings.ground_truth_comparison[region];
                                                if (!comp) return null;
                                                const label = region.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
                                                return (
                                                    <tr key={region} style={{ borderBottom: '1px solid var(--border-color)' }}>
                                                        <td style={{ padding: '0.75rem', fontWeight: 600 }}>{label}</td>
                                                        <td style={{ padding: '0.75rem', color: 'var(--color-primary)', fontWeight: 'bold' }}>
                                                            {(comp.dice * 100).toFixed(2)}%
                                                        </td>
                                                        <td style={{ padding: '0.75rem' }}>{(comp.iou * 100).toFixed(2)}%</td>
                                                        <td style={{ padding: '0.75rem' }}>{formatVolume(comp.gt_volume)}</td>
                                                        <td style={{ padding: '0.75rem' }}>{formatVolume(comp.pred_volume)}</td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    )
                }

                {/* Detailed JSON Descriptor Card */}
                {
                    resultData.structured_findings && (
                        <div className="card" style={{ gridColumn: '1 / -1' }}>
                            <div className="card-header">
                                <h3><FileText size={20} style={{ marginRight: '0.5rem' }} /> Detailed JSON Descriptor</h3>
                            </div>
                            <div className="card-body">
                                <p style={{ fontSize: '0.85rem', color: '#6B7280', marginBottom: '1rem' }}>
                                    Underlying JSON schema containing the atlas-mapped clinical features used for report generation.
                                </p>
                                <div style={{ maxHeight: '400px', overflowY: 'auto', background: '#f8fafc', padding: '1rem', borderRadius: '0.5rem', border: '1px solid #e2e8f0' }}>
                                    <pre style={{ margin: 0, fontSize: '0.8rem', color: '#334155', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                                        {JSON.stringify(resultData.structured_findings, null, 2)}
                                    </pre>
                                </div>
                            </div>
                        </div>
                    )
                }

                {/* Model Info Card */}
                {
                    resultData.structured_findings && (
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
                    )
                }
            </div >

            {/* MRI Viewer Modal */}
            {
                viewerImage && (
                    <MRIViewer
                        imageUrl={viewerImage.url}
                        modality={viewerImage.modality}
                        onClose={() => setViewerImage(null)}
                    />
                )
            }
        </div>
    );
}
