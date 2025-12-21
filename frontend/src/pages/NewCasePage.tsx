import { useState } from 'react';
import { Upload, CheckCircle2, AlertCircle, X } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { apiService } from '../services/api';
import { useNotification } from '../contexts/NotificationContext';
import './NewCasePage.css';

type Modality = 't1' | 't1ce' | 't2' | 'flair';

export default function NewCasePage() {
    const navigate = useNavigate();
    const { success, error: showError } = useNotification();

    const [patientId, setPatientId] = useState('');
    const [patientAge, setPatientAge] = useState('');
    const [patientSex, setPatientSex] = useState<'M' | 'F' | ''>('');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [uploadedFiles, setUploadedFiles] = useState<Record<Modality, File | null>>({
        t1: null,
        t1ce: null,
        t2: null,
        flair: null,
    });

    const modalityLabels: Record<Modality, string> = {
        t1: 'T1-weighted',
        t1ce: 'T1-weighted Contrast Enhanced',
        t2: 'T2-weighted',
        flair: 'FLAIR',
    };

    // Auto-detect modality from filename
    const detectModality = (filename: string): Modality | null => {
        const lower = filename.toLowerCase();
        if (lower.includes('t1ce') || lower.includes('t1c')) return 't1ce';
        if (lower.includes('t1')) return 't1';
        if (lower.includes('t2')) return 't2';
        if (lower.includes('flair') || lower.includes('t2f')) return 'flair';
        return null;
    };

    // Handle bulk upload
    const handleBulkUpload = (files: FileList) => {
        const newFiles = { ...uploadedFiles };

        Array.from(files).forEach(file => {
            const modality = detectModality(file.name);
            if (modality) {
                newFiles[modality] = file;
            }
        });

        setUploadedFiles(newFiles);
    };

    const handleFileUpload = (modality: Modality, file: File | null) => {
        setUploadedFiles(prev => ({ ...prev, [modality]: file }));
    };

    const handleRemoveFile = (modality: Modality) => {
        setUploadedFiles(prev => ({ ...prev, [modality]: null }));
    };

    const handleSubmit = async () => {
        if (!canSubmit) return;

        setIsSubmitting(true);

        try {
            // Create the case
            const caseData = {
                patientId: patientId,
                age: parseInt(patientAge),
                sex: patientSex,
                status: 'uploading',
            };

            const createdCase = await apiService.createCase(caseData);
            success('Case created successfully!');

            // Upload MRI images
            const uploadPromises = Object.entries(uploadedFiles)
                .filter(([_, file]) => file !== null)
                .map(([modality, file]) =>
                    apiService.uploadMRIImage(createdCase.caseId, file!, modality)
                );

            await Promise.all(uploadPromises);
            success(`Successfully uploaded ${uploadPromises.length} MRI scans!`);

            // Navigate to case details
            navigate(`/cases/${createdCase.caseId}`);
        } catch (err: any) {
            showError(err.message || 'Failed to create case');
        } finally {
            setIsSubmitting(false);
        }
    };

    const uploadedCount = Object.values(uploadedFiles).filter(Boolean).length;
    const canSubmit = patientId && patientAge && patientSex && uploadedCount >= 1 && !isSubmitting;

    return (
        <div className="new-case-page">
            <div className="page-header">
                <div>
                    <h1 className="page-title">New Case</h1>
                    <p className="page-subtitle">Upload MRI scans for analysis</p>
                </div>
            </div>

            <div className="case-form-grid">
                {/* Patient Information */}
                <div className="card">
                    <div className="card-header">
                        <h3>Patient Information</h3>
                    </div>
                    <div className="card-body">
                        <div className="form-grid">
                            <div className="form-group">
                                <label className="form-label">Patient ID</label>
                                <input
                                    type="text"
                                    value={patientId}
                                    onChange={(e) => setPatientId(e.target.value)}
                                    placeholder="e.g., PATIENT-001"
                                    required
                                />
                            </div>

                            <div className="form-group">
                                <label className="form-label">Age (years)</label>
                                <input
                                    type="number"
                                    value={patientAge}
                                    onChange={(e) => setPatientAge(e.target.value)}
                                    placeholder="e.g., 58"
                                    min="1"
                                    max="120"
                                    required
                                />
                            </div>

                            <div className="form-group">
                                <label className="form-label">Sex</label>
                                <select
                                    value={patientSex}
                                    onChange={(e) => setPatientSex(e.target.value as 'M' | 'F')}
                                    required
                                >
                                    <option value="">Select...</option>
                                    <option value="M">Male</option>
                                    <option value="F">Female</option>
                                </select>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Bulk Upload Option */}
                <div className="card">
                    <div className="card-header">
                        <div className="flex justify-between items-center">
                            <h3>Upload MRI Scans</h3>
                            <span className="upload-count">
                                {uploadedCount}/4 uploaded
                            </span>
                        </div>
                    </div>
                    <div className="card-body">
                        <div className="bulk-upload-section">
                            <input
                                type="file"
                                id="bulk-upload"
                                multiple
                                accept=".nii,.nii.gz"
                                onChange={(e) => e.target.files && handleBulkUpload(e.target.files)}
                                className="upload-input"
                            />
                            <label htmlFor="bulk-upload" className="bulk-upload-label">
                                <Upload size={32} className="bulk-upload-icon" />
                                <div className="bulk-upload-text">
                                    <span className="bulk-upload-title">Drop files or click to upload</span>
                                    <span className="bulk-upload-hint">
                                        Upload multiple files at once - we'll auto-detect modalities from filenames (t1, t1ce, t2, flair/t2f)
                                    </span>
                                </div>
                            </label>
                        </div>

                        <div className="upload-divider">
                            <span>or upload individually</span>
                        </div>

                        {/* Individual Uploads */}
                        <div className="upload-grid">
                            {(Object.keys(modalityLabels) as Modality[]).map((modality) => (
                                <UploadCard
                                    key={modality}
                                    modality={modality}
                                    label={modalityLabels[modality]}
                                    file={uploadedFiles[modality]}
                                    onFileChange={(file) => handleFileUpload(modality, file)}
                                    onRemove={() => handleRemoveFile(modality)}
                                />
                            ))}
                        </div>

                        <div className="upload-info">
                            <AlertCircle size={16} />
                            <p>
                                Supported formats: NIfTI (.nii, .nii.gz). Maximum size: 500MB per file.
                                <br />
                                <strong>Tip:</strong> Include modality name in filename (e.g., patient_t1.nii.gz, case_t1ce.nii.gz, scan_t2f.nii.gz)
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Actions */}
            <div className="actions-bar">
                <button className="btn btn-outline" onClick={() => navigate('/cases')}>
                    Cancel
                </button>
                <button
                    className="btn btn-primary btn-lg"
                    disabled={!canSubmit}
                    onClick={handleSubmit}
                >
                    {isSubmitting ? 'Creating Case...' : 'Create Case & Start Analysis'}
                </button>
            </div>
        </div>
    );
}

interface UploadCardProps {
    modality: string;
    label: string;
    file: File | null;
    onFileChange: (file: File | null) => void;
    onRemove: () => void;
}

function UploadCard({ modality, label, file, onFileChange, onRemove }: UploadCardProps) {
    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0] || null;
        onFileChange(selectedFile);
    };

    return (
        <div className={`upload-card ${file ? 'uploaded' : ''}`}>
            <input
                type="file"
                id={`upload-${modality}`}
                accept=".nii,.nii.gz"
                onChange={handleChange}
                className="upload-input"
            />
            <label htmlFor={`upload-${modality}`} className="upload-label">
                {file ? (
                    <>
                        <CheckCircle2 size={32} className="upload-icon success" />
                        <span className="upload-text">
                            <span className="upload-title">{label}</span>
                            <span className="upload-filename">{file.name}</span>
                        </span>
                        <button
                            type="button"
                            className="remove-file-btn"
                            onClick={(e) => {
                                e.preventDefault();
                                onRemove();
                            }}
                        >
                            <X size={16} />
                        </button>
                    </>
                ) : (
                    <>
                        <Upload size={32} className="upload-icon" />
                        <span className="upload-text">
                            <span className="upload-title">{label}</span>
                            <span className="upload-hint">Click to upload</span>
                        </span>
                    </>
                )}
            </label>
        </div>
    );
}
