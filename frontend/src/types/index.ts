// User and Authentication Types
export interface User {
    id: string;
    username: string;
    email: string;
    firstName: string;
    lastName: string;
    role: 'doctor' | 'radiologist' | 'researcher' | 'admin';
    specialty?: string;
    institution?: string;
    phoneNumber?: string;
    isEmailVerified: boolean;
    createdAt: string;
    updatedAt: string;
}

export interface SignUpData {
    email: string;
    password: string;
    confirmPassword: string;
    firstName: string;
    lastName: string;
    role: 'doctor' | 'radiologist' | 'researcher';
    specialty?: string;
    institution?: string;
}

export interface AuthState {
    user: User | null;
    token: string | null;
    isAuthenticated: boolean;
    isLoading: boolean;
}

// Case Management Types
export interface Case {
    id: string;
    caseId: string;
    patientName: string;
    patientAge: number;
    patientSex: 'M' | 'F' | 'O';
    status: 'created' | 'uploaded' | 'processing' | 'completed' | 'error';
    createdAt: string;
    updatedAt: string;
    clinician: User;
    hasT1: boolean;
    hasT1ce: boolean;
    hasT2: boolean;
    hasFlair: boolean;
}

// Segmentation Types
export interface SegmentationResult {
    id: string;
    caseId: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    progress: number;
    volumeWT: number | null;  // Whole Tumor volume (mm³)
    volumeTC: number | null;  // Tumor Core volume (mm³)
    volumeET: number | null;  // Enhancing Tumor volume (mm³)
    diceWT: number | null;
    diceTC: number | null;
    diceET: number | null;
    maskUrl: string | null;
    glbUrl: string | null;
    createdAt: string;
    completedAt: string | null;
    errorMessage: string | null;
}

// Report Types
export interface Report {
    id: string;
    caseId: string;
    content: string;
    isEdited: boolean;
    originalContent: string;
    status: 'generating' | 'ready' | 'approved';
    generatedAt: string;
    approvedAt: string | null;
    approvedBy: User | null;
    pdfUrl: string | null;
}

// Feedback Types
export interface Feedback {
    id: string;
    caseId: string;
    feedbackType: 'segmentation_error' | 'report_error' | 'suggestion';
    description: string;
    correctionMask: string | null;
    createdAt: string;
    clinician: User;
}

// Upload Types
export interface MRIUpload {
    t1?: File;
    t1ce?: File;
    t2?: File;
    flair?: File;
}

export interface UploadProgress {
    t1: number;
    t1ce: number;
    t2: number;
    flair: number;
}

// API Response Types
export interface ApiResponse<T> {
    success: boolean;
    data: T;
    message?: string;
}

export interface PaginatedResponse<T> {
    count: number;
    next: string | null;
    previous: string | null;
    results: T[];
}

// Dashboard Stats
export interface DashboardStats {
    totalCases: number;
    completedCases: number;
    pendingCases: number;
    averageProcessingTime: number;
    recentCases: Case[];
}

// Admin Stats
export interface AdminStats {
    totalUsers: number;
    totalCases: number;
    totalReports: number;
    usersByRole: Record<string, number>;
    recentUsers: User[];
    topUsers: {
        user: User;
        casesCount: number;
        reportsCount: number;
    }[];
}
