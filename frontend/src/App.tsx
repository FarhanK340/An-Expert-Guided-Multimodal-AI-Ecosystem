import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import { NotificationProvider } from './contexts/NotificationContext';
import DashboardLayout from './layouts/DashboardLayout';
import ProtectedRoute from './components/ProtectedRoute';
import LandingPage from './pages/LandingPage';
import LoginPage from './pages/LoginPage';
import SignUpPage from './pages/SignUpPage';
import DashboardPage from './pages/DashboardPage';
import CasesPage from './pages/CasesPage';
import NewCasePage from './pages/NewCasePage';
import CaseDetailsPage from './pages/CaseDetailsPage';
import ResultsPage from './pages/ResultsPage';
import ReportsPage from './pages/ReportsPage';
import ReportDetailPage from './pages/ReportDetailPage';
import SettingsPage from './pages/SettingsPage';
import AdminDashboardPage from './pages/AdminDashboardPage';
import './App.css';

/** Wrap a page inside DashboardLayout + ProtectedRoute in one line. */
function Layout({ children, allowedRoles }: { children: React.ReactNode; allowedRoles?: string[] }) {
  return (
    <DashboardLayout>
      <ProtectedRoute allowedRoles={allowedRoles}>
        {children}
      </ProtectedRoute>
    </DashboardLayout>
  );
}

const CLINICIAN = ['doctor', 'researcher', 'admin'];
const ALL_ROLES = ['doctor', 'researcher', 'patient', 'admin'];
const ADMIN_ONLY = ['admin'];

function App() {
  return (
    <BrowserRouter>
      <NotificationProvider>
        <AuthProvider>
          <Routes>
            {/* Public */}
            <Route path="/" element={<LandingPage />} />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/signup" element={<SignUpPage />} />

            {/* Dashboard — all authenticated roles */}
            <Route path="/dashboard" element={<Layout allowedRoles={ALL_ROLES}><DashboardPage /></Layout>} />

            {/* Cases — clinicians only (patients see their cases via Dashboard / Reports) */}
            <Route path="/cases" element={<Layout allowedRoles={CLINICIAN}><CasesPage /></Layout>} />
            <Route path="/cases/new" element={<Layout allowedRoles={CLINICIAN}><NewCasePage /></Layout>} />
            <Route path="/cases/:id" element={<Layout allowedRoles={CLINICIAN}><CaseDetailsPage /></Layout>} />
            <Route path="/cases/:id/results" element={<Layout allowedRoles={CLINICIAN}><ResultsPage /></Layout>} />

            {/* Reports — all roles (patients see only their own via API filter) */}
            <Route path="/reports" element={<Layout allowedRoles={ALL_ROLES}><ReportsPage /></Layout>} />
            <Route path="/reports/:id" element={<Layout allowedRoles={ALL_ROLES}><ReportDetailPage /></Layout>} />

            {/* Settings — all roles */}
            <Route path="/settings" element={<Layout allowedRoles={ALL_ROLES}><SettingsPage /></Layout>} />

            {/* Admin — admin only */}
            <Route path="/admin" element={<Layout allowedRoles={ADMIN_ONLY}><AdminDashboardPage /></Layout>} />

            {/* Catch-all */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </AuthProvider>
      </NotificationProvider>
    </BrowserRouter>
  );
}

export default App;
