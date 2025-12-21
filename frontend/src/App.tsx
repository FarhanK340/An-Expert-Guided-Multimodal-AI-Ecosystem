import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import { NotificationProvider } from './contexts/NotificationContext';
import DashboardLayout from './layouts/DashboardLayout';
import LandingPage from './pages/LandingPage';
import LoginPage from './pages/LoginPage';
import SignUpPage from './pages/SignUpPage';
import DashboardPage from './pages/DashboardPage';
import CasesPage from './pages/CasesPage';
import NewCasePage from './pages/NewCasePage';
import CaseDetailsPage from './pages/CaseDetailsPage';
import SettingsPage from './pages/SettingsPage';
import AdminDashboardPage from './pages/AdminDashboardPage';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <NotificationProvider>
        <AuthProvider>
          <Routes>
            {/* Public Routes */}
            <Route path="/" element={<LandingPage />} />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/signup" element={<SignUpPage />} />

            {/* Protected Routes */}
            <Route path="/dashboard" element={
              <DashboardLayout>
                <DashboardPage />
              </DashboardLayout>
            } />
            <Route path="/cases" element={
              <DashboardLayout>
                <CasesPage />
              </DashboardLayout>
            } />
            <Route path="/cases/new" element={
              <DashboardLayout>
                <NewCasePage />
              </DashboardLayout>
            } />
            <Route path="/cases/:id" element={
              <DashboardLayout>
                <CaseDetailsPage />
              </DashboardLayout>
            } />
            <Route path="/reports" element={
              <DashboardLayout>
                <div style={{ padding: '2rem' }}>
                  <h1>Reports</h1>
                  <p>Reports page coming soon...</p>
                </div>
              </DashboardLayout>
            } />
            <Route path="/settings" element={
              <DashboardLayout>
                <SettingsPage />
              </DashboardLayout>
            } />

            {/* Admin Routes */}
            <Route path="/admin" element={
              <DashboardLayout>
                <AdminDashboardPage />
              </DashboardLayout>
            } />
          </Routes>
        </AuthProvider>
      </NotificationProvider>
    </BrowserRouter>
  );
}

export default App;
