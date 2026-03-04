/**
 * ProtectedRoute — wraps a route so only users with the allowed roles can access it.
 *
 * Usage:
 *   <ProtectedRoute allowedRoles={['admin']}>
 *     <AdminDashboardPage />
 *   </ProtectedRoute>
 *
 * If unauthenticated  → redirect to /login
 * If wrong role       → show an "Access Restricted" screen (no redirect,
 *                        so the user knows why they can't see the page)
 */

import { type ReactNode } from 'react';
import { Navigate } from 'react-router-dom';
import { ShieldAlert } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

interface ProtectedRouteProps {
    children: ReactNode;
    /** Roles that are permitted. Empty / undefined = any authenticated user. */
    allowedRoles?: string[];
}

export default function ProtectedRoute({ children, allowedRoles }: ProtectedRouteProps) {
    const { user, isAuthenticated, isLoading } = useAuth();

    // While loading, render nothing (DashboardLayout shows its own spinner)
    if (isLoading) return null;

    // Not logged in → go to login
    if (!isAuthenticated) return <Navigate to="/login" replace />;

    // Role check
    if (allowedRoles && allowedRoles.length > 0) {
        const userRole = user?.role ?? '';
        if (!allowedRoles.includes(userRole)) {
            return <AccessDenied />;
        }
    }

    return <>{children}</>;
}


function AccessDenied() {
    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '60vh',
            gap: '1rem',
            textAlign: 'center',
            padding: '2rem',
        }}>
            <ShieldAlert size={64} style={{ color: 'var(--color-error, #EF4444)', opacity: 0.7 }} />
            <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                Access Restricted
            </h2>
            <p style={{ color: 'var(--text-secondary)', maxWidth: 380, lineHeight: 1.6 }}>
                You don't have permission to view this page.
                Please contact your administrator if you believe this is a mistake.
            </p>
            <a
                href="/dashboard"
                style={{
                    marginTop: '0.5rem',
                    padding: '0.55rem 1.4rem',
                    borderRadius: '0.5rem',
                    background: 'var(--color-primary)',
                    color: '#fff',
                    fontWeight: 600,
                    textDecoration: 'none',
                    fontSize: '0.9rem',
                }}
            >
                ← Back to Dashboard
            </a>
        </div>
    );
}
