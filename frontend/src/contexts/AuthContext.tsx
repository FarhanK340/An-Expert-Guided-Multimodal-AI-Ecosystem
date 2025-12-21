import { createContext, useContext, useState, useEffect, type ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import { apiService } from '../services/api';
import type { User } from '../types';

interface AuthContextType {
    user: User | null;
    isAuthenticated: boolean;
    isLoading: boolean;
    login: (email: string, password: string) => Promise<void>;
    signup: (data: any) => Promise<void>;
    logout: () => void;
    updateUser: (updates: Partial<User>) => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
    const navigate = useNavigate();
    const [user, setUser] = useState<User | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    // Check authentication status on mount
    useEffect(() => {
        const checkAuth = async () => {
            if (apiService.isAuthenticated()) {
                try {
                    const profile = await apiService.getProfile();
                    setUser(profile);
                } catch (error) {
                    console.error('Failed to fetch profile:', error);
                    // Token might be invalid, clear it
                    await apiService.logout();
                }
            }
            setIsLoading(false);
        };

        checkAuth();
    }, []);

    const login = async (email: string, password: string) => {
        try {
            const response = await apiService.login(email, password);
            setUser(response.user);
            navigate('/dashboard');
        } catch (error: any) {
            throw new Error(error.message || 'Login failed');
        }
    };

    const signup = async (data: any) => {
        try {
            const payload = {
                email: data.email,
                password: data.password,
                confirm_password: data.confirmPassword,
                first_name: data.firstName,
                last_name: data.lastName,
                role: data.role,
                specialty: data.specialty || '',
                institution: data.institution || '',
            };

            const response = await apiService.register(payload);
            setUser(response.user);
            navigate('/dashboard');
        } catch (error: any) {
            throw new Error(error.message || 'Registration failed');
        }
    };

    const logout = async () => {
        await apiService.logout();
        setUser(null);
        navigate('/login');
    };

    const updateUser = async (updates: Partial<User>) => {
        try {
            const response = await apiService.updateProfile(updates);
            setUser(response.user);
        } catch (error: any) {
            throw new Error(error.message || 'Profile update failed');
        }
    };

    return (
        <AuthContext.Provider
            value={{
                user,
                isAuthenticated: !!user,
                isLoading,
                login,
                signup,
                logout,
                updateUser
            }}
        >
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
}
