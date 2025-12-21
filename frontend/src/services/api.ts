/**
 * API Service for Medical AI Dashboard
 * Handles all HTTP requests to the Django backend
 */

import type { User } from '../types';

const API_BASE_URL = 'http://localhost:8000/api';

interface LoginResponse {
    user: User;
    access: string;
    refresh: string;
}

interface RegisterResponse {
    user: User;
    access: string;
    refresh: string;
    message: string;
}

class ApiService {
    private accessToken: string | null = null;
    private refreshToken: string | null = null;

    constructor() {
        // Load tokens from localStorage on initialization
        this.accessToken = localStorage.getItem('access_token');
        this.refreshToken = localStorage.getItem('refresh_token');
    }

    private async request(endpoint: string, options: RequestInit = {}) {
        const url = `${API_BASE_URL}${endpoint}`;

        const headers: Record<string, string> = {
            'Content-Type': 'application/json',
            ...(options.headers as Record<string, string>),
        };

        // Add auth token if available
        if (this.accessToken) {
            headers['Authorization'] = `Bearer ${this.accessToken}`;
        }

        const config: RequestInit = {
            ...options,
            headers,
        };

        try {
            const response = await fetch(url, config);

            // Handle 401 Unauthorized - try to refresh token
            if (response.status === 401 && this.refreshToken) {
                const refreshed = await this.refreshAccessToken();
                if (refreshed) {
                    // Retry the original request with new token
                    const retryHeaders: Record<string, string> = {
                        ...headers,
                        'Authorization': `Bearer ${this.accessToken}`,
                    };
                    const retryResponse = await fetch(url, { ...config, headers: retryHeaders });
                    return await this.handleResponse(retryResponse);
                }
            }

            return await this.handleResponse(response);
        } catch (error) {
            console.error('API Request failed:', error);
            throw error;
        }
    }

    private async handleResponse(response: Response) {
        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Request failed' }));

            // If error is validation errors object, stringify it for parsing later
            if (error && typeof error === 'object' && !error.error && !error.message) {
                throw new Error(JSON.stringify(error));
            }

            throw new Error(error.error || error.message || `HTTP ${response.status}`);
        }

        // Handle 204 No Content
        if (response.status === 204) {
            return null;
        }

        return await response.json();
    }

    private async refreshAccessToken(): Promise<boolean> {
        try {
            const response = await fetch(`${API_BASE_URL}/users/refresh/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ refresh: this.refreshToken }),
            });

            if (response.ok) {
                const data = await response.json();
                this.setTokens(data.access, this.refreshToken!);
                return true;
            }

            // Refresh failed - clear tokens
            this.clearTokens();
            return false;
        } catch {
            this.clearTokens();
            return false;
        }
    }

    private setTokens(access: string, refresh: string) {
        this.accessToken = access;
        this.refreshToken = refresh;
        localStorage.setItem('access_token', access);
        localStorage.setItem('refresh_token', refresh);
    }

    private clearTokens() {
        this.accessToken = null;
        this.refreshToken = null;
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
    }

    // Authentication APIs
    async login(email: string, password: string): Promise<LoginResponse> {
        const data = await this.request('/users/login/', {
            method: 'POST',
            body: JSON.stringify({ email, password }),
        });

        this.setTokens(data.access, data.refresh);
        return data;
    }

    async register(userData: {
        email: string;
        password: string;
        confirm_password: string;
        first_name: string;
        last_name: string;
        role: string;
        specialty?: string;
        institution?: string;
    }): Promise<RegisterResponse> {
        const data = await this.request('/users/register/', {
            method: 'POST',
            body: JSON.stringify(userData),
        });

        this.setTokens(data.access, data.refresh);
        return data;
    }

    async logout(): Promise<void> {
        try {
            if (this.refreshToken) {
                await this.request('/users/logout/', {
                    method: 'POST',
                    body: JSON.stringify({ refresh: this.refreshToken }),
                });
            }
        } finally {
            this.clearTokens();
        }
    }

    // User Profile APIs
    async getProfile(): Promise<User> {
        return await this.request('/users/profile/');
    }

    async updateProfile(updates: Partial<User>): Promise<{ user: User; message: string }> {
        return await this.request('/users/profile/update/', {
            method: 'PATCH',
            body: JSON.stringify(updates),
        });
    }

    async changePassword(currentPassword: string, newPassword: string, confirmPassword: string): Promise<{ message: string }> {
        return await this.request('/users/profile/change-password/', {
            method: 'POST',
            body: JSON.stringify({
                currentPassword,
                newPassword,
                confirmPassword,
            }),
        });
    }

    // Admin APIs
    async getAllUsers(): Promise<{ users: User[]; stats: any }> {
        return await this.request('/users/users/');
    }

    // Cases APIs
    async getCases(): Promise<any[]> {
        return await this.request('/cases/');
    }

    async createCase(caseData: any): Promise<any> {
        return await this.request('/cases/', {
            method: 'POST',
            body: JSON.stringify(caseData),
        });
    }

    async getCase(caseId: string): Promise<any> {
        return await this.request(`/cases/${caseId}/`);
    }

    async deleteCase(caseId: string): Promise<void> {
        return await this.request(`/cases/${caseId}/delete/`, {
            method: 'DELETE',
        });
    }

    async uploadMRIImage(caseId: string, file: File, modality: string): Promise<any> {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('modality', modality);

        const url = `${API_BASE_URL}/cases/${caseId}/upload/`;
        const headers: Record<string, string> = {};

        if (this.accessToken) {
            headers['Authorization'] = `Bearer ${this.accessToken}`;
        }

        const response = await fetch(url, {
            method: 'POST',
            headers,
            body: formData,
        });

        return await this.handleResponse(response);
    }

    async getMRIImages(caseId: string): Promise<any[]> {
        return await this.request(`/cases/${caseId}/images/`);
    }

    // Check if user is authenticated
    isAuthenticated(): boolean {
        return !!this.accessToken;
    }
}

// Export singleton instance
export const apiService = new ApiService();
