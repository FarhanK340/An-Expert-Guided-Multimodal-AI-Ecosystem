import { useState, useEffect } from 'react';
import { Plus, Search, Filter } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { apiService } from '../services/api';
import './CasesPage.css';

export default function CasesPage() {
    const [searchQuery, setSearchQuery] = useState('');
    const [cases, setCases] = useState<any[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const navigate = useNavigate();

    useEffect(() => {
        const fetchCases = async () => {
            try {
                const data = await apiService.getCases();
                setCases(data);
            } catch (error) {
                console.error('Failed to fetch cases:', error);
            } finally {
                setIsLoading(false);
            }
        };

        fetchCases();
    }, []);

    // Filter cases based on search query
    const filteredCases = cases.filter(case_ =>
        case_.patientId?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        case_.status?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        case_.age?.toString().includes(searchQuery)
    );

    const getStatusBadge = (status: string) => {
        const badgeMap: Record<string, string> = {
            completed: 'badge-success',
            processing: 'badge-warning',
            pending: 'badge-neutral',
            uploaded: 'badge-neutral',
            created: 'badge-neutral',
            failed: 'badge-error',
        };
        return `badge ${badgeMap[status] || 'badge-neutral'}`;
    };

    return (
        <div className="cases-page">
            <div className="page-header">
                <div>
                    <h1 className="page-title">Cases</h1>
                    <p className="page-subtitle">Manage and analyze brain MRI cases</p>
                </div>
                <button className="btn btn-primary" onClick={() => navigate('/cases/new')}>
                    <Plus size={18} />
                    New Case
                </button>
            </div>

            {/* Search & Filters */}
            <div className="card search-card">
                <div className="search-wrapper">
                    <div className="search-input-wrapper">
                        <Search size={18} className="search-icon" />
                        <input
                            type="text"
                            placeholder="Search by case ID, age, or status..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="search-input"
                        />
                    </div>
                    <button className="btn btn-outline">
                        <Filter size={18} />
                        Filter
                    </button>
                </div>
            </div>

            {/* Cases Table */}
            <div className="card">
                <div className="table-wrapper">
                    {isLoading ? (
                        <div style={{ padding: '3rem', textAlign: 'center', color: '#6B7280' }}>
                            Loading cases...
                        </div>
                    ) : filteredCases.length === 0 ? (
                        <div style={{ padding: '3rem', textAlign: 'center', color: '#6B7280' }}>
                            {searchQuery ? 'No cases match your search.' : 'No cases found. Create your first case to get started.'}
                        </div>
                    ) : (
                        <table className="cases-table">
                            <thead>
                                <tr>
                                    <th>Case ID</th>
                                    <th>Patient Info</th>
                                    <th>Field Strength</th>
                                    <th>Status</th>
                                    <th>Created</th>
                                    <th></th>
                                </tr>
                            </thead>
                            <tbody>
                                {filteredCases.map((case_) => (
                                    <tr key={case_.caseId || case_.patientId}>
                                        <td>
                                            <span className="case-id">{case_.patientId}</span>
                                        </td>
                                        <td>
                                            <div className="patient-info">
                                                <span>{case_.age || 'N/A'} years</span>
                                                <span className="patient-divider">•</span>
                                                <span>{case_.sex === 'M' ? 'Male' : case_.sex === 'F' ? 'Female' : 'N/A'}</span>
                                            </div>
                                        </td>
                                        <td>
                                            <span className="modalities-count">{case_.fieldStrength || 'N/A'}</span>
                                        </td>
                                        <td>
                                            <span className={getStatusBadge(case_.status)}>
                                                {case_.status}
                                            </span>
                                        </td>
                                        <td className="text-neutral">
                                            {case_.createdAt ? new Date(case_.createdAt).toISOString().split('T')[0] : 'N/A'}
                                        </td>
                                        <td>
                                            <div className="actions-cell">
                                                <button
                                                    className="btn btn-sm btn-outline"
                                                    onClick={() => navigate(`/cases/${case_.caseId}`)}
                                                >
                                                    View Details
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                </div>
            </div>
        </div>
    );
}
