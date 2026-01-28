# NeuroAI Dashboard - Frontend

A modern, responsive React application for managing medical cases, viewing 3D MRI scans, and interacting with the medical AI ecosystem for brain tumor segmentation and analysis.

## Features

- **User Authentication**: JWT-based authentication with role-based access control (Doctor, Radiologist, Admin)
- **Case Management**: Create, view, update, and delete medical cases
- **3D MRI Visualization**: Interactive 3D MRI viewer powered by Niivue with support for multiple modalities (T1, T2, T1CE, FLAIR)
- **Dashboard Analytics**: Overview of cases, statistics, and recent activity
- **User Settings**: Profile management and password change functionality
- **Admin Dashboard**: User management and system statistics (Admin only)
- **Responsive Design**: Fully responsive UI with modern aesthetics and dark theme support
- **Real-time Notifications**: Toast notifications for user feedback

## Prerequisites

- Node.js 18+ and npm/yarn
- Backend API running at `http://localhost:8000` (or configured URL)

## Installation

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Configure environment variables:**
   Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and update the API URL if needed:
   ```env
   VITE_API_BASE_URL=http://localhost:8000/api
   ```

## Running the Application

### Development Mode
Start the development server with hot module replacement:
```bash
npm run dev
```

The application will be available at `http://localhost:5173` (or the next available port).

### Production Build
Build the application for production:
```bash
npm run build
```

The optimized build will be created in the `dist/` directory.

### Preview Production Build
Preview the production build locally:
```bash
npm run preview
```

## Project Structure

```
frontend/
├── public/                 # Static assets
├── src/
│   ├── components/        # Reusable components
│   │   └── MRIViewer.tsx  # 3D MRI visualization component
│   ├── contexts/          # React contexts
│   │   ├── AuthContext.tsx           # Authentication state management
│   │   └── NotificationContext.tsx   # Toast notification system
│   ├── layouts/           # Layout components
│   │   └── DashboardLayout.tsx       # Main dashboard layout with sidebar
│   ├── pages/             # Page components
│   │   ├── LandingPage.tsx           # Landing/home page
│   │   ├── LoginPage.tsx             # User login
│   │   ├── SignUpPage.tsx            # User registration
│   │   ├── DashboardPage.tsx         # Dashboard overview
│   │   ├── CasesPage.tsx             # Cases listing
│   │   ├── NewCasePage.tsx           # Create new case
│   │   ├── CaseDetailsPage.tsx       # Case details with MRI viewer
│   │   ├── SettingsPage.tsx          # User settings
│   │   └── AdminDashboardPage.tsx    # Admin dashboard
│   ├── services/          # API services
│   │   └── api.ts         # API client with authentication
│   ├── types/             # TypeScript type definitions
│   │   └── index.ts       # Shared types
│   ├── App.tsx            # Main app component with routing
│   ├── main.tsx           # Application entry point
│   └── index.css          # Global styles and design system
├── index.html             # HTML template
├── package.json           # Dependencies and scripts
├── tsconfig.json          # TypeScript configuration
└── vite.config.ts         # Vite configuration
```

## Key Technologies

- **React 19**: Modern React with hooks
- **TypeScript**: Type-safe development
- **Vite**: Fast build tool and dev server
- **React Router**: Client-side routing
- **Niivue**: 3D medical image visualization
- **Lucide React**: Modern icon library
- **CSS Modules**: Component-scoped styling

## Key Features Explained

### Authentication System
The app uses JWT tokens (access and refresh) stored in localStorage. The `AuthContext` provides global authentication state and methods (`login`, `logout`, `isAuthenticated`).

### API Service
The `api.ts` service handles all HTTP requests with automatic:
- Token injection in request headers
- Token refresh on 401 errors
- Error handling and response parsing

### 3D MRI Viewer
Built with Niivue library, supports:
- Multiple MRI modalities (T1, T2, T1CE, FLAIR)
- Volume rendering and slice views
- Interactive controls for brightness, contrast, and opacity
- Crosshair positioning and orientation views

### Role-Based Access Control
Three user roles:
- **Doctor**: Create and manage their own cases
- **Radiologist**: View and analyze cases
- **Admin**: Full access including user management

## Development

### Code Linting
Run ESLint to check code quality:
```bash
npm run lint
```

### Type Checking
TypeScript compilation:
```bash
npx tsc --noEmit
```

### Code Style Guidelines
- Use functional components with hooks
- Follow TypeScript best practices
- Maintain consistent CSS custom properties for theming
- Use semantic HTML elements
- Keep components small and focused

## API Integration

The frontend communicates with the Django REST API backend. Key endpoints:

- **Auth**: `/api/users/login/`, `/api/users/register/`, `/api/users/logout/`
- **Cases**: `/api/cases/`, `/api/cases/{id}/`
- **MRI Images**: `/api/cases/{id}/upload/`, `/api/cases/{id}/images/`
- **Users**: `/api/users/profile/`, `/api/users/users/` (admin)

## Responsive Design

The application is fully responsive with breakpoints:
- Mobile: < 768px
- Tablet: 768px - 1024px
- Desktop: > 1024px

## Theming

Global CSS variables defined in `index.css`:
```css
--primary-color: #3b82f6
--secondary-color: #1e293b
--background-color: #0f172a
--surface-color: #1e293b
--text-primary: #f1f5f9
--text-secondary: #94a3b8
```

## Future Enhancements

- **Real-time Collaboration**: WebSocket integration for live case updates and collaborative viewing
- **Advanced Segmentation Visualization**: Overlay segmentation masks on MRI scans with color-coded tumor regions
- **Report Generation UI**: Interactive interface for generating and editing medical reports
- **Offline Support**: PWA capabilities with offline case viewing
- **Internationalization**: Multi-language support
- **Advanced Filtering**: Complex filtering and search across cases
- **Export Functionality**: Export case data and reports in various formats (PDF, DICOM)
- **Annotation Tools**: Drawing and annotation tools for radiologists
- **3D Reconstruction**: Enhanced 3D rendering with custom shaders
- **Performance Monitoring**: Client-side performance tracking and error reporting
- **Accessibility**: WCAG 2.1 AA compliance

## License

Part of the Expert-Guided Multimodal AI Ecosystem project.
