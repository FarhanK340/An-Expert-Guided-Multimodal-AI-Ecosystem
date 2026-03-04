import { createContext, useContext, useState, type ReactNode } from 'react';
import { X, CheckCircle, AlertCircle, Info, AlertTriangle } from 'lucide-react';
import './NotificationContext.css';

type NotificationType = 'success' | 'error' | 'info' | 'warning';

interface Notification {
    id: string;
    type: NotificationType;
    message: string;
    duration?: number;
}

interface NotificationContextType {
    showNotification: (type: NotificationType, message: string, duration?: number) => void;
    success: (message: string, duration?: number) => void;
    error: (message: string, duration?: number) => void;
    info: (message: string, duration?: number) => void;
    warning: (message: string, duration?: number) => void;
}

const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

export function NotificationProvider({ children }: { children: ReactNode }) {
    const [notifications, setNotifications] = useState<Notification[]>([]);

    const showNotification = (type: NotificationType, message: string, duration: number = 5000) => {
        const id = Math.random().toString(36).substr(2, 9);
        const notification: Notification = { id, type, message, duration };

        setNotifications((prev) => [...prev, notification]);

        if (duration > 0) {
            setTimeout(() => {
                removeNotification(id);
            }, duration);
        }
    };

    const removeNotification = (id: string) => {
        setNotifications((prev) => prev.filter((n) => n.id !== id));
    };

    const getIcon = (type: NotificationType) => {
        switch (type) {
            case 'success':
                return <CheckCircle size={20} />;
            case 'error':
                return <AlertCircle size={20} />;
            case 'warning':
                return <AlertTriangle size={20} />;
            case 'info':
                return <Info size={20} />;
        }
    };

    return (
        <NotificationContext.Provider
            value={{
                showNotification,
                success: (msg, dur) => showNotification('success', msg, dur),
                error: (msg, dur) => showNotification('error', msg, dur),
                info: (msg, dur) => showNotification('info', msg, dur),
                warning: (msg, dur) => showNotification('warning', msg, dur),
            }}
        >
            {children}

            {/* Notification Container */}
            <div className="notification-container">
                {notifications.map((notification) => (
                    <div
                        key={notification.id}
                        className={`notification notification-${notification.type}`}
                    >
                        <div className="notification-icon">
                            {getIcon(notification.type)}
                        </div>
                        <div className="notification-message">{notification.message}</div>
                        <button
                            className="notification-close"
                            onClick={() => removeNotification(notification.id)}
                            aria-label="Close notification"
                        >
                            <X size={18} />
                        </button>
                    </div>
                ))}
            </div>
        </NotificationContext.Provider>
    );
}

export function useNotification() {
    const context = useContext(NotificationContext);
    if (context === undefined) {
        throw new Error('useNotification must be used within a NotificationProvider');
    }
    return context;
}
