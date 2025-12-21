import { useEffect, useRef, useState } from 'react';
import { X, Download, RotateCcw, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
import { Niivue } from '@niivue/niivue';
import './MRIViewer.css';

interface MRIViewerProps {
    imageUrl: string;
    modality: string;
    onClose: () => void;
}

export default function MRIViewer({ imageUrl, modality, onClose }: MRIViewerProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [nv, setNv] = useState<Niivue | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [sliceType, setSliceType] = useState(0); // Track current slice type

    useEffect(() => {
        if (!canvasRef.current) return;

        // Initialize Niivue
        const niivue = new Niivue({
            backColor: [0, 0, 0, 1],
        });

        niivue.attachToCanvas(canvasRef.current);
        setNv(niivue);

        // Load the MRI image
        loadImage(niivue);

        return () => {
            // Remove all loaded volumes on cleanup
            while (niivue.volumes.length > 0) {
                niivue.removeVolumeByIndex(0);
            }
        };
    }, []);

    const loadImage = async (niivue: Niivue) => {
        try {
            setIsLoading(true);
            setError(null);

            console.log('Loading MRI image from URL:', imageUrl);

            // Try loading with just the URL string array format
            // Add a timeout to prevent infinite loading
            const loadPromise = niivue.loadVolumes([{ url: imageUrl }]);
            const timeoutPromise = new Promise((_, reject) =>
                setTimeout(() => reject(new Error('Loading timeout - file may be too large or inaccessible')), 30000)
            );

            await Promise.race([loadPromise, timeoutPromise]);

            // Set the colormap after loading
            if (niivue.volumes && niivue.volumes.length > 0) {
                niivue.volumes[0].colormap = 'gray';
            }

            console.log('MRI image loaded successfully');
            setIsLoading(false);
        } catch (err: any) {
            console.error('Error loading MRI image:', err);
            console.error('Error details:', err.message, err.stack);
            setError(`Failed to load MRI image: ${err.message || 'Unknown error'}. Please ensure the file is accessible and in NIfTI format.`);
            setIsLoading(false);
        }
    };

    const handleReset = () => {
        if (nv) {
            nv.setScale(1);
            nv.setSliceType(0); // 0 = multiplanar view
            setSliceType(0);
        }
    };

    const handleZoomIn = () => {
        if (nv) {
            const currentScale = nv.scene?.volScaleMultiplier || 1;
            nv.setScale(currentScale * 1.2);
        }
    };

    const handleZoomOut = () => {
        if (nv) {
            const currentScale = nv.scene?.volScaleMultiplier || 1;
            nv.setScale(currentScale / 1.2);
        }
    };

    const handleToggleView = () => {
        if (nv) {
            // Cycle through different view modes: 0=multiplanar, 1=axial, 2=coronal, 3=sagittal
            const nextType = sliceType === 0 ? 1 : sliceType === 1 ? 2 : sliceType === 2 ? 3 : 0;
            nv.setSliceType(nextType);
            setSliceType(nextType);
        }
    };

    const handleDownload = () => {
        // Download the original file
        const link = document.createElement('a');
        link.href = imageUrl;
        link.download = `${modality}_mri.nii.gz`;
        link.click();
    };

    return (
        <div className="mri-viewer-overlay">
            <div className="mri-viewer-container">
                {/* Header */}
                <div className="mri-viewer-header">
                    <div>
                        <h2 className="mri-viewer-title">MRI Viewer - {modality.toUpperCase()}</h2>
                        <p className="mri-viewer-subtitle">3D Medical Image Visualization</p>
                    </div>
                    <button className="mri-viewer-close" onClick={onClose}>
                        <X size={24} />
                    </button>
                </div>

                {/* Controls */}
                <div className="mri-viewer-controls">
                    <button className="viewer-control-btn" onClick={handleReset} title="Reset View">
                        <RotateCcw size={18} />
                        Reset
                    </button>
                    <button className="viewer-control-btn" onClick={handleZoomIn} title="Zoom In">
                        <ZoomIn size={18} />
                        Zoom In
                    </button>
                    <button className="viewer-control-btn" onClick={handleZoomOut} title="Zoom Out">
                        <ZoomOut size={18} />
                        Zoom Out
                    </button>
                    <button className="viewer-control-btn" onClick={handleToggleView} title="Toggle View">
                        <Maximize2 size={18} />
                        Toggle View
                    </button>
                    <button className="viewer-control-btn viewer-control-primary" onClick={handleDownload} title="Download">
                        <Download size={18} />
                        Download
                    </button>
                </div>

                {/* Canvas */}
                <div className="mri-viewer-canvas-container">
                    {isLoading && (
                        <div className="mri-viewer-loading">
                            <div className="loading-spinner"></div>
                            <p>Loading MRI image...</p>
                        </div>
                    )}

                    {error && (
                        <div className="mri-viewer-error">
                            <p>{error}</p>
                        </div>
                    )}

                    <canvas
                        ref={canvasRef}
                        className="mri-viewer-canvas"
                        style={{ display: isLoading || error ? 'none' : 'block' }}
                    />
                </div>

                {/* Info */}
                <div className="mri-viewer-info">
                    <div className="viewer-info-item">
                        <span className="viewer-info-label">Modality:</span>
                        <span className="viewer-info-value">{modality.toUpperCase()}</span>
                    </div>
                    <div className="viewer-info-item">
                        <span className="viewer-info-label">Controls:</span>
                        <span className="viewer-info-value">
                            Left Click: Crosshair • Scroll: Navigate Slices • Right Drag: Pan
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
}
