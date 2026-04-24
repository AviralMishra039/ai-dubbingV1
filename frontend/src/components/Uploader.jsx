import { useState, useRef, useCallback } from 'react';

const ACCEPTED_TYPES = ['.mp4', '.mkv', '.mov', '.avi', '.webm'];
const MAX_SIZE_MB = 500;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;

export default function Uploader({ onStartDubbing }) {
  const [file, setFile] = useState(null);
  const [direction, setDirection] = useState('hi_to_en');
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState('');
  const inputRef = useRef(null);

  const validateFile = useCallback((f) => {
    setError('');
    if (!f) return false;

    const ext = '.' + f.name.split('.').pop().toLowerCase();
    if (!ACCEPTED_TYPES.includes(ext)) {
      setError(`Unsupported format: ${ext}. Accepted: ${ACCEPTED_TYPES.join(', ')}`);
      return false;
    }

    if (f.size > MAX_SIZE_BYTES) {
      setError(`File too large (${(f.size / (1024 * 1024)).toFixed(1)} MB). Max: ${MAX_SIZE_MB} MB`);
      return false;
    }

    return true;
  }, []);

  const handleFile = useCallback((f) => {
    if (validateFile(f)) {
      setFile(f);
    }
  }, [validateFile]);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, [handleFile]);

  const handleInputChange = useCallback((e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  }, [handleFile]);

  const handleSubmit = () => {
    if (!file) {
      setError('Please select a video file');
      return;
    }
    onStartDubbing(file, direction);
  };

  const removeFile = () => {
    setFile(null);
    setError('');
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <div className="uploader-container">
      {/* Header */}
      <div className="uploader-header">
        <div className="logo-icon">
          <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
            <rect width="40" height="40" rx="12" fill="url(#logo-gradient)" />
            <path d="M12 14L20 10L28 14V26L20 30L12 26V14Z" stroke="white" strokeWidth="1.5" fill="none" />
            <path d="M16 18L20 16L24 18V24L20 26L16 24V18Z" fill="rgba(255,255,255,0.3)" stroke="white" strokeWidth="1" />
            <circle cx="20" cy="20" r="2" fill="white" />
            <defs>
              <linearGradient id="logo-gradient" x1="0" y1="0" x2="40" y2="40">
                <stop offset="0%" stopColor="#6C5CE7" />
                <stop offset="100%" stopColor="#A855F7" />
              </linearGradient>
            </defs>
          </svg>
        </div>
        <h1 className="app-title">DubGraph</h1>
        <p className="app-subtitle">AI-Powered Video Dubbing Pipeline</p>
      </div>

      {/* Drop Zone */}
      <div
        className={`drop-zone ${dragActive ? 'drop-zone--active' : ''} ${file ? 'drop-zone--has-file' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => !file && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(',')}
          onChange={handleInputChange}
          className="drop-zone__input"
        />

        {file ? (
          <div className="file-preview">
            <div className="file-icon">
              <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                <rect x="6" y="4" width="36" height="40" rx="4" stroke="currentColor" strokeWidth="2" fill="none" />
                <path d="M18 22L22 26L30 18" stroke="#10B981" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                <rect x="12" y="32" width="24" height="2" rx="1" fill="currentColor" opacity="0.3" />
                <rect x="12" y="36" width="16" height="2" rx="1" fill="currentColor" opacity="0.2" />
              </svg>
            </div>
            <p className="file-name">{file.name}</p>
            <p className="file-size">{(file.size / (1024 * 1024)).toFixed(1)} MB</p>
            <button className="file-remove" onClick={(e) => { e.stopPropagation(); removeFile(); }}>
              Remove
            </button>
          </div>
        ) : (
          <div className="drop-zone__content">
            <div className="upload-icon">
              <svg width="56" height="56" viewBox="0 0 56 56" fill="none">
                <circle cx="28" cy="28" r="27" stroke="currentColor" strokeWidth="1" strokeDasharray="4 4" opacity="0.4" />
                <path d="M28 18V38M18 28L28 18L38 28" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <p className="drop-zone__text">Drop your video here or click to upload</p>
            <p className="drop-zone__hint">
              MP4, MKV, MOV, AVI, WebM — Max {MAX_SIZE_MB} MB
            </p>
          </div>
        )}
      </div>

      {error && <div className="upload-error">{error}</div>}

      {/* Direction Selector */}
      <div className="direction-selector">
        <p className="direction-label">Dubbing Direction</p>
        <div className="direction-buttons">
          <button
            className={`direction-btn ${direction === 'hi_to_en' ? 'direction-btn--active' : ''}`}
            onClick={() => setDirection('hi_to_en')}
          >
            <span className="direction-flag">🇮🇳</span>
            <span className="direction-arrow">→</span>
            <span className="direction-flag">🇺🇸</span>
            <span className="direction-text">Hindi → English</span>
          </button>
          <button
            className={`direction-btn ${direction === 'en_to_hi' ? 'direction-btn--active' : ''}`}
            onClick={() => setDirection('en_to_hi')}
          >
            <span className="direction-flag">🇺🇸</span>
            <span className="direction-arrow">→</span>
            <span className="direction-flag">🇮🇳</span>
            <span className="direction-text">English → Hindi</span>
          </button>
        </div>
      </div>

      {/* Submit Button */}
      <button
        className={`submit-btn ${!file ? 'submit-btn--disabled' : ''}`}
        onClick={handleSubmit}
        disabled={!file}
      >
        <span className="submit-btn__icon">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path d="M3 10L17 10M12 5L17 10L12 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
        Start Dubbing
      </button>

      {/* Pipeline Info */}
      <div className="pipeline-info">
        <p className="pipeline-info__title">Pipeline Stages</p>
        <div className="pipeline-steps">
          {['Scene Analysis', 'Transcription', 'Translation', 'Voice Synthesis', 'Assembly'].map((step, i) => (
            <div key={step} className="pipeline-step">
              <div className="pipeline-step__number">{i + 1}</div>
              <span className="pipeline-step__name">{step}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
