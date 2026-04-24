import { useState } from 'react';
import Uploader from './components/Uploader';
import Processing from './components/Processing';
import Result from './components/Result';
import './index.css';

const API_BASE = '';

function App() {
  const [view, setView] = useState('upload');
  const [direction, setDirection] = useState('hi_to_en');
  const [videoUrl, setVideoUrl] = useState('');
  const [errorInfo, setErrorInfo] = useState({ step: '', message: '' });

  const handleStartDubbing = async (file, dir) => {
    setDirection(dir);
    setView('processing');
    setErrorInfo({ step: '', message: '' });

    const formData = new FormData();
    formData.append('video', file);
    formData.append('direction', dir);

    try {
      const response = await fetch(`${API_BASE}/dub`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        try {
          const errorData = await response.json();
          setErrorInfo({
            step: errorData.step || 'unknown',
            message: errorData.message || 'An unknown error occurred',
          });
        } catch {
          setErrorInfo({
            step: 'network',
            message: `Server returned ${response.status}: ${response.statusText}`,
          });
        }
        setView('error');
        return;
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setVideoUrl(url);
      setView('result');
    } catch (err) {
      setErrorInfo({
        step: 'network',
        message: err.message || 'Network error — is the server running?',
      });
      setView('error');
    }
  };

  const handleReset = () => {
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl('');
    setView('upload');
    setErrorInfo({ step: '', message: '' });
  };

  return (
    <div className="app">
      <div className="bg-grid"></div>
      <div className="bg-glow bg-glow--1"></div>
      <div className="bg-glow bg-glow--2"></div>

      <main className="main-content">
        {view === 'upload' && <Uploader onStartDubbing={handleStartDubbing} />}
        {view === 'processing' && <Processing direction={direction} />}
        {view === 'result' && <Result videoUrl={videoUrl} onReset={handleReset} />}
        {view === 'error' && (
          <div className="error-container">
            <div className="error-icon">
              <svg width="56" height="56" viewBox="0 0 56 56" fill="none">
                <circle cx="28" cy="28" r="26" stroke="#EF4444" strokeWidth="2" fill="rgba(239,68,68,0.1)" />
                <path d="M20 20L36 36M36 20L20 36" stroke="#EF4444" strokeWidth="3" strokeLinecap="round" />
              </svg>
            </div>
            <h2 className="error-title">Something went wrong</h2>
            <div className="error-details">
              <p className="error-step"><strong>Failed at:</strong> {errorInfo.step}</p>
              <p className="error-message">{errorInfo.message}</p>
            </div>
            <button className="action-btn action-btn--reset" onClick={handleReset}>Try Again</button>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
