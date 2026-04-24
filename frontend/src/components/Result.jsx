import { useRef, useState, useEffect } from 'react';

export default function Result({ videoUrl, onReset }) {
  const videoRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    // Auto-scroll to the result when it appears
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);

  const handleDownload = () => {
    const a = document.createElement('a');
    a.href = videoUrl;
    a.download = 'dubbed_video.mp4';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="result-container">
      <div className="result-header">
        <div className="result-success-icon">
          <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
            <circle cx="24" cy="24" r="22" stroke="#10B981" strokeWidth="2" fill="rgba(16, 185, 129, 0.1)" />
            <path d="M15 24L21 30L33 18" stroke="#10B981" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        <h2 className="result-title">Dubbing Complete!</h2>
        <p className="result-subtitle">Your video has been successfully dubbed.</p>
      </div>

      {/* Video Player */}
      <div className="video-player-wrapper">
        <video
          ref={videoRef}
          src={videoUrl}
          controls
          className="video-player"
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
        >
          Your browser does not support the video element.
        </video>
      </div>

      {/* Action Buttons */}
      <div className="result-actions">
        <button className="action-btn action-btn--download" onClick={handleDownload}>
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path d="M10 3V14M5 9L10 14L15 9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M3 17H17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
          Download Dubbed Video
        </button>
        <button className="action-btn action-btn--reset" onClick={onReset}>
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path d="M4 10C4 6.68629 6.68629 4 10 4C12.2091 4 14.1175 5.22675 15.0711 7.04919" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            <path d="M16 10C16 13.3137 13.3137 16 10 16C7.79086 16 5.88252 14.7733 4.92893 12.9508" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            <path d="M15 4V7H12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M5 16V13H8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Dub Another Video
        </button>
      </div>
    </div>
  );
}
