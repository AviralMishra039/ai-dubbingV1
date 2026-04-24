import { useState, useEffect } from 'react';

const STAGES = [
  { name: 'Scene Analysis', icon: '🎬', desc: 'Analyzing video content and speakers' },
  { name: 'Transcription', icon: '📝', desc: 'Extracting speech with AI models' },
  { name: 'Translation', icon: '🌐', desc: 'Translating dialogue with context' },
  { name: 'Voice Synthesis', icon: '🎙️', desc: 'Generating natural speech audio' },
  { name: 'Assembly', icon: '🎞️', desc: 'Building the final dubbed video' },
];

export default function Processing({ direction }) {
  const [elapsed, setElapsed] = useState(0);
  const [activeStage, setActiveStage] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsed((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Simulate stage progression for visual feedback
  useEffect(() => {
    const stageTimer = setInterval(() => {
      setActiveStage((prev) => (prev < STAGES.length - 1 ? prev + 1 : prev));
    }, 45000); // ~45 seconds per stage estimate
    return () => clearInterval(stageTimer);
  }, []);

  const formatTime = (secs) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const directionText = direction === 'hi_to_en' ? 'Hindi → English' : 'English → Hindi';

  return (
    <div className="processing-container">
      <div className="processing-header">
        <div className="processing-spinner">
          <div className="spinner-ring spinner-ring--outer"></div>
          <div className="spinner-ring spinner-ring--middle"></div>
          <div className="spinner-ring spinner-ring--inner"></div>
          <div className="spinner-core">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <path d="M8 10L14 6L20 10V18L14 22L8 18V10Z" stroke="white" strokeWidth="1.5" fill="none" />
              <circle cx="14" cy="14" r="2" fill="white" />
            </svg>
          </div>
        </div>
        <h2 className="processing-title">Dubbing in Progress</h2>
        <p className="processing-direction">{directionText}</p>
        <p className="processing-time">Elapsed: {formatTime(elapsed)}</p>
      </div>

      {/* Stage Progress */}
      <div className="stage-progress">
        {STAGES.map((stage, i) => (
          <div
            key={stage.name}
            className={`stage-item ${
              i < activeStage ? 'stage-item--done' : 
              i === activeStage ? 'stage-item--active' : 
              'stage-item--pending'
            }`}
          >
            <div className="stage-indicator">
              {i < activeStage ? (
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <path d="M3 8L6 11L13 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              ) : (
                <span className="stage-number">{i + 1}</span>
              )}
            </div>
            <div className="stage-details">
              <span className="stage-icon">{stage.icon}</span>
              <div className="stage-text">
                <span className="stage-name">{stage.name}</span>
                <span className="stage-desc">{stage.desc}</span>
              </div>
            </div>
            {i < STAGES.length - 1 && <div className="stage-connector"></div>}
          </div>
        ))}
      </div>

      <div className="processing-notice">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
          <path d="M8 5V9M8 11V11.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
        <p>This usually takes 5–15 minutes depending on video length. Please keep this tab open.</p>
      </div>
    </div>
  );
}
