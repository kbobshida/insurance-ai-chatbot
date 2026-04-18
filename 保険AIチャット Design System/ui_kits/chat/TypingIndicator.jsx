// TypingIndicator.jsx — three navy dots pulsing, no scale bounce
const TypingIndicator = () => (
  <div style={{
    display: 'flex', alignItems: 'flex-start',
    marginBottom: 20, gap: 12, maxWidth: '90%',
  }}>
    <Avatar kind="ai" />
    <div style={{
      padding: '14px 16px',
      background: '#fff',
      border: '1px solid #e6dfd5',
      borderTop: '1px solid #16375f',
      borderRadius: 4,
      display: 'inline-flex', alignItems: 'center', gap: 6,
    }}>
      {[0, 1, 2].map(i => (
        <span key={i} style={{
          width: 6, height: 6, borderRadius: '50%',
          background: '#16375f', display: 'inline-block',
          animation: `typePulse 1.2s infinite ease-in-out ${i * 0.2}s`,
        }} />
      ))}
    </div>
  </div>
);

window.TypingIndicator = TypingIndicator;
