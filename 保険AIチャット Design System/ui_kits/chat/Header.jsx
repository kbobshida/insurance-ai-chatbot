// Header.jsx — corporate card header with serif title and hairline rule
const Header = () => (
  <header style={{
    background: '#fff',
    padding: '20px 24px',
    borderBottom: '1px solid #e6dfd5',
    display: 'flex',
    alignItems: 'center',
    gap: 14,
  }}>
    <div style={{ width: 36, height: 36, flexShrink: 0 }}>
      <svg viewBox="0 0 48 48" width="36" height="36">
        <path d="M24 4 L42 10 L42 24 C42 34 34 42 24 44 C14 42 6 34 6 24 L6 10 Z" fill="#0b1d33"/>
        <path d="M24 4 L42 10 L42 24 C42 34 34 42 24 44 C14 42 6 34 6 24 L6 10 Z" fill="none" stroke="#b88a3a" strokeWidth="0.8" opacity="0.6"/>
        <text x="24" y="31" textAnchor="middle" fontFamily="'Noto Serif JP', serif" fontWeight="700" fontSize="22" fill="#faf8f5">保</text>
      </svg>
    </div>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <h1 style={{
        fontFamily: "'Noto Serif JP', serif",
        fontSize: 20, fontWeight: 700, color: '#0b1d33',
        margin: 0, lineHeight: 1.2, letterSpacing: '0.02em',
      }}>保険ドキュメント AIチャット</h1>
      <div style={{
        fontSize: 11, letterSpacing: '0.12em', textTransform: 'uppercase',
        color: '#847a6b', fontWeight: 500,
      }}>Policy Assistant · ご契約のしおりに関する質問</div>
    </div>
  </header>
);

window.Header = Header;
