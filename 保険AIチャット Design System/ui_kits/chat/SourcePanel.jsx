// SourcePanel.jsx — right aside with serif title, gold page-number badge
const CiteIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
       stroke="currentColor" strokeWidth="1.5"
       strokeLinecap="round" strokeLinejoin="round"
       style={{ flexShrink: 0, color: '#16375f' }}>
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/>
    <path d="M14 3v5h5"/>
  </svg>
);

const SourcePanel = ({ sources }) => (
  <aside style={{
    width: 300, flexShrink: 0,
    background: '#faf8f5',
    borderLeft: '1px solid #e6dfd5',
    padding: 24,
    overflowY: 'auto',
  }}>
    <div style={{
      fontSize: 11, letterSpacing: '0.12em', textTransform: 'uppercase',
      color: '#847a6b', fontWeight: 500, marginBottom: 6,
    }}>Citations</div>
    <h3 style={{
      fontFamily: "'Noto Serif JP', serif",
      fontSize: 20, fontWeight: 700, color: '#0b1d33',
      margin: 0, lineHeight: 1.2,
    }}>引用元</h3>
    <div style={{ width: 28, height: 1, background: '#16375f', margin: '10px 0 18px' }} />

    {(!sources || sources.length === 0) ? (
      <div style={{
        color: '#847a6b', fontSize: 13, lineHeight: 1.7,
      }}>
        回答の根拠となった文書とページ番号が、こちらに表示されます。
      </div>
    ) : (
      sources.map((s, i) => (
        <div key={i} className="source-item">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 0 }}>
            <CiteIcon />
            <div style={{
              fontSize: 13, color: '#211d17',
              whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
            }}>{s.name}</div>
          </div>
          <div style={{
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: 11, color: '#9a7026', background: '#ead8b2',
            padding: '3px 8px', borderRadius: 2, flexShrink: 0,
            letterSpacing: '0.02em',
          }}>p. {String(s.page).padStart(3, '0')}</div>
        </div>
      ))
    )}
  </aside>
);

window.SourcePanel = SourcePanel;
