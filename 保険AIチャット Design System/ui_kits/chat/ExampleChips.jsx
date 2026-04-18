// ExampleChips.jsx — starter questions as rectilinear cards
const ExampleChips = ({ items, onPick, hidden }) => {
  if (hidden) return null;
  return (
    <div style={{
      padding: '8px 4px 12px',
      transition: 'opacity 0.24s ease-out',
    }}>
      <div style={{
        fontSize: 11, letterSpacing: '0.12em', textTransform: 'uppercase',
        color: '#847a6b', fontWeight: 500, textAlign: 'center',
        marginBottom: 12,
      }}>質問例</div>
      <div style={{
        display: 'flex', flexWrap: 'wrap',
        justifyContent: 'center', gap: 8,
      }}>
        {items.map((q, i) => (
          <button key={i} className="example-btn" onClick={() => onPick(q)}>
            {q}
          </button>
        ))}
      </div>
    </div>
  );
};

window.ExampleChips = ExampleChips;
