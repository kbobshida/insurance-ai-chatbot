// Avatar.jsx — serif monogram avatars (AI navy, user stone)
const Avatar = ({ kind }) => (
  <div style={{
    width: 34, height: 34, borderRadius: '50%',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    flexShrink: 0,
    fontFamily: "'Noto Serif JP', serif",
    fontWeight: 700, fontSize: 13,
    background: kind === 'ai' ? '#0b1d33' : '#f3efe9',
    color: kind === 'ai' ? '#faf8f5' : '#211d17',
    border: kind === 'ai' ? 'none' : '1px solid #cec3b3',
  }}>
    {kind === 'ai' ? 'AI' : '私'}
  </div>
);

window.Avatar = Avatar;
