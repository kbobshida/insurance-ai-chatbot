// Composer.jsx — minimal input + solid navy send button
const Composer = ({ value, onChange, onSend, sending }) => {
  const submit = (e) => { e.preventDefault(); if (!sending) onSend(); };
  return (
    <footer style={{
      background: '#fff',
      borderTop: '1px solid #e6dfd5',
      padding: '16px 20px',
    }}>
      <form onSubmit={submit} style={{
        display: 'flex', alignItems: 'stretch', gap: 10,
      }}>
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="質問を入力してください…"
          className="composer-input"
          disabled={sending}
        />
        <button
          type="submit"
          disabled={sending || !value.trim()}
          className="composer-send"
        >
          {sending ? '送信中…' : '送信'}
        </button>
      </form>
    </footer>
  );
};

window.Composer = Composer;
