// MessageBubble.jsx — rectilinear bubbles; AI has navy top-hairline, user is filled navy
const MessageBubble = ({ sender, children }) => {
  const isUser = sender === 'user';
  return (
    <div style={{
      display: 'flex',
      alignItems: 'flex-start',
      marginBottom: 20,
      gap: 12,
      maxWidth: '90%',
      marginLeft: isUser ? 'auto' : 0,
      marginRight: isUser ? 0 : 'auto',
      flexDirection: isUser ? 'row-reverse' : 'row',
      animation: 'fadeInUp 0.24s cubic-bezier(0.4,0,0.2,1) forwards',
      opacity: 0,
      transform: 'translateY(8px)',
    }}>
      <Avatar kind={isUser ? 'user' : 'ai'} />
      <div style={{
        padding: '12px 16px',
        borderRadius: 4,
        lineHeight: 1.7,
        fontSize: 15,
        background: isUser ? '#16375f' : '#fff',
        color: isUser ? '#faf8f5' : '#211d17',
        border: isUser ? 'none' : '1px solid #e6dfd5',
        borderTop: isUser ? 'none' : '1px solid #16375f',
      }}>
        {children}
      </div>
    </div>
  );
};

window.MessageBubble = MessageBubble;
