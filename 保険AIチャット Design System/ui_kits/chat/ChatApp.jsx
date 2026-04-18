// ChatApp.jsx — top-level state + mocked replies
const { useState, useRef, useEffect } = React;

const CANNED = {
  'ロードアシストでレッカー搬送は何キロまで無料ですか？': {
    answer: 'ロードアシスト特約により、事故・故障いずれの場合でも**レッカー搬送は距離無制限で無料**でご利用いただけます。ご希望の修理工場まで搬送が可能です。',
    sources: [
      { name: 'total_assist_yakkan_240101.pdf', page: 119 },
      { name: 'total_assist_pamphlet_240101.pdf', page: 8 },
    ],
  },
  'レンタカー費用補償は事故と故障で日数が違いますか？': {
    answer: 'はい、補償日数が異なります。\n- 事故の場合: 最大30日\n- 故障の場合: 最大15日\n\n詳細は約款本文をご確認ください。',
    sources: [
      { name: 'total_assist_yakkan_240101.pdf', page: 142 },
    ],
  },
  '入院時選べるアシスト特約ではどんなサービスが選べますか？': {
    answer: '入院時選べるアシスト特約では、ご契約者さまの状況に応じて以下から1つをお選びいただけます。\n1. 家事代行サービス\n2. ペット預かりサービス\n3. 育児サポートサービス',
    sources: [
      { name: 'total_assist_pamphlet_240101.pdf', page: 14 },
      { name: 'total_assist_yakkan_240101.pdf', page: 201 },
    ],
  },
};

const DEFAULT_EXAMPLES = [
  'ロードアシストでレッカー搬送は何キロまで無料ですか？',
  'レンタカー費用補償は事故と故障で日数が違いますか？',
  '入院時選べるアシスト特約ではどんなサービスが選べますか？',
];

const mockReply = (q) => CANNED[q] ?? {
  answer: 'ご質問ありがとうございます。ご契約のしおりを確認したところ、該当の記述がございました。詳細は右側の引用元をご覧ください。',
  sources: [
    { name: 'total_assist_yakkan_240101.pdf', page: 42 },
    { name: 'total_assist_pamphlet_240101.pdf', page: 3 },
  ],
};

const renderInline = (s) => {
  const parts = s.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((p, i) =>
    p.startsWith('**') && p.endsWith('**')
      ? <strong key={i} style={{ fontWeight: 700 }}>{p.slice(2, -2)}</strong>
      : <span key={i}>{p}</span>
  );
};

const MarkdownBlock = ({ text }) => {
  const lines = text.split('\n');
  const out = [];
  let buf = [];
  const flush = () => {
    if (buf.length) {
      out.push(
        <ul key={`ul-${out.length}`} style={{ margin: '6px 0', paddingLeft: 20 }}>
          {buf.map((li, i) => <li key={i}>{renderInline(li)}</li>)}
        </ul>
      );
      buf = [];
    }
  };
  lines.forEach((ln, i) => {
    const m = ln.match(/^\s*(?:[-*]|\d+\.)\s+(.*)/);
    if (m) { buf.push(m[1]); return; }
    flush();
    if (ln.trim() === '') out.push(<div key={`sp-${i}`} style={{ height: 6 }} />);
    else out.push(<div key={`p-${i}`}>{renderInline(ln)}</div>);
  });
  flush();
  return <>{out}</>;
};

const ChatApp = () => {
  const [messages, setMessages] = useState([
    { sender: 'ai', text: 'こんにちは。ご契約のしおりやパンフレットに関するご質問をどうぞ。' },
  ]);
  const [sources, setSources] = useState([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [chipsHidden, setChipsHidden] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, sending]);

  const send = async (q) => {
    const query = (q ?? input).trim();
    if (!query || sending) return;
    setChipsHidden(true);
    setMessages((m) => [...m, { sender: 'user', text: query }]);
    setInput('');
    setSending(true);

    await new Promise((r) => setTimeout(r, 900));
    const { answer, sources: srcs } = mockReply(query);
    setMessages((m) => [...m, { sender: 'ai', text: answer, markdown: true }]);
    setSources(srcs);
    setSending(false);
  };

  return (
    <div style={{
      display: 'flex', height: '100vh',
      maxWidth: 1100, margin: '0 auto', padding: 24, gap: 0,
      boxSizing: 'border-box',
    }}>
      <div style={{
        display: 'flex', flexDirection: 'row', flexGrow: 1,
        background: '#fff',
        border: '1px solid #e6dfd5',
        borderRadius: 10,
        boxShadow: '0 2px 4px rgba(11,29,51,.06), 0 4px 12px rgba(11,29,51,.05)',
        overflow: 'hidden',
      }}>
        <div style={{
          display: 'flex', flexDirection: 'column', flexGrow: 1,
          minWidth: 0,
        }}>
          <Header />
          <main ref={scrollRef} style={{
            flexGrow: 1, overflowY: 'auto',
            padding: '24px 24px 8px',
            background: '#fff',
          }}>
            {messages.map((m, i) => (
              <MessageBubble key={i} sender={m.sender}>
                {m.markdown ? <MarkdownBlock text={m.text} /> : m.text}
              </MessageBubble>
            ))}
            {sending && <TypingIndicator />}
            <ExampleChips
              items={DEFAULT_EXAMPLES}
              onPick={(q) => send(q)}
              hidden={chipsHidden}
            />
          </main>
          <Composer
            value={input}
            onChange={setInput}
            onSend={() => send()}
            sending={sending}
          />
        </div>
        <SourcePanel sources={sources} />
      </div>
    </div>
  );
};

window.ChatApp = ChatApp;
