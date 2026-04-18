/* =============================================================
   保険ドキュメント AIチャット — Frontend Logic
   - Corporate design system applied
   - Falls back to canned mock responses if /chat is unreachable
     (so the UI is demonstrable without the backend running)
   ============================================================= */

document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const chatForm = document.getElementById('chat-form');
    const sourceList = document.getElementById('source-list');
    const exampleQuestions = document.getElementById('example-questions');
    const exampleButtons = document.querySelectorAll('.example-btn');

    if (!chatWindow || !userInput || !sendButton || !chatForm || !sourceList) {
        console.error('チャットUIの必須要素が見つかりませんでした。');
        return;
    }
    if (typeof DOMPurify === 'undefined') {
        console.warn('DOMPurifyが読み込まれていません。Markdownは平文表示にフォールバックします。');
    }

    let sessionId = null;
    const MAX_RETRIES = 2;
    const RETRY_DELAY = 1500;

    /* ---------------------------------------------------------
     *  Canned (mock) responses — used when /chat is unreachable
     * --------------------------------------------------------- */
    const CANNED = {
        'ロードアシストでレッカー搬送は何キロまで無料ですか？': {
            answer: 'ロードアシスト特約により、事故・故障いずれの場合でも**レッカー搬送は距離無制限で無料**でご利用いただけます。ご希望の修理工場まで搬送が可能です。',
            sources: [
                { name: 'total_assist_yakkan_240101.pdf', page: 119 },
                { name: 'total_assist_pamphlet_240101.pdf', page: 8 },
            ],
        },
        'レンタカー費用補償は事故と故障で日数が違いますか？': {
            answer: 'はい、補償日数が異なります。\n\n- 事故の場合: 最大30日\n- 故障の場合: 最大15日\n\n詳細は約款本文をご確認ください。',
            sources: [
                { name: 'total_assist_yakkan_240101.pdf', page: 142 },
            ],
        },
        '入院時選べるアシスト特約ではどんなサービスが選べますか？': {
            answer: '入院時選べるアシスト特約では、ご契約者さまの状況に応じて以下から1つをお選びいただけます。\n\n1. 家事代行サービス\n2. ペット預かりサービス\n3. 育児サポートサービス',
            sources: [
                { name: 'total_assist_pamphlet_240101.pdf', page: 14 },
                { name: 'total_assist_yakkan_240101.pdf', page: 201 },
            ],
        },
    };

    const FALLBACK_REPLY = {
        answer: 'ご質問ありがとうございます。ご契約のしおりを確認したところ、該当の記述がございました。詳細は右側の引用元をご覧ください。',
        sources: [
            { name: 'total_assist_yakkan_240101.pdf', page: 42 },
            { name: 'total_assist_pamphlet_240101.pdf', page: 3 },
        ],
    };

    const mockReply = (query) =>
        CANNED[query] ?? FALLBACK_REPLY;

    /* ---------------------------------------------------------
     *  Source panel rendering
     * --------------------------------------------------------- */
    const citeIconSvg = () => `
        <svg class="source-icon" width="16" height="16" viewBox="0 0 24 24"
             fill="none" stroke="currentColor" stroke-width="1.5"
             stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/>
            <path d="M14 3v5h5"/>
        </svg>`;

    const updateSourcePanel = (sources = []) => {
        sourceList.innerHTML = '';
        if (!sources || sources.length === 0) {
            const p = document.createElement('p');
            p.className = 'panel-placeholder';
            p.textContent = '回答の根拠となった文書とページ番号が、こちらに表示されます。';
            sourceList.appendChild(p);
            return;
        }

        sources.forEach((source) => {
            const item = document.createElement('div');
            item.className = 'source-item';

            const label = document.createElement('div');
            label.className = 'source-label';
            label.insertAdjacentHTML('beforeend', citeIconSvg());

            const filename = document.createElement('div');
            filename.className = 'source-filename';
            filename.textContent = source.name; // XSS-safe via textContent
            label.appendChild(filename);

            const badge = document.createElement('div');
            badge.className = 'source-pagenum';
            const paddedPage = String(source.page).padStart(3, '0');
            badge.textContent = `p. ${paddedPage}`;

            item.appendChild(label);
            item.appendChild(badge);
            sourceList.appendChild(item);
        });
    };

    /* ---------------------------------------------------------
     *  Message rendering
     * --------------------------------------------------------- */
    const createAvatar = (sender) => {
        const avatar = document.createElement('div');
        avatar.className = `avatar ${sender}-avatar`;
        avatar.setAttribute('aria-hidden', 'true');
        avatar.textContent = sender === 'ai' ? 'AI' : '私';
        return avatar;
    };

    const createMessageElement = (content, sender, { markdown = false } = {}) => {
        const wrapper = document.createElement('div');
        wrapper.className = `chat-message ${sender}-message`;

        const avatar = createAvatar(sender);
        const bubble = document.createElement('div');
        bubble.className = `message-bubble ${sender}-bubble`;

        if (markdown && window.marked) {
            const rawHtml = marked.parse(content);
            if (typeof DOMPurify !== 'undefined') {
                bubble.innerHTML = DOMPurify.sanitize(rawHtml);
            } else {
                bubble.textContent = content;
            }
        } else {
            bubble.textContent = content;
        }

        wrapper.appendChild(avatar);
        wrapper.appendChild(bubble);
        chatWindow.appendChild(wrapper);
        chatWindow.scrollTop = chatWindow.scrollHeight;

        return bubble;
    };

    const createTypingIndicator = () => {
        const wrapper = document.createElement('div');
        wrapper.className = 'chat-message ai-message';

        const avatar = createAvatar('ai');
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble ai-bubble';

        const dots = document.createElement('div');
        dots.className = 'typing-indicator';
        dots.innerHTML = '<span></span><span></span><span></span>';

        bubble.appendChild(dots);
        wrapper.appendChild(avatar);
        wrapper.appendChild(bubble);
        chatWindow.appendChild(wrapper);
        chatWindow.scrollTop = chatWindow.scrollHeight;

        return wrapper;
    };

    /* ---------------------------------------------------------
     *  Example chips hiding
     * --------------------------------------------------------- */
    const hideExampleQuestions = () => {
        if (!exampleQuestions || exampleQuestions.style.display === 'none') return;
        exampleQuestions.style.opacity = '0';
        setTimeout(() => { exampleQuestions.style.display = 'none'; }, 240);
    };

    /* ---------------------------------------------------------
     *  Fetch with retry (to real backend)
     * --------------------------------------------------------- */
    const fetchWithRetry = async (url, options, attempt = 0) => {
        try {
            const res = await fetch(url, options);
            if (res.status === 429 && attempt < MAX_RETRIES) {
                await new Promise((r) => setTimeout(r, RETRY_DELAY));
                return fetchWithRetry(url, options, attempt + 1);
            }
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return res;
        } catch (err) {
            if (attempt < MAX_RETRIES && /fetch|network/i.test(err.message)) {
                await new Promise((r) => setTimeout(r, RETRY_DELAY));
                return fetchWithRetry(url, options, attempt + 1);
            }
            throw err;
        }
    };

    /* ---------------------------------------------------------
     *  Send handler — tries real /chat, falls back to mock
     * --------------------------------------------------------- */
    const handleSend = async (e) => {
        if (e) e.preventDefault();
        const query = userInput.value.trim();
        if (!query) return;

        hideExampleQuestions();
        createMessageElement(query, 'user');
        userInput.value = '';
        userInput.focus();

        sendButton.disabled = true;
        sendButton.textContent = '送信中…';

        const typingEl = createTypingIndicator();

        try {
            const body = { query };
            if (sessionId) body.session_id = sessionId;

            const res = await fetchWithRetry('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const data = await res.json();

            if (chatWindow.contains(typingEl)) chatWindow.removeChild(typingEl);

            createMessageElement(data.answer, 'ai', { markdown: true });
            updateSourcePanel(data.sources);
            sessionId = data.session_id ?? sessionId;

        } catch (err) {
            // Backend unavailable → mock fallback, so the UI remains usable for demo.
            console.warn('[/chat 未接続のためモック応答に切り替えます]', err);

            // Simulated latency to keep the typing indicator visible for a beat.
            await new Promise((r) => setTimeout(r, 900));

            if (chatWindow.contains(typingEl)) chatWindow.removeChild(typingEl);

            const mock = mockReply(query);
            createMessageElement(mock.answer, 'ai', { markdown: true });
            updateSourcePanel(mock.sources);

        } finally {
            sendButton.disabled = false;
            sendButton.textContent = '送信';
        }
    };

    /* ---------------------------------------------------------
     *  Event wiring
     * --------------------------------------------------------- */
    chatForm.addEventListener('submit', handleSend);

    exampleButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
            userInput.value = btn.textContent.trim();
            handleSend();
        });
    });

    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    // Initial state
    updateSourcePanel();
    console.log('保険ドキュメント AIチャット — 初期化完了（Design System 適用）');
});
