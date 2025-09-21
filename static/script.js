document.addEventListener('DOMContentLoaded', () => {


    // 操作するHTML要素をすべて取得
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const chatForm = document.getElementById('chat-form');
    const sourcePanel = document.getElementById('source-panel'); // 右側パネルを追加
    const exampleQuestionsContainer = document.getElementById('example-questions');
    const exampleButtons = document.querySelectorAll('.example-btn');

    if (!chatWindow || !userInput || !sendButton || !chatForm || !sourcePanel) {
        console.error('チャットUIの必須要素が見つかりませんでした。');
        return;
    }

    let sessionId = null;


    // 右側の引用元パネルを更新する関数
    const updateSourcePanel = (sources = []) => {
        // パネルの中身を一度空にする
        sourcePanel.innerHTML = '';

        if (sources.length === 0) {
            // 引用元がない場合は、パネルにプレースホルダーを表示
            const placeholder = document.createElement('div');
            placeholder.className = 'panel-placeholder';
            placeholder.textContent = '引用元はここに表示されます';
            sourcePanel.appendChild(placeholder);
            return;
        }

        // パネルのヘッダーを追加
        const header = document.createElement('h3');
        header.className = 'panel-header';
        header.textContent = '引用元';
        sourcePanel.appendChild(header);
        
        // 各引用元アイテムを生成して追加
        sources.forEach(source => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            
            const fileNameDiv = document.createElement('div');
            fileNameDiv.className = 'source-filename';
            fileNameDiv.textContent = source.name;

            const pageNumDiv = document.createElement('div');
            pageNumDiv.className = 'source-pagenum';
            pageNumDiv.textContent = `p. ${source.page}`;

            sourceItem.appendChild(fileNameDiv);
            sourceItem.appendChild(pageNumDiv);
            sourcePanel.appendChild(sourceItem);
        });
    };

    const getAvatarIcon = (sender) => {
        if (sender === 'ai') {
            return `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"></path><rect x="4" y="12" width="16" height="8" rx="2"></rect><path d="M2 12h20"></path><path d="M12 12V8h4"></path></svg>`;
        } else {
            return `You`;
        }
    };

    const createMessageElement = (content, sender, isMarkdown = false) => {
        const messageWrapper = document.createElement('div');
        messageWrapper.className = `chat-message ${sender}-message`;
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.innerHTML = getAvatarIcon(sender);
        const messageBubble = document.createElement('div');
        messageBubble.className = 'message-bubble';
        if (isMarkdown && window.marked) {
            messageBubble.innerHTML = marked.parse(content);
        } else {
            messageBubble.textContent = content;
        }
        messageWrapper.appendChild(avatar);
        messageWrapper.appendChild(messageBubble);
        chatWindow.appendChild(messageWrapper);
        chatWindow.scrollTop = chatWindow.scrollHeight;
        return messageBubble;
    };

    const createTypingIndicator = () => {
        const messageWrapper = document.createElement('div');
        messageWrapper.className = 'chat-message ai-message';
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.innerHTML = getAvatarIcon('ai');
        const messageBubble = document.createElement('div');
        messageBubble.className = 'message-bubble';
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        typingIndicator.innerHTML = '<span></span><span></span><span></span>';
        messageBubble.appendChild(typingIndicator);
        messageWrapper.appendChild(avatar);
        messageWrapper.appendChild(messageBubble);
        chatWindow.appendChild(messageWrapper);
        chatWindow.scrollTop = chatWindow.scrollHeight;
        return messageWrapper;
    };

    const handleSend = async (e) => {
        if (e) e.preventDefault();
        const query = userInput.value.trim();
        if (!query) return;

        if (exampleQuestionsContainer && exampleQuestionsContainer.style.display !== 'none') {
            exampleQuestionsContainer.style.opacity = '0';
            setTimeout(() => {
                exampleQuestionsContainer.style.display = 'none';
            }, 300);
        }

        createMessageElement(query, 'user');
        userInput.value = '';
        userInput.focus();

        const typingIndicator = createTypingIndicator();

        try {
            const requestBody = { query };
            if (sessionId) requestBody.session_id = sessionId;
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody),
            });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            const data = await response.json();
            
            chatWindow.removeChild(typingIndicator);
            createMessageElement(data.answer, 'ai', true);
            
    
            // 右側パネルを新しい引用元情報で更新
            updateSourcePanel(data.sources);
            
            sessionId = data.session_id;
        } catch (error) {
            console.error('Error:', error);
            chatWindow.removeChild(typingIndicator);
            createMessageElement('エラーが発生しました。サーバーとの通信を確認してください。', 'ai');
        }
    };

    chatForm.addEventListener('submit', handleSend);

    exampleButtons.forEach(button => {
        button.addEventListener('click', () => {
            const question = button.textContent;
            userInput.value = question;
            handleSend();
        });
    });

    
    // ページ読み込み時に、パネルの初期状態を設定
    updateSourcePanel();
});