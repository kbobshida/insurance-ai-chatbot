document.addEventListener('DOMContentLoaded', () => {
    // 操作するHTML要素をすべて取得
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const chatForm = document.getElementById('chat-form');
    const sourcePanel = document.getElementById('source-panel');
    const exampleQuestionsContainer = document.getElementById('example-questions');
    const exampleButtons = document.querySelectorAll('.example-btn');

    // 必須要素の存在チェック
    if (!chatWindow || !userInput || !sendButton || !chatForm || !sourcePanel) {
        console.error('チャットUIの必須要素が見つかりませんでした。');
        return;
    }

    // DOMPurifyの存在チェック
    if (typeof DOMPurify === 'undefined') {
        console.error('DOMPurifyが読み込まれていません。XSS対策が無効です。');
    }

    let sessionId = null;
    let retryCount = 0;
    const MAX_RETRIES = 3;
    const RETRY_DELAY = 2000; // 2秒

    /**
     * 右側の引用元パネルを更新する関数
     */
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
            // XSS対策: textContentを使用
            fileNameDiv.textContent = source.name;

            const pageNumDiv = document.createElement('div');
            pageNumDiv.className = 'source-pagenum';
            pageNumDiv.textContent = `p. ${source.page}`;

            sourceItem.appendChild(fileNameDiv);
            sourceItem.appendChild(pageNumDiv);
            sourcePanel.appendChild(sourceItem);
        });
    };

    /**
     * アバターアイコンのSVGを取得
     */
    const getAvatarIcon = (sender) => {
        if (sender === 'ai') {
            return `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"></path><rect x="4" y="12" width="16" height="8" rx="2"></rect><path d="M2 12h20"></path><path d="M12 12V8h4"></path></svg>`;
        } else {
            return `You`;
        }
    };

    /**
     * メッセージ要素を作成して追加
     */
    const createMessageElement = (content, sender, isMarkdown = false) => {
        const messageWrapper = document.createElement('div');
        messageWrapper.className = `chat-message ${sender}-message`;
        
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.innerHTML = getAvatarIcon(sender);
        
        const messageBubble = document.createElement('div');
        messageBubble.className = 'message-bubble';
        
        if (isMarkdown && window.marked) {
            // ★★★ XSS対策: DOMPurifyでサニタイズしてから表示 ★★★
            const rawHtml = marked.parse(content);
            if (typeof DOMPurify !== 'undefined') {
                messageBubble.innerHTML = DOMPurify.sanitize(rawHtml);
            } else {
                // フォールバック: DOMPurifyがない場合はテキストのみ表示
                console.warn('DOMPurify未読み込み: HTMLをエスケープします');
                messageBubble.textContent = content;
            }
        } else {
            // XSS対策: textContentを使用
            messageBubble.textContent = content;
        }
        
        messageWrapper.appendChild(avatar);
        messageWrapper.appendChild(messageBubble);
        chatWindow.appendChild(messageWrapper);
        chatWindow.scrollTop = chatWindow.scrollHeight;
        
        return messageBubble;
    };

    /**
     * タイピングインジケーターを作成
     */
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

    /**
     * エラーメッセージを詳細化
     */
    const getDetailedErrorMessage = (error, response = null) => {
        let baseMessage = 'エラーが発生しました。';
        
        if (!navigator.onLine) {
            return baseMessage + 'インターネット接続を確認してください。';
        }
        
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            return baseMessage + 'サーバーに接続できません。サーバーが起動しているか確認してください。';
        }
        
        if (response) {
            if (response.status === 429) {
                return baseMessage + 'リクエストが多すぎます。しばらく待ってから再度お試しください。';
            } else if (response.status === 500) {
                return baseMessage + 'サーバー内部エラーです。';
            } else if (response.status === 503) {
                return baseMessage + 'サーバーが一時的に利用できません。';
            } else if (response.status >= 400 && response.status < 500) {
                return baseMessage + 'リクエストが正しくありません。';
            }
        }
        
        return baseMessage + 'しばらくしてから再度お試しください。';
    };

    /**
     * リトライ機能付きのFetch処理
     */
    const fetchWithRetry = async (url, options, currentRetry = 0) => {
        try {
            const response = await fetch(url, options);
            
            // レート制限の場合はリトライ
            if (response.status === 429 && currentRetry < MAX_RETRIES) {
                console.warn(`レート制限エラー。${RETRY_DELAY / 1000}秒後にリトライします... (${currentRetry + 1}/${MAX_RETRIES})`);
                await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
                return fetchWithRetry(url, options, currentRetry + 1);
            }
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return response;
        } catch (error) {
            // ネットワークエラーの場合もリトライ
            if (currentRetry < MAX_RETRIES && error.message.includes('fetch')) {
                console.warn(`ネットワークエラー。${RETRY_DELAY / 1000}秒後にリトライします... (${currentRetry + 1}/${MAX_RETRIES})`);
                await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
                return fetchWithRetry(url, options, currentRetry + 1);
            }
            throw error;
        }
    };

    /**
     * メッセージ送信処理
     */
    const handleSend = async (e) => {
        if (e) e.preventDefault();
        
        const query = userInput.value.trim();
        if (!query) return;

        // 質問例を非表示
        if (exampleQuestionsContainer && exampleQuestionsContainer.style.display !== 'none') {
            exampleQuestionsContainer.style.opacity = '0';
            setTimeout(() => {
                exampleQuestionsContainer.style.display = 'none';
            }, 300);
        }

        // ユーザーメッセージを表示
        createMessageElement(query, 'user');
        userInput.value = '';
        userInput.focus();

        // 送信ボタンを無効化（連続送信を防止）
        sendButton.disabled = true;
        sendButton.textContent = '送信中...';

        const typingIndicator = createTypingIndicator();

        try {
            const requestBody = { query };
            if (sessionId) {
                requestBody.session_id = sessionId;
            }

            // リトライ機能付きでリクエスト送信
            const response = await fetchWithRetry('/chat', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody),
            });
            
            const data = await response.json();
            
            // タイピングインジケーターを削除
            chatWindow.removeChild(typingIndicator);
            
            // AIの回答を表示
            createMessageElement(data.answer, 'ai', true);
            
            // 右側パネルを新しい引用元情報で更新
            updateSourcePanel(data.sources);
            
            // セッションIDを保存
            sessionId = data.session_id;
            
            // リトライカウントをリセット
            retryCount = 0;
            
        } catch (error) {
            console.error('Error:', error);
            
            // タイピングインジケーターを削除
            if (chatWindow.contains(typingIndicator)) {
                chatWindow.removeChild(typingIndicator);
            }
            
            // 詳細なエラーメッセージを表示
            const errorMessage = getDetailedErrorMessage(error);
            createMessageElement(errorMessage, 'ai');
            
        } finally {
            // 送信ボタンを再度有効化
            sendButton.disabled = false;
            sendButton.textContent = '送信';
        }
    };

    // イベントリスナーの登録
    chatForm.addEventListener('submit', handleSend);

    // 質問例ボタンのイベントリスナー
    exampleButtons.forEach(button => {
        button.addEventListener('click', () => {
            const question = button.textContent;
            userInput.value = question;
            handleSend();
        });
    });

    // Enter キーでの送信（Shift+Enter で改行）
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    // ページ読み込み時に、パネルの初期状態を設定
    updateSourcePanel();
    
    console.log('保険ドキュメント AIチャット - 初期化完了');
});