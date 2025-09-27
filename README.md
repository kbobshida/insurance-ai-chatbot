# 保険ドキュメント特化型 AI チャットボット

指定した PDF を知識源として参照しながら、保険に関する質問へ正確かつ根拠付きで回答する RAG（Retrieval-Augmented Generation）アプリケーションです。LangChain と OpenAI API を組み合わせ、ドキュメント検索から回答生成までを一貫して自動化しています。

## 特徴
- **RAG 構成**: `docs/` 配下の PDF を分割・埋め込みし、FAISS ベクトルストアから関連チャンクを取得して回答を生成します。
- **HyDE 検索**: ユーザー質問から仮想回答を作り、その文章をクエリとして検索することでリトリーバの精度を高めています。
- **引用表示**: 回答に利用した文書名とページ番号を UI のサイドパネルに表示し、回答根拠を可視化します。
- **質問分類ガードレール**: 質問を「保険に関する内容 / 挨拶などのメタ / 無関係トピック」に分類し、対応方針を自動で切り替えます。
- **LangSmith 対応**: トレース用の環境変数を設定すれば、LLM の推論過程を LangSmith から観測できます。

## システム構成
```
.
├── docs/               # 知識源にする PDF を保存
├── faiss_index/        # ingest.py で生成されるベクトルストア
├── static/             # フロントエンド (HTML/CSS/JS)
├── ingest.py           # PDF 取り込みとインデックス作成
├── server.py           # FastAPI + LangChain による API
├── requirements.txt    # 依存パッケージ一覧
└── README.md
```

バックエンドは FastAPI、ベクトルストアには FAISS、LLM には `gpt-4o-mini` を利用しています。フロントエンドは素の HTML/CSS/JavaScript で構成され、引用パネルやサンプル質問など最小限の UI を提供します。

## セットアップ
1. **リポジトリの取得**
   ```bash
   git clone https://github.com/your-username/insurance-ai-chatbot.git
   cd insurance-ai-chatbot
   ```

2. **仮想環境の作成と有効化**
   ```bash
   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate

   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate
   ```

3. **依存ライブラリのインストール**
   ```bash
   pip install -r requirements.txt
   ```

4. **環境変数の設定** – ルートディレクトリに `.env` を作成し、以下を記入します。
   ```env
   OPENAI_API_KEY="sk-..."
   # 任意: LangSmith を使う場合のみ
   LANGCHAIN_TRACING_V2="true"
   LANGCHAIN_API_KEY="ls__..."
   LANGCHAIN_PROJECT="保険相談AIチャット"
   ```

5. **PDF の配置** – `docs/` ディレクトリに回答根拠として使用したい PDF をコピーします。

## 使い方
1. **ベクトルストアの生成**
   ```bash
   python ingest.py
   ```
   実行後に `faiss_index/` が作成されます。PDF を更新した際も同じコマンドで再生成してください。

2. **サーバーの起動**
   ```bash
   uvicorn server:app --reload --port 8000
   ```

3. **ブラウザからアクセス** – `http://127.0.0.1:8000` を開き、チャット画面で質問します。引用パネルに回答根拠が表示されます。

## 運用のヒント
- **HyDE の挙動**: `server.py` の `search_insurance_documents` ツール内で仮想回答を生成し、検索クエリに利用しています。挙動を調整したい場合はプロンプトや取得件数 (`k`) を変更してください。
- **LangSmith 連携**: `.env` にトレース用キーを設定すると、質問分類チェインや HyDE ツールの呼び出し履歴を LangSmith 上で確認できます。
- **インデックスの再生成**: PDF を追加・差し替えたあとで `faiss_index/` を削除し、再度 `python ingest.py` を実行することで最新内容に同期できます。

## トラブルシューティング
- **`faiss_index` が見つからない**: 先に `python ingest.py` を実行し、インデックスを生成してください。
- **OpenAI API のエラー**: API キーが `.env` に設定されているか、リクエスト上限に達していないか確認します。
- **PDF が読み込まれない**: `docs/` 直下に PDF があるか、ファイルのパーミッションに問題がないか確認してください。

## 参考資料
- [FastAPI 公式ドキュメント](https://fastapi.tiangolo.com/)
- [LangChain](https://python.langchain.com/)
- [Tokio Marine: トータルアシスト自動車保険 約款 (PDF)](https://www.tokiomarine-nichido.co.jp/service/pdf/total_assist_yakkan_240101.pdf)
- [Tokio Marine: トータルアシスト自動車保険 パンフレット (PDF)](https://www.tokiomarine-nichido.co.jp/service/pdf/total_assist_pamphlet_240101.pdf)
