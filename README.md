# 保険ドキュメント特化型 AI チャットボット

指定した PDF を知識源として参照しながら、保険に関する質問へ正確かつ根拠付きで回答する RAG（Retrieval-Augmented Generation）アプリケーションです。LangChain と OpenAI API を組み合わせ、ドキュメント検索から回答生成までを一貫して自動化しています。

## 特徴

- **RAG 構成**: `docs/` 配下の PDF を分割・埋め込みし、FAISS ベクトルストアから関連チャンクを取得して回答を生成します。
- **HyDE 検索**: ユーザー質問から仮想回答を作り、その文章をクエリとして検索することでリトリーバの精度を高めています。
- **引用表示**: 回答に利用した文書名とページ番号を UI のサイドパネルに表示し、回答根拠を可視化します。
- **質問分類ガードレール**: 質問を「保険に関する内容 / 挨拶などのメタ / 無関係トピック」に分類し、対応方針を自動で切り替えます。
- **セキュリティ強化**: XSS 対策（DOMPurify）、レート制限、セッション管理など本番環境を考慮した実装。
- **LangSmith 対応**: トレース用の環境変数を設定すれば、LLM の推論過程を LangSmith から観測できます。

## システム構成

```
.
├── docs/               # 知識源にする PDF を保存
├── faiss_index/        # ingest.py で生成されるベクトルストア
├── static/             # フロントエンド (HTML/CSS/JS)
│   ├── index.html
│   ├── script.js
│   └── style.css
├── config.py           # アプリケーション設定（新規）
├── ingest.py           # PDF 取り込みとインデックス作成
├── server.py           # FastAPI + LangChain による API
├── requirements.txt    # 依存パッケージ一覧
├── .env                # 環境変数設定
├── app.log             # アプリケーションログ
└── README.md
```

バックエンドは FastAPI、ベクトルストアには FAISS、LLM には `gpt-4o-mini` を利用しています。フロントエンドは素の HTML/CSS/JavaScript で構成され、引用パネルやサンプル質問など最小限の UI を提供します。

## セットアップ

### 1. リポジトリの取得

```bash
git clone https://github.com/your-username/insurance-ai-chatbot.git
cd insurance-ai-chatbot
```

### 2. 仮想環境の作成と有効化

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate
```

### 3. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

**インストールされる主要パッケージ:**
- FastAPI & Uvicorn（Web サーバー）
- LangChain（RAG フレームワーク）
- FAISS（ベクトルストア）
- OpenAI API（LLM・埋め込み）
- pydantic-settings（設定管理）
- slowapi（レート制限）

### 4. 環境変数の設定

ルートディレクトリに `.env` ファイルを作成し、以下を記入します。

```env
# 必須: OpenAI API キー
OPENAI_API_KEY="sk-..."
# OpenMP の警告を回避
KMP_DUPLICATE_LIB_OK=TRUE

# オプション: LangSmith を使う場合のみ
# LANGCHAIN_TRACING_V2="true"
# LANGCHAIN_API_KEY="ls__..."
# LANGCHAIN_PROJECT="保険相談AIチャット"

# オプション: デフォルト値があるため未設定でも動作します
# MAX_SESSIONS=1000
# SESSION_TIMEOUT_HOURS=24
# RATE_LIMIT_PER_MINUTE=10
```

**最低限必要な設定:**
```env
OPENAI_API_KEY="sk-..."
```

その他の設定は `config.py` のデフォルト値が使用されます。

### 5. PDF の配置

`docs/` ディレクトリに回答根拠として使用したい PDF をコピーします。

```bash
mkdir -p docs
# PDFファイルを docs/ にコピー
```

**サンプル PDF（参考）:**
- [トータルアシスト自動車保険 約款](https://www.tokiomarine-nichido.co.jp/service/pdf/total_assist_yakkan_240101.pdf)
- [トータルアシスト自動車保険 パンフレット](https://www.tokiomarine-nichido.co.jp/service/pdf/total_assist_pamphlet_240101.pdf)

## 使い方

### 1. ベクトルストアの生成

```bash
python ingest.py
```

実行後に `faiss_index/` が作成されます。PDF を更新した際も同じコマンドで再生成してください。

**出力例:**
```
==========================================================
ベクトルストア作成プログラム
==========================================================

[1/4] PDFの読み込みを開始: docs
✓ 2個のドキュメントを読み込みました

[2/4] テキストの分割を開始
    - chunk_size: 1000
    - chunk_overlap: 50
✓ 245個のチャンクに分割しました

[3/4] テキストのベクトル化とインデックス作成を開始
    - embedding_chunk_size: 200

[4/4] インデックスの保存: faiss_index
✓ インデックスを 'faiss_index' に保存しました

==========================================================
ベクトルストアの作成が完了しました！
次のコマンドでサーバーを起動してください:
  uvicorn server:app --reload --port 8000
==========================================================
```

### 2. サーバーの起動

```bash
uvicorn server:app --reload --port 8000
```

**起動ログ例:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### 3. ブラウザからアクセス

`http://127.0.0.1:8000` を開き、チャット画面で質問します。引用パネルに回答根拠が表示されます。

## 機能詳細

### 質問分類ガードレール

ユーザーの質問を自動的に3つのカテゴリに分類します：

1. **insurance（保険関連）**: PDF 検索ツールを使用して回答
2. **meta（挨拶・雑談）**: シンプルな対話で応答
3. **off_topic（無関係）**: 丁重に断る

### HyDE 検索

従来の検索では質問文をそのまま使用しますが、HyDE（Hypothetical Document Embeddings）では：

1. 質問から「理想的な回答」を LLM に生成させる
2. その仮想回答を使ってベクトル検索
3. より関連性の高い文書を取得

これにより検索精度が向上します。

### セキュリティ機能

- **XSS 対策**: DOMPurify によるサニタイゼーション
- **レート制限**: 1分間に10リクエストまで（デフォルト）
- **セッション管理**: メモリリーク防止、タイムアウト処理
- **安全なデシリアライゼーション**: FAISS の安全な読み込み

### ログ機能

すべてのリクエスト・エラーは `app.log` に記録されます。

```bash
# ログのリアルタイム監視
tail -f app.log
```

## 運用のヒント

### HyDE の挙動調整

`server.py` の `search_insurance_documents` ツール内で調整可能：

```python
# 検索件数の変更
retriever = db.as_retriever(search_kwargs={"k": 5})  # デフォルトは5件
```

### 設定のカスタマイズ

`config.py` または `.env` で調整可能な主要設定：

| 設定項目 | デフォルト値 | 説明 |
|---------|------------|------|
| `CHUNK_SIZE` | 1000 | テキスト分割のチャンクサイズ |
| `CHUNK_OVERLAP` | 50 | チャンク間のオーバーラップ |
| `RETRIEVAL_K` | 5 | 検索で取得する文書数 |
| `MAX_SESSIONS` | 1000 | 保持するセッション数の上限
