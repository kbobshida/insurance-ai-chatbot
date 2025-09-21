# 保険ドキュメント特化型 AIチャットボット

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1-orange.svg)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-blue)](https://openai.com/)

指定したPDFドキュメントの内容に基づいた質疑応答ができる、RAG (Retrieval-Augmented Generation) ベースのAIチャットボットです。



## 主な機能

* **RAG (Retrieval-Augmented Generation)**: `docs`フォルダ内のPDF文書を知識源とし、その内容に基づいた回答を生成します。

* **HyDE (Hypothetical Document Embeddings)**: ユーザーの質問から理想的な回答文を一度生成し、それを使って検索を行うことで、検索精度を向上させています。

* **引用元表示**: AIの回答が、どのPDFの何ページ目に基づいているのかをサイドパネルに表示し、情報の信頼性を担保します。

* **質問分類ガードレール**: ユーザーの質問の意図を「保険関連」「挨拶・自己紹介」「無関係な話題」の3つに分類。保険と関係ない質問には回答を拒否し、チャットボットの専門性を維持します。

* **LangSmith連携**: 環境変数を設定するだけで、エージェントの思考プロセスやAPIコールをLangSmith上で詳細に追跡・デバッグできます。

## 技術スタック

* **バックエンド**: Python, FastAPI

* **LLM / AI**: LangChain, OpenAI API (gpt-4o-mini), FAISS (ベクトルストア)

* **フロントエンド**: HTML, CSS, JavaScript

* **開発環境**: Virtualenv, Uvicorn

## フォルダ構成

```

.
├── docs/               \# 知識源となるPDFファイルを格納するディレクトリ
├── faiss\_index/        \# ingest.pyによって生成されるベクトルストア
├── static/             \# フロントエンドのファイル (HTML, CSS, JS)
│   ├── index.html
│   ├── script.js
│   └── style.css
├── .env                \# APIキーなどを保存する環境変数ファイル (各自作成)
├── ingest.py           \# docs/ 内のPDFを読み込み、ベクトルストアを生成するスクリプト
├── server.py           \# FastAPIサーバーとチャットボットのメインロジック
└── README.md           \# このファイル

````

## セットアップとインストール

### 1. リポジトリのクローン

```bash
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
````

### 2\. Python仮想環境の作成と有効化

```bash
# Mac / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3\. 必要なライブラリのインストール

`requirements.txt`ファイルを作成し、以下の内容を貼り付けてからインストールを実行してください。

**requirements.txt:**

```
fastapi
uvicorn[standard]
python-dotenv
langchain
langchain-openai
langchain-community
faiss-cpu
pypdf
langsmith
```

**インストールコマンド:**

```bash
pip install -r requirements.txt
```

### 4\. 環境変数ファイル (.env) の作成

プロジェクトのルートディレクトリに`.env`という名前のファイルを作成し、ご自身のAPIキーなどを記述します。

**.env:**

```
# --- OpenAI API Key ---
OPENAI_API_KEY="sk-..."

# --- LangSmith (Optional) ---
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="ls__..."
LANGCHAIN_PROJECT="保険相談AIチャット"
```

### 5\. PDFファイルの準備

`docs`フォルダを作成し、チャットボットに読み込ませたいPDFファイルをその中に入れてください。

## 使い方

### 1\. ベクトルストアの作成

まず、`docs`フォルダ内のPDFをAIが検索できる形式に変換します。この処理は、PDFを追加・更新した場合にのみ実行が必要です。

```bash
python ingest.py
```

実行すると、`faiss_index`というフォルダが生成されます。

### 2\. チャットサーバーの起動

以下のコマンドでWebサーバーを起動します。

```bash
uvicorn server:app --reload --port 8000
```

### 3\. ブラウザでアクセス

サーバーが起動したら、Webブラウザで `http://1.27.0.0.1:8000` にアクセスしてください。チャット画面が表示されます。

## 参考

本プロジェクトを作成するにあたり、以下の記事を参考にさせていただきました。

  - [FastAPI + LangChain + RAGでPDFドキュメントへのQ\&Aボットを作ってみる - Qiita](https://qiita.com/yukinaka_data/items/8270e1a559e8fc3c047d)

