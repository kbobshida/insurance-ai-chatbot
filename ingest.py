import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# .envファイルから環境変数を読み込む
load_dotenv()

# 定数
DOCS_PATH = "docs"
INDEX_PATH = "faiss_index"

def create_vector_store():
    """
    docsフォルダ内のPDFを読み込み、ベクトルストアを作成して保存する
    """
    print("--- PDFの読み込みを開始 ---")
    # docsフォルダ内のPDFをすべて読み込む
    loader = PyPDFDirectoryLoader(DOCS_PATH)
    documents = loader.load()
    if not documents:
        print("PDFファイルが見つかりません。docsフォルダにPDFファイルを入れてください。")
        return

    print(f"--- {len(documents)}個のドキュメントを読み込みました ---")

    print("--- テキストの分割を開始 ---")
    # テキストを適切なサイズのチャンクに分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"--- {len(texts)}個のチャンクに分割しました ---")

    print("--- テキストのベクトル化とインデックス作成を開始 ---")
    # OpenAIの埋め込みモデルを初期化
    # ★★★ 変更点: chunk_sizeを指定して、APIリクエストを分割するように設定 ★★★
    embeddings = OpenAIEmbeddings(chunk_size=200)

    # FAISSベクトルストアを作成
    db = FAISS.from_documents(texts, embeddings)

    # ローカルに保存
    db.save_local(INDEX_PATH)
    print(f"--- インデックスを '{INDEX_PATH}' に保存しました ---")

if __name__ == "__main__":
    create_vector_store()