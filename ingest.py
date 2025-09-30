import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 設定ファイルのインポート
from config import settings

# .envファイルから環境変数を読み込む
load_dotenv()

def create_vector_store():
    """
    docsフォルダ内のPDFを読み込み、ベクトルストアを作成して保存する
    """
    print("=" * 60)
    print("ベクトルストア作成プログラム")
    print("=" * 60)
    
    print(f"\n[1/4] PDFの読み込みを開始: {settings.docs_path}")
    
    # docsフォルダ内のPDFをすべて読み込む
    if not os.path.exists(settings.docs_path):
        print(f"エラー: '{settings.docs_path}' フォルダが見つかりません。")
        print(f"'{settings.docs_path}' フォルダを作成し、PDFファイルを配置してください。")
        return
    
    loader = PyPDFDirectoryLoader(settings.docs_path)
    documents = loader.load()
    
    if not documents:
        print(f"エラー: '{settings.docs_path}' フォルダ内にPDFファイルが見つかりません。")
        print("PDFファイルを配置してから再度実行してください。")
        return
    
    print(f"✓ {len(documents)}個のドキュメントを読み込みました")
    
    print(f"\n[2/4] テキストの分割を開始")
    print(f"    - chunk_size: {settings.chunk_size}")
    print(f"    - chunk_overlap: {settings.chunk_overlap}")
    
    # テキストを適切なサイズのチャンクに分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    print(f"✓ {len(texts)}個のチャンクに分割しました")
    
    print(f"\n[3/4] テキストのベクトル化とインデックス作成を開始")
    print(f"    - embedding_chunk_size: {settings.embedding_chunk_size}")
    
    # OpenAIの埋め込みモデルを初期化
    # chunk_sizeを指定して、APIリクエストを分割するように設定
    embeddings = OpenAIEmbeddings(chunk_size=settings.embedding_chunk_size)
    
    # FAISSベクトルストアを作成
    db = FAISS.from_documents(texts, embeddings)
    
    print(f"\n[4/4] インデックスの保存: {settings.index_path}")
    
    # ローカルに保存
    db.save_local(settings.index_path)
    print(f"✓ インデックスを '{settings.index_path}' に保存しました")
    
    print("\n" + "=" * 60)
    print("ベクトルストアの作成が完了しました！")
    print("次のコマンドでサーバーを起動してください:")
    print("  uvicorn server:app --reload --port 8000")
    print("=" * 60)

if __name__ == "__main__":
    try:
        create_vector_store()
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        print("環境変数とPDFファイルの配置を確認してください。")
        import traceback
        traceback.print_exc()