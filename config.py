from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """アプリケーション設定"""
    
    # OpenAI API設定
    openai_api_key: str
    
    # LangSmith設定（オプション）
    langchain_tracing_v2: Optional[str] = None
    langchain_api_key: Optional[str] = None
    langchain_project: Optional[str] = "保険相談AIチャット"
    
    # ドキュメント処理設定
    chunk_size: int = 1000
    chunk_overlap: int = 50
    embedding_chunk_size: int = 200
    
    # 検索設定
    retrieval_k: int = 5
    
    # LLM設定
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    
    # パス設定
    docs_path: str = "docs"
    index_path: str = "faiss_index"
    
    # セッション管理設定
    max_sessions: int = 1000
    session_timeout_hours: int = 24
    
    # レート制限設定
    rate_limit_per_minute: int = 10
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # 余分な環境変数を無視（KMP_DUPLICATE_LIB_OK など）

# グローバル設定インスタンス
settings = Settings()