# 必要なライブラリやモジュールをインポート
import os
import json
import logging
from uuid import uuid4
from datetime import datetime, timedelta
from collections import OrderedDict
from typing import List, Dict, Literal
import asyncio

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# LangChainの主要コンポーネントをインポート
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.exceptions import OutputParserException
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.agents import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

# 設定ファイルのインポート
from config import settings

# .envファイルから環境変数を読み込む
load_dotenv()

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPIアプリケーションを初期化
app = FastAPI()

# '/static'パスで'static'フォルダ内のファイル(CSS, JS)を提供
app.mount("/static", StaticFiles(directory="static"), name="static")

# ルートURL("/")へのアクセス時に'index.html'を返す
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

# CORS (クロスオリジンリソース共有) の設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- セッション管理クラス: メモリリークと並行処理に対応 ---
class SessionManager:
    """スレッドセーフなセッション管理クラス"""
    
    def __init__(self, max_sessions: int = 1000, timeout_hours: int = 24):
        self.sessions = OrderedDict()
        self.last_access = {}
        self.max_sessions = max_sessions
        self.timeout = timedelta(hours=timeout_hours)
        self.lock = asyncio.Lock()
        logger.info(f"SessionManager初期化: max_sessions={max_sessions}, timeout={timeout_hours}h")
    
    async def get(self, session_id: str) -> List:
        """セッションの取得"""
        async with self.lock:
            await self._cleanup_old_sessions()
            self.last_access[session_id] = datetime.now()
            return self.sessions.get(session_id, [])
    
    async def set(self, session_id: str, history: List):
        """セッションの保存"""
        async with self.lock:
            # セッション数の上限チェック
            if session_id not in self.sessions and len(self.sessions) >= self.max_sessions:
                oldest_sid = next(iter(self.sessions))
                self.sessions.pop(oldest_sid)
                self.last_access.pop(oldest_sid, None)
                logger.info(f"セッション上限到達。最古のセッション削除: {oldest_sid}")
            
            self.sessions[session_id] = history
            self.last_access[session_id] = datetime.now()
            # OrderedDictの順序を更新（最新アクセスを末尾に）
            self.sessions.move_to_end(session_id)
    
    async def _cleanup_old_sessions(self):
        """タイムアウトしたセッションをクリーンアップ"""
        now = datetime.now()
        expired = [
            sid for sid, last in self.last_access.items()
            if now - last > self.timeout
        ]
        for sid in expired:
            self.sessions.pop(sid, None)
            self.last_access.pop(sid, None)
            logger.info(f"タイムアウトによりセッション削除: {sid}")

# --- グローバル変数 ---
llm = None
non_streaming_llm = None
embeddings = None
db = None
agent_executor = None
classification_chain = None
session_manager = None

# --- Pydanticモデル定義 ---

class SourceAPI(BaseModel):
    """引用元情報の型"""
    name: str
    page: int

class ChatRequest(BaseModel):
    """フロントエンドからのリクエストの型"""
    query: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    """フロントエンドへのレスポンスの型"""
    answer: str
    sources: List[SourceAPI]
    session_id: str

class AgentOutput(BaseModel):
    """LangChainエージェントの出力形式"""
    answer: str = Field(description="ユーザーへの最終的な回答。会話形式で記述する。")
    sources: List[SourceAPI] = Field(description="回答の生成に使用した引用元のリスト。引用元がない場合は空のリストにする。")

class QueryClassifier(BaseModel):
    """質問を3カテゴリに分類するためのモデル"""
    category: Literal["insurance", "meta", "off_topic"] = Field(
        description=(
            "ユーザーの質問を以下の3つのカテゴリに分類する:\n"
            "- 'insurance': 保険の補償内容、約款、手続きなど、提供されたPDF文書に関する具体的な質問。\n"
            "- 'meta': 挨拶、感謝、AI自身に関する質問（例：「あなたは何ができるの？」）。\n"
            "- 'off_topic': 上記以外。保険と全く関係ない話題（例：スポーツ、天気、歴史など）。"
        )
    )

# パーサーの初期化
parser = PydanticOutputParser(pydantic_object=AgentOutput)
classifier_parser = PydanticOutputParser(pydantic_object=QueryClassifier)

# --- 初期化関数 ---
def load_models_and_db():
    """サーバー起動時にモデルやDBを読み込む"""
    global llm, non_streaming_llm, embeddings, db
    
    try:
        if llm is None:
            logger.info("LLMモデルを読み込み中...")
            llm = ChatOpenAI(
                model=settings.model_name,
                temperature=settings.temperature
            )
        
        if non_streaming_llm is None:
            logger.info("非ストリーミングLLMモデルを読み込み中...")
            non_streaming_llm = ChatOpenAI(
                model=settings.model_name,
                temperature=settings.temperature
            )
        
        if embeddings is None:
            logger.info("埋め込みモデルを読み込み中...")
            embeddings = OpenAIEmbeddings()
        
        if db is None:
            logger.info(f"ベクトルストア '{settings.index_path}' を読み込み中...")
            if not os.path.exists(settings.index_path):
                raise FileNotFoundError(
                    f"インデックス '{settings.index_path}' が見つかりません。"
                    f"先に 'python ingest.py' を実行してください。"
                )
            # 自分で作成したインデックスなので True に設定
            db = FAISS.load_local(
                settings.index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("ベクトルストアの読み込み完了")
    
    except Exception as e:
        logger.error(f"モデル/DB読み込みエラー: {e}", exc_info=True)
        raise

# --- ツール定義 ---
@tool
def search_insurance_documents(query: str) -> dict:
    """
    保険の約款、パンフレット、補償内容など、与えられた保険ドキュメントに関する具体的な質問に答えるために使用。
    挨拶や一般的な会話には使用しないこと。
    """
    logger.info(f"HyDE PDF検索ツール実行: query='{query[:50]}...'")
    
    try:
        # HyDE: 質問から理想的な回答(仮説文書)をAIに生成させる
        hyde_prompt = ChatPromptTemplate.from_template(
            "'{question}' という質問に対する、保険の約款に記載されていそうな理想的な回答を日本語で生成せよ。"
        )
        hyde_chain = hyde_prompt | non_streaming_llm
        hypothetical_document = hyde_chain.invoke({"question": query}).content
        
        # 仮説文書を使い、ベクトルDBから関連文書を検索
        retriever = db.as_retriever(search_kwargs={"k": settings.retrieval_k})
        retrieved_docs = retriever.invoke(hypothetical_document)
        
        # 検索結果と元の質問から最終的な回答を生成
        qa_prompt = ChatPromptTemplate.from_template(
            "提供されたコンテキスト情報のみに基づいて質問に答えよ。\n"
            "コンテキスト:\n{context}\n\n質問: {input}"
        )
        document_chain = create_stuff_documents_chain(non_streaming_llm, qa_prompt)
        
        # ドキュメントをcontextキーで渡す
        answer = document_chain.invoke({
            "context": retrieved_docs,  # input_documents ではなく context
            "input": query
        })
        
        if isinstance(answer, dict):
            answer = answer.get("output_text", "")
        
        # 検索結果から引用元情報(ファイル名とページ番号)を抽出
        unique_sources = set()
        for doc in retrieved_docs:
            source_name = os.path.basename(doc.metadata.get('source', '不明'))
            page_num = doc.metadata.get('page', -1) + 1
            if page_num > 0:
                unique_sources.add((source_name, page_num))
        
        sources_list = [
            {"name": name, "page": page}
            for name, page in sorted(list(unique_sources))
        ]
        
        logger.info(f"検索完了: {len(sources_list)}件の引用元")
        return {"answer": answer, "sources": sources_list}
    
    except Exception as e:
        logger.error(f"検索ツールエラー: {e}", exc_info=True)
        return {
            "answer": "申し訳ございません。検索中にエラーが発生しました。",
            "sources": []
        }

# --- エージェント構築 ---
def create_agent_executor():
    """ツールとプロンプトを組み合わせてAIエージェントを作成"""
    tools = [search_insurance_documents]
    format_instructions = parser.get_format_instructions()
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "あなたは親切で優秀な保険相談アシスタントだ。"
         "挨拶や一般的な会話には、ツールを使わずに自然に応答せよ。"
         "保険に関する質問の場合は、必ずツールを使い、その結果を元に回答すること。"
         "最終的な回答は、以下のフォーマット指示に厳密に従うこと。\n\n"
         "{format_instructions}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    prompt = prompt_template.partial(format_instructions=format_instructions)
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    logger.info("エージェント構築完了")
    return executor

# --- 質問分類チェイン構築 ---
def create_classification_chain():
    """質問の意図を判断する「門番」を作成"""
    format_instructions = classifier_parser.get_format_instructions()
    
    prompt_template = ChatPromptTemplate.from_template(
        "ユーザーの質問を分析し、最も適切なカテゴリに分類せよ。\n"
        "{format_instructions}\n\n"
        "ユーザーの質問: '{query}'"
    )
    
    prompt = prompt_template.partial(format_instructions=format_instructions)
    chain = prompt | llm | classifier_parser
    
    logger.info("質問分類チェイン構築完了")
    return chain

# --- サーバー起動時の処理 ---
@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化処理"""
    global agent_executor, classification_chain, session_manager
    
    logger.info("=" * 50)
    logger.info("サーバー起動処理を開始")
    logger.info(f"設定: model={settings.model_name}, chunk_size={settings.chunk_size}")
    
    try:
        load_models_and_db()
        agent_executor = create_agent_executor()
        classification_chain = create_classification_chain()
        session_manager = SessionManager(
            max_sessions=settings.max_sessions,
            timeout_hours=settings.session_timeout_hours
        )
        
        logger.info("サーバーの準備完了")
        logger.info("=" * 50)
    
    except Exception as e:
        logger.error(f"起動処理エラー: {e}", exc_info=True)
        raise

# --- APIエンドポイント ---
@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, chat_request: ChatRequest):
    """チャットAPIエンドポイント"""
    
    session_id = chat_request.session_id or str(uuid4())
    logger.info(f"リクエスト受信: session={session_id}, query_length={len(chat_request.query)}")
    
    # セッション履歴の取得
    chat_history = await session_manager.get(session_id)
    
    try:
        # 1. 質問の分類
        logger.info("質問の分類を開始")
        category_response = classification_chain.invoke({"query": chat_request.query})
        category = category_response.category
        logger.info(f"分類結果: {category}")
        
        answer = ""
        sources = []
        
        # 2. 分類結果に応じて処理を分岐
        if category == "insurance":
            # 保険に関する質問: AIエージェントに処理を任せる
            response = agent_executor.invoke({
                "input": chat_request.query,
                "chat_history": chat_history
            })
            raw_output = response["output"]
            
            try:
                parsed_output = parser.parse(raw_output)
                answer = parsed_output.answer
                sources = parsed_output.sources
            except OutputParserException as e:
                logger.warning(f"パース失敗: {e}. 生の出力を使用します。")
                answer = raw_output
                sources = []
            except Exception as e:
                logger.error(f"予期しないパースエラー: {e}", exc_info=True)
                answer = "申し訳ございません。回答の生成中にエラーが発生しました。"
                sources = []
        
        elif category == "meta":
            # 挨拶や自己紹介: シンプルなAIに直接答えさせる
            meta_prompt = ChatPromptTemplate.from_template(
                "あなたは「保険ドキュメントAIチャット」という名の親切なアシスタントだ。"
                "ユーザーからの'{query}'という質問に、自然な会話で簡潔に答えよ。"
            )
            meta_chain = meta_prompt | llm
            answer = meta_chain.invoke({"query": chat_request.query}).content
            sources = []
        
        else:  # category == "off_topic"
            # 無関係な質問: 固定の拒否メッセージ
            answer = "申し訳ないが、その質問には答えられない。保険の約款に関する質問にのみ回答できる。"
            sources = []
        
        # 3. 会話履歴の更新
        chat_history.append(HumanMessage(content=chat_request.query))
        chat_history.append(AIMessage(content=answer))
        await session_manager.set(session_id, chat_history)
        
        logger.info(f"回答送信: category={category}, sources_count={len(sources)}")
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=session_id
        )
    
    except Exception as e:
        logger.error(f"チャット処理エラー: {e}", exc_info=True)
        return ChatResponse(
            answer="申し訳ございません。処理中にエラーが発生しました。しばらくしてから再度お試しください。",
            sources=[],
            session_id=session_id
        )