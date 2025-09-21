# 必要なライブラリやモジュールをインポート
import os
import json
import logging
from uuid import uuid4
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Literal

# LangChainの主要コンポーネントをインポート
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.agents import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser

# .envファイルから環境変数を読み込む
load_dotenv()

# FastAPIアプリケーションを初期化
app = FastAPI()

# '/static'パスで'static'フォルダ内のファイル(CSS, JS)を提供
app.mount("/static", StaticFiles(directory="static"), name="static")

# ルートURL("/")へのアクセス時に'index.html'を返す
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

# CORS (クロスオリジンリソース共有) の設定。ブラウザからのAPIアクセスを許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- グローバル変数: アプリケーション全体で共有する変数を定義 ---
llm = None
embeddings = None
db = None
agent_executor = None
classification_chain = None # 質問分類用のチェイン
chat_histories = {} 
INDEX_PATH = "faiss_index" 

# --- Pydanticモデル定義: APIのデータ構造を定義 ---

# 引用元情報の型
class SourceAPI(BaseModel):
    name: str
    page: int

# フロントエンドからのリクエストの型
class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None

# フロントエンドへのレスポンスの型
class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceAPI]
    session_id: str

# LangChainエージェントに出力形式を指示するためのモデル
class AgentOutput(BaseModel):
    answer: str = Field(description="ユーザーへの最終的な回答。会話形式で記述する。")
    sources: List[SourceAPI] = Field(description="回答の生成に使用した引用元のリスト。引用元がない場合は空のリストにする。")

# 質問を3カテゴリに分類するためのモデル
class QueryClassifier(BaseModel):
    category: Literal["insurance", "meta", "off_topic"] = Field(
        description=(
            "ユーザーの質問を以下の3つのカテゴリに分類する:\n"
            "- 'insurance': 保険の補償内容、約款、手続きなど、提供されたPDF文書に関する具体的な質問。\n"
            "- 'meta': 挨拶、感謝、AI自身に関する質問（例：「あなたは何ができるの？」）。\n"
            "- 'off_topic': 上記以外。保険と全く関係ない話題（例：スポーツ、天気、歴史など）。"
        )
    )

# Pydanticモデルを使い、AI出力を解析するパーサーを作成
parser = PydanticOutputParser(pydantic_object=AgentOutput)
classifier_parser = PydanticOutputParser(pydantic_object=QueryClassifier)

# --- 初期化関数: サーバー起動時にモデルやDBを読み込む ---
def load_models_and_db():
    global llm, embeddings, db
    if llm is None:
        print("--- LLMモデルを読み込み中 ---")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    if embeddings is None:
        print("--- 埋め込みモデルを読み込み中 ---")
        embeddings = OpenAIEmbeddings()
    if db is None:
        print(f"--- ベクトルストア '{INDEX_PATH}' を読み込み中 ---")
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"インデックス '{INDEX_PATH}' が見つからない。")
        db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# --- ツール定義: AIエージェントが使用できる道具を定義 ---
@tool
def search_insurance_documents(query: str) -> dict:
    """
    保険の約款、パンフレット、補償内容など、与えられた保険ドキュメントに関する具体的な質問に答えるために使用。
    挨拶や一般的な会話には使用しないこと。
    """
    print(f"--- HyDE PDF検索ツール実行: query='{query}' ---")

    # HyDE: 質問から理想的な回答(仮説文書)をAIに生成させる
    hyde_prompt = ChatPromptTemplate.from_template("'{question}' という質問に対する、保険の約款に記載されていそうな理想的な回答を日本語で生成せよ。")
    non_streaming_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # ツール内では非ストリーミングモデルを使用
    hyde_chain = hyde_prompt | non_streaming_llm
    hypothetical_document = hyde_chain.invoke({"question": query}).content
    
    # 仮説文書を使い、ベクトルDBから関連文書を検索
    retriever = db.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(hypothetical_document)

    # 検索結果と元の質問から最終的な回答を生成
    qa_prompt = ChatPromptTemplate.from_template("提供されたコンテキスト情報のみに基づいて質問に答えよ。\nコンテキスト:\n{context}\n\n質問: {input}")
    document_chain = create_stuff_documents_chain(non_streaming_llm, qa_prompt)
    answer = document_chain.invoke({"input": query, "context": retrieved_docs})
    
    # 検索結果から引用元情報(ファイル名とページ番号)を抽出
    unique_sources = set()
    for doc in retrieved_docs:
        source_name = os.path.basename(doc.metadata.get('source', '不明'))
        page_num = doc.metadata.get('page', -1) + 1
        if page_num > 0:
            unique_sources.add((source_name, page_num))
    sources_list = [{"name": name, "page": page} for name, page in sorted(list(unique_sources))]
    
    return {"answer": answer, "sources": sources_list}

# --- エージェント構築: ツールとプロンプトを組み合わせてAIエージェントを作成 ---
def create_agent_executor():
    tools = [search_insurance_documents]
    format_instructions = parser.get_format_instructions()
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "あなたは親切で優秀な保険相談アシスタントだ。挨拶や一般的な会話には、ツールを使わずに自然に応答せよ。保険に関する質問の場合は、必ずツールを使い、その結果を元に回答すること。最終的な回答は、以下のフォーマット指示に厳密に従うこと。\n\n{format_instructions}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    prompt = prompt_template.partial(format_instructions=format_instructions)
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor

# --- 質問分類チェイン構築: 質問の意図を判断する「門番」を作成 ---
def create_classification_chain():
    format_instructions = classifier_parser.get_format_instructions()
    prompt_template = ChatPromptTemplate.from_template(
        "ユーザーの質問を分析し、最も適切なカテゴリに分類せよ。\n"
        "{format_instructions}\n\n"
        "ユーザーの質問: '{query}'"
    )
    prompt = prompt_template.partial(format_instructions=format_instructions)
    chain = prompt | llm | classifier_parser
    return chain

# --- サーバー起動時の処理 ---
@app.on_event("startup")
async def startup_event():
    global agent_executor, classification_chain
    print("--- サーバー起動処理を開始 ---")
    load_models_and_db()
    agent_executor = create_agent_executor()
    classification_chain = create_classification_chain()
    print("--- サーバーの準備完了 ---")

# --- APIエンドポイント: ブラウザからのリクエストを処理 ---
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global agent_executor, chat_histories, classification_chain
    
    session_id = request.session_id or str(uuid4())
    chat_history = chat_histories.get(session_id, [])
    
    # 1. 最初に質問を分類
    print(f"--- 質問の分類を開始: query='{request.query}' ---")
    category_response = classification_chain.invoke({"query": request.query})
    category = category_response.category
    print(f"--- 分類結果: {category} ---")

    answer = ""
    sources = []

    # 2. 分類結果に応じて処理を分岐
    if category == "insurance":
        # 「保険に関する質問」なら、AIエージェントに処理を任せる
        response = agent_executor.invoke({"input": request.query, "chat_history": chat_history})
        raw_output = response["output"]
        try:
            parsed_output = parser.parse(raw_output)
            answer = parsed_output.answer
            sources = parsed_output.sources
        except Exception:
            answer = raw_output
            sources = []
            
    elif category == "meta":
        # 「挨拶や自己紹介」なら、シンプルなAIに直接答えさせる
        meta_prompt = ChatPromptTemplate.from_template("あなたは「保険ドキュメントAIチャット」という名の親切なアシスタントだ。ユーザーからの'{query}'という質問に、自然な会話で簡潔に答えよ。")
        meta_chain = meta_prompt | llm
        answer = meta_chain.invoke({"query": request.query}).content
        sources = []
        
    else: # category == "off_topic"
        # 「無関係な質問」なら、固定の拒否メッセージを返す
        answer = "申し訳ないが、その質問には答えられない。保険の約款に関する質問にのみ回答できる。"
        sources = []

    # 3. 会話履歴を更新し、最終的な回答をブラウザに返す
    chat_history.append(HumanMessage(content=request.query))
    chat_history.append(AIMessage(content=answer))
    chat_histories[session_id] = chat_history
    
    return ChatResponse(answer=answer, sources=sources, session_id=session_id)

