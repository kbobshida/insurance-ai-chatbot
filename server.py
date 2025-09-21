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

load_dotenv()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = None
embeddings = None
db = None
agent_executor = None
classification_chain = None
chat_histories = {} 
INDEX_PATH = "faiss_index" 

# --- Pydanticモデル定義 ---
class SourceAPI(BaseModel):
    name: str
    page: int

class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceAPI]
    session_id: str

class AgentOutput(BaseModel):
    answer: str = Field(description="ユーザーへの最終的な回答。会話形式で記述する。")
    sources: List[SourceAPI] = Field(description="回答の生成に使用した引用元のリスト。引用元がない場合は空のリストにする。")

class QueryClassifier(BaseModel):
    category: Literal["insurance", "meta", "off_topic"] = Field(
        description=(
            "ユーザーの質問を以下の3つのカテゴリに分類する:\n"
            "- 'insurance': 保険の補償内容、約款、手続きなど、提供されたPDF文書に関する具体的な質問。\n"
            "- 'meta': 挨拶、感謝、AI自身に関する質問（例：「あなたは何ができるの？」）。\n"
            "- 'off_topic': 上記以外。保険と全く関係ない話題（例：スポーツ、天気、歴史など）。"
        )
    )

parser = PydanticOutputParser(pydantic_object=AgentOutput)
classifier_parser = PydanticOutputParser(pydantic_object=QueryClassifier)

def load_models_and_db():
    global llm, embeddings, db
    if llm is None:
        print("--- LLMモデルを読み込んでいます ---")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    if embeddings is None:
        print("--- 埋め込みモデルを読み込んでいます ---")
        embeddings = OpenAIEmbeddings()
    if db is None:
        print(f"--- ベクトルストア '{INDEX_PATH}' を読み込んでいます ---")
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"インデックス '{INDEX_PATH}' が見つかりません。")
        db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# --- ツール定義 (変更なし) ---
@tool
def search_insurance_documents(query: str) -> dict:
    """
    保険の約款、パンフレット、補償内容など、与えられた保険ドキュメントに関する具体的な質問に答えるために使用します。
    挨拶や一般的な会話には使用しないでください。
    """
    print(f"--- HyDE PDF検索ツール実行: query='{query}' ---")
    hyde_prompt = ChatPromptTemplate.from_template("'{question}' という質問に対する、保険の約款に記載されていそうな理想的な回答を日本語で生成してください。")
    non_streaming_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    hyde_chain = hyde_prompt | non_streaming_llm
    hypothetical_document = hyde_chain.invoke({"question": query}).content
    retriever = db.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(hypothetical_document)
    qa_prompt = ChatPromptTemplate.from_template("提供されたコンテキスト情報のみに基づいて、質問に答えてください。\nコンテキスト:\n{context}\n\n質問: {input}")
    document_chain = create_stuff_documents_chain(non_streaming_llm, qa_prompt)
    answer = document_chain.invoke({"input": query, "context": retrieved_docs})
    unique_sources = set()
    for doc in retrieved_docs:
        source_name = os.path.basename(doc.metadata.get('source', '不明'))
        page_num = doc.metadata.get('page', -1) + 1
        if page_num > 0:
            unique_sources.add((source_name, page_num))
    sources_list = [{"name": name, "page": page} for name, page in sorted(list(unique_sources))]
    return {"answer": answer, "sources": sources_list}

# --- エージェント構築 (変更なし) ---
def create_agent_executor():
    tools = [search_insurance_documents]
    format_instructions = parser.get_format_instructions()
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "あなたは親切で優秀な保険相談アシスタントです。挨拶や一般的な会話には、ツールを使わずに自然に応答してください。保険に関する質問の場合は、必ずツールを使い、その結果を元に回答してください。あなたの最終的な回答は、以下のフォーマット指示に厳密に従ってください。\n\n{format_instructions}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    prompt = prompt_template.partial(format_instructions=format_instructions)
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor

# --- ★★★ ここを修正 ★★★ ---
def create_classification_chain():
    """質問を分類するためのチェインを構築する関数"""
    format_instructions = classifier_parser.get_format_instructions()
    
    # テンプレートを先に定義
    prompt_template = ChatPromptTemplate.from_template(
        "ユーザーの質問を分析し、最も適切なカテゴリに分類してください。\n"
        "{format_instructions}\n\n"
        "ユーザーの質問: '{query}'"
    )
    # .partial()メソッドで固定の指示を埋め込む
    prompt = prompt_template.partial(format_instructions=format_instructions)
    
    chain = prompt | llm | classifier_parser
    return chain

# --- サーバー起動時の処理を修正 (変更なし) ---
@app.on_event("startup")
async def startup_event():
    global agent_executor, classification_chain
    print("--- サーバー起動処理を開始 ---")
    load_models_and_db()
    agent_executor = create_agent_executor()
    classification_chain = create_classification_chain()
    print("--- サーバーの準備が完了しました ---")

# --- APIエンドポイントのロジック (変更なし) ---
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global agent_executor, chat_histories, classification_chain
    
    session_id = request.session_id or str(uuid4())
    chat_history = chat_histories.get(session_id, [])
    
    print(f"--- 質問の分類を開始: query='{request.query}' ---")
    category_response = classification_chain.invoke({"query": request.query})
    category = category_response.category
    print(f"--- 分類結果: {category} ---")

    answer = ""
    sources = []

    if category == "insurance":
        response = agent_executor.invoke({
            "input": request.query,
            "chat_history": chat_history,
        })
        raw_output = response["output"]
        try:
            parsed_output = parser.parse(raw_output)
            answer = parsed_output.answer
            sources = parsed_output.sources
        except Exception:
            answer = raw_output
            sources = []
            
    elif category == "meta":
        meta_prompt = ChatPromptTemplate.from_template(
            "あなたは「保険ドキュメントAIチャット」という名前の、親切なアシスタントです。"
            "ユーザーからの'{query}'という質問に、自然な会話で簡潔に答えてください。"
        )
        meta_chain = meta_prompt | llm
        answer = meta_chain.invoke({"query": request.query}).content
        sources = []
        
    else: # category == "off_topic"
        answer = "申し訳ありませんが、そのご質問にはお答えできません。私は保険の約款に関するご質問のみ回答できます。"
        sources = []

    chat_history.append(HumanMessage(content=request.query))
    chat_history.append(AIMessage(content=answer))
    chat_histories[session_id] = chat_history
    
    return ChatResponse(answer=answer, sources=sources, session_id=session_id)

