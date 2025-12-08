from typing import Annotated, Optional
import operator
import asyncio
from pydantic import BaseModel, Field
from langgraph.types import Send
from typing_extensions import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mcp_adapters.client import MultiServerMCPClient
from pinecone import Pinecone
from langchain.agents import create_agent
import requests
from requests_aws4auth import AWS4Auth
from llama_cloud_services import LlamaParse
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import logging
import json
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

# Global instances
orchestration_model = ChatOpenAI(model_name="gpt-4o-mini", api_key=OPENAI_API_KEY)
model = ChatOpenAI(model_name="gpt-4o-mini", api_key=OPENAI_API_KEY)

MCP_TOOLS = []
MCP_CLIENT = None
PINECONE_CLIENT = None
TEXT_SPLITTER = None
GRAPH = None

# Removed DOCUMENT_CACHE - using checkpointer instead

def get_text_splitter():
    global TEXT_SPLITTER
    if TEXT_SPLITTER is None:
        TEXT_SPLITTER = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=10000, chunk_overlap=200)
    return TEXT_SPLITTER

def get_pinecone_client():
    global PINECONE_CLIENT
    if PINECONE_CLIENT is None:
        PINECONE_CLIENT = Pinecone(api_key=PINECONE_API_KEY)
    return PINECONE_CLIENT

@tool
async def retrieve_relevant_documents(query: str):
    """Search internal database for relevant documents using dense and sparse search."""
    pc = get_pinecone_client()
    dense_index = pc.Index(host="https://safeclause-dc5puwa.svc.aped-4627-b74a.pinecone.io")
    sparse_index = pc.Index(host="https://safeclause-sparse-dc5puwa.svc.aped-4627-b74a.pinecone.io")

    dense_task = asyncio.create_task(asyncio.to_thread(lambda: dense_index.search(namespace="Acts_and_Clause_Namespace", query={"top_k": 35, "inputs": {"text": query}})))
    sparse_task = asyncio.create_task(asyncio.to_thread(lambda: sparse_index.search(namespace="Acts_and_Clause_Namespace", query={"top_k": 35, "inputs": {"text": query}})))
    
    dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

    combined_results = {}
    for match in dense_results['result']['hits']:
        combined_results[match['_id']] = match
    for match in sparse_results['result']['hits']:
        if match['_id'] not in combined_results:
            combined_results[match['_id']] = match

    documents = [{"id": match['_id'], "text": match['fields']['text'], "act_name": match['fields']['act_name']} for match in combined_results.values()]

    reranked = pc.inference.rerank(model="pinecone-rerank-v0", query=query, documents=documents, rank_fields=["text"], top_n=30, return_documents=True)
    return str(reranked)

async def setup_mcp_client():
    global MCP_CLIENT
    if MCP_CLIENT is None:
        MCP_CLIENT = MultiServerMCPClient({"tavily": {"transport": "streamable_http", "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"}})
        mcp_tools = await MCP_CLIENT.get_tools()
        return [t for t in mcp_tools if t.name in ["tavily_search", "tavily_extract"]]
    return []

class Section(BaseModel):
    name: str = Field(description="Section name")
    content: str = Field(description="Chunked content")
    chunk_index: int = Field(description="Chunk index")

class State(TypedDict):
    query: str
    session_id: str
    doc_filepath: str
    doc_content: str
    has_document: bool
    chunks: list[Section]
    completed_sections: Annotated[list, operator.add]
    final_report: str
    messages: Annotated[list, operator.add]

class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]

async def download_and_parse_document(state: State):
    session_id = state.get("session_id", "default")
    
    # Check if document already exists in state (from checkpointer)
    if state.get("doc_content") and len(state["doc_content"].strip()) > 0:
        return {
            "has_document": True,
            "messages": [HumanMessage(content="Document already in memory")]
        }
    
    doc_filepath = state.get("doc_filepath")
    if not doc_filepath:
        return {
            "doc_content": "",
            "has_document": False,
            "messages": [HumanMessage(content="No document filepath provided")]
        }
    
    try:
        access_key = AWS_ACCESS_KEY
        secret_key = AWS_SECRET_KEY
        auth = AWS4Auth(access_key, secret_key, 'us-east-1', 's3')
        
        response = requests.get(doc_filepath, auth=auth)
        if response.status_code != 200:
            return {
                "doc_content": "",
                "has_document": False,
                "messages": [HumanMessage(content=f"Failed to download: {response.status_code}")]
            }
        
        filename = doc_filepath.split('/')[-1]
        temp_path = f"/tmp/{filename}"
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        
        try:
            parser = LlamaParse(api_key=LLAMAPARSE_API_KEY, num_workers=4, verbose=False, language="en")
            result = parser.parse(temp_path)
            
            all_text = ""
            for page_object in result.pages:
                all_text += f"\n\n=== Page {page_object.page} ===\n{page_object.text.strip()}\n"
            
            return {
                "doc_content": all_text,
                "has_document": True if all_text.strip() else False,
                "messages": [HumanMessage(content="Document downloaded and parsed")]
            }
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    except Exception as e:
        logger.error(f"Error downloading document: {str(e)}")
        return {
            "doc_content": "",
            "has_document": False,
            "messages": [HumanMessage(content=f"Error: {str(e)}")]
        }

async def orchestration_agent(state: State):
    has_document = bool(state.get("doc_content") and len(state["doc_content"].strip()) > 0)
    if has_document:
        return {
            "has_document": True,
            "messages": [HumanMessage(content="Document detected")]
        }
    
    all_tools = [retrieve_relevant_documents]
    if MCP_TOOLS:
        all_tools.extend(MCP_TOOLS)
    
    agent = create_agent(
        orchestration_model,
        tools=all_tools,
        system_prompt="You are an expert Indian Legal AI Agent. Provide accurate legal guidance based on the Indian Constitution, Acts, and Judicial Precedents. Use RAG as primary source and WebSearch for recent judgments."
    )
    
    # Use messages from state for conversation history
    messages = state.get("messages", [])
    if not messages:
        messages = [{"role": "user", "content": state['query']}]
    else:
        # Append new query as a user message
        messages.append({"role": "user", "content": state['query']})
    
    result = await agent.ainvoke({"messages": messages})
    final_answer = result["messages"][-1].content if result["messages"] else "No answer generated"
    
    return {
        "has_document": False,
        "final_report": final_answer,
        "messages": [HumanMessage(content=final_answer)]
    }

async def route_after_orchestration(state: State) -> Literal["chunk_document", "answer_from_cache", "END"]:
    has_document = bool(state.get("doc_content") and len(state["doc_content"].strip()) > 0)
    if not has_document:
        return "END"
    if state.get("completed_sections") and len(state["completed_sections"]) > 0:
        return "answer_from_cache"
    return "chunk_document"

async def answer_from_cache(state: State):
    cached_sections = state.get("completed_sections", [])
    combined_context = "\n\n---\n\n".join(cached_sections)
    
    all_tools = [retrieve_relevant_documents]
    if MCP_TOOLS:
        all_tools.extend(MCP_TOOLS)
    
    agent = create_agent(
        model,
        tools=all_tools,
        system_prompt="You are an expert Indian Legal AI Agent. Answer the user's question based on the provided document analysis."
    )
    
    # Use messages from state for conversation history
    messages = state.get("messages", [])
    if not messages:
        messages = [{"role": "user", "content": f"Document Analysis:\n{combined_context}\n\nUser Question: {state['query']}"}]
    else:
        # Append new query
        messages.append({"role": "user", "content": f"Document Analysis:\n{combined_context}\n\nUser Question: {state['query']}"})
    
    result = await agent.ainvoke({"messages": messages})
    final_answer = result["messages"][-1].content if result["messages"] else "No answer generated"
    
    return {
        "final_report": final_answer,
        "messages": [HumanMessage(content=final_answer)]
    }

async def chunk_document(state: State):
    text_splitter = get_text_splitter()
    splits = text_splitter.split_text(state["doc_content"])
    chunks = [Section(name=f"Section {i+1}", content=splits[i], chunk_index=i) for i in range(len(splits))]
    return {"chunks": chunks}

async def analyze_chunk(state: WorkerState):
    all_tools = [retrieve_relevant_documents]
    if MCP_TOOLS:
        all_tools.extend(MCP_TOOLS)
    
    agent = create_agent(
        model,
        tools=all_tools,
        system_prompt="Analyze this legal document section for validity, risks, and compliance."
    )
    
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": f"Section {state['section'].chunk_index}:\n{state['section'].content}\n\nAnalyze this section."}]
    })
    
    analysis = result["messages"][-1].content if result["messages"] else "No analysis generated"
    return {"completed_sections": [analysis]}

async def assign_workers(state: State):
    return [Send("analyze_chunk", {"section": chunk}) for chunk in state["chunks"]]

async def synthesizer(state: State):
    all_tools = [retrieve_relevant_documents]
    if MCP_TOOLS:
        all_tools.extend(MCP_TOOLS)
    
    completed_sections = state["completed_sections"]
    combined_analysis = "\n\n---\n\n".join(completed_sections)
    
    agent = create_agent(
        model,
        tools=all_tools,
        system_prompt="Synthesize these section analyses into a comprehensive legal review report with risk scores and recommendations."
    )
    
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": f"Combine these {len(completed_sections)} analyses into a comprehensive report:\n\n{combined_analysis}"}]
    })
    
    final_report = result["messages"][-1].content if result["messages"] else combined_analysis
    return {"final_report": final_report}

async def build_graph():
    global MCP_TOOLS, GRAPH
    
    MCP_TOOLS = await setup_mcp_client()
    
    builder = StateGraph(State)
    builder.add_node("download_and_parse_document", download_and_parse_document)
    builder.add_node("orchestration_agent", orchestration_agent)
    builder.add_node("answer_from_cache", answer_from_cache)
    builder.add_node("chunk_document", chunk_document)
    builder.add_node("analyze_chunk", analyze_chunk)
    builder.add_node("synthesizer", synthesizer)

    builder.add_edge(START, "download_and_parse_document")
    builder.add_edge("download_and_parse_document", "orchestration_agent")
    
    builder.add_conditional_edges(
        "orchestration_agent",
        route_after_orchestration,
        {"chunk_document": "chunk_document", "answer_from_cache": "answer_from_cache", "END": END}
    )
    
    builder.add_edge("answer_from_cache", END)
    builder.add_conditional_edges("chunk_document", assign_workers, ["analyze_chunk"])
    builder.add_edge("analyze_chunk", "synthesizer")
    builder.add_edge("synthesizer", END)

    checkpointer = MemorySaver()
    GRAPH = builder.compile(checkpointer=checkpointer)

class QueryRequest(BaseModel):
    query: str = Field(..., description="User's legal query")
    doc_filepath: Optional[str] = Field(None, description="URL path to document")
    session_id: str = Field(default="default_session", description="Session ID")

class HealthResponse(BaseModel):
    status: str
    message: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    await build_graph()
    yield

app = FastAPI(title="Legal Document Analysis API", version="1.0.0", lifespan=lifespan)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", message="Legal Document Analysis API is running")

@app.post("/query")
async def query(request: QueryRequest):
    try:
        if not GRAPH:
            raise HTTPException(status_code=500, detail="Graph not initialized")
        
        session_id = request.session_id
        config = {"configurable": {"thread_id": session_id}}
        
        # Retrieve previous state from checkpointer
        previous_state = GRAPH.get_state(config)
        
        # Extract previous messages and state values
        if previous_state and previous_state.values:
            previous_messages = previous_state.values.get("messages", [])
            previous_doc_content = previous_state.values.get("doc_content", "")
            previous_chunks = previous_state.values.get("chunks", [])
            previous_completed_sections = previous_state.values.get("completed_sections", [])
        else:
            previous_messages = []
            previous_doc_content = ""
            previous_chunks = []
            previous_completed_sections = []
        
        # Append new user message to conversation history
        new_user_message = HumanMessage(content=request.query)
        all_messages = previous_messages + [new_user_message]
        
        # Build input state with conversation history
        input_state = {
            "query": request.query,
            "session_id": session_id,
            "doc_filepath": request.doc_filepath or "",
            "doc_content": previous_doc_content,  # Keep previous document if it exists
            "has_document": bool(previous_doc_content.strip()) if previous_doc_content else False,
            "chunks": previous_chunks,  # Keep previous chunks
            "completed_sections": previous_completed_sections,  # Keep previous analysis
            "final_report": "",
            "messages": all_messages,  # Full conversation history
        }
        
        # Invoke graph with config to save checkpoint
        result = await GRAPH.ainvoke(input_state, config)
        
        async def generate():
            final_report = result.get("final_report", "")
            for chunk in final_report.split('\n'):
                if chunk.strip():
                    yield json.dumps({"text": chunk}) + "\n"
                    await asyncio.sleep(0.05)
        
        return StreamingResponse(generate(), media_type="application/x-ndjson")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        async def error():
            yield json.dumps({"error": str(e)}) + "\n"
        return StreamingResponse(error(), media_type="application/x-ndjson")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")