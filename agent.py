import logging
from typing import Annotated
import operator
import asyncio
import time
import tiktoken
from pydantic import BaseModel, Field
from langgraph.types import Send
from typing_extensions import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool
from langchain.messages import HumanMessage, trim_messages
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mcp_adapters.client import MultiServerMCPClient
from pinecone import Pinecone
from langchain.agents import create_agent
import requests
from requests_aws4auth import AWS4Auth
import os
from dotenv import load_dotenv
from tools import retrieve_documents_helper
from utils import document_parser

load_dotenv()

logger = logging.getLogger(__name__)

# Environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

# Token management and rate limiting
MAX_MESSAGES_TO_KEEP = 3  # Reduced from 5
MAX_TOKENS_PER_REQUEST = 6000  # Well under 8000 TPM limit
MAX_CONTEXT_TOKENS = 4000  # For document context
MAX_CHUNK_SIZE = 3000  # Reduced chunk size
MIN_REQUEST_INTERVAL = 8  # Minimum 8 seconds between requests to stay under TPM
LAST_REQUEST_TIME = 0

# Initialize tiktoken encoder for token counting
try:
    ENCODER = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Use compatible encoder
except Exception:
    ENCODER = tiktoken.get_encoding("cl100k_base")  # Fallback


class Section(BaseModel):
    name: str = Field(description="Section name")
    content: str = Field(description="Chunked content")
    chunk_index: int = Field(description="Chunk index")


class State(TypedDict):
    query: str
    session_id: str
    doc_filepaths: list[str]  
    doc_contents: list[dict]  
    has_documents: bool  
    chunks: list[Section]
    completed_sections: Annotated[list, operator.add]
    final_report: str
    messages: Annotated[list, operator.add]


class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]


orchestration_model = ChatGroq(model_name="openai/gpt-oss-120b", api_key=GROQ_API_KEY)
model = ChatGroq(model_name="openai/gpt-oss-20b", api_key=GROQ_API_KEY)

MCP_TOOLS = []
MCP_CLIENT = None
TEXT_SPLITTER = None
GRAPH = None


def count_tokens(text: str) -> int:
    """Count tokens in a text string using tiktoken."""
    try:
        return len(ENCODER.encode(text))
    except Exception as e:
        logger.warning(f"Token counting failed: {e}")
        # Fallback: rough estimate (4 chars per token)
        return len(text) // 4

def truncate_by_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token limit."""
    try:
        tokens = ENCODER.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        truncated_text = ENCODER.decode(truncated_tokens)
        logger.info(f"Truncated text from {len(tokens)} to {len(truncated_tokens)} tokens")
        return truncated_text + "\n... [TRUNCATED DUE TO TOKEN LIMIT]"
    except Exception as e:
        logger.warning(f"Token truncation failed: {e}")
        # Fallback: character-based truncation
        char_limit = max_tokens * 4  # Rough estimate
        if len(text) <= char_limit:
            return text
        return text[:char_limit] + "\n... [TRUNCATED DUE TO TOKEN LIMIT]"

def trim_message_history(messages: list) -> list:
    """
    Trim message history to prevent context length exceeded errors.
    Keeps only the last MAX_MESSAGES_TO_KEEP messages and ensures token limits.
    """
    if not messages:
        return messages
    
    # Keep only the last N messages
    if len(messages) > MAX_MESSAGES_TO_KEEP:
        logger.info(f"Trimming messages from {len(messages)} to {MAX_MESSAGES_TO_KEEP}")
        messages = messages[-MAX_MESSAGES_TO_KEEP:]
    
    # Check and truncate individual messages if needed
    trimmed_messages = []
    total_tokens = 0
    
    for msg in reversed(messages):  # Process newest first
        if isinstance(msg, dict) and "content" in msg:
            content = msg["content"]
        elif hasattr(msg, "content"):
            content = msg.content
        else:
            content = str(msg)
        
        msg_tokens = count_tokens(content)
        
        # If adding this message would exceed limit, truncate it
        if total_tokens + msg_tokens > MAX_CONTEXT_TOKENS:
            remaining_tokens = MAX_CONTEXT_TOKENS - total_tokens
            if remaining_tokens > 100:  # Only include if we have reasonable space
                content = truncate_by_tokens(content, remaining_tokens)
                if isinstance(msg, dict):
                    msg = {**msg, "content": content}
                elif hasattr(msg, "content"):
                    msg.content = content
                trimmed_messages.insert(0, msg)
            break
        else:
            trimmed_messages.insert(0, msg)
            total_tokens += msg_tokens
    
    logger.info(f"Final message history: {len(trimmed_messages)} messages, ~{total_tokens} tokens")
    return trimmed_messages

async def enforce_rate_limit():
    """Enforce rate limiting to stay within Groq's TPM limits."""
    global LAST_REQUEST_TIME
    
    current_time = time.time()
    time_since_last = current_time - LAST_REQUEST_TIME
    
    if time_since_last < MIN_REQUEST_INTERVAL:
        sleep_time = MIN_REQUEST_INTERVAL - time_since_last
        logger.info(f"Rate limiting: sleeping for {sleep_time:.1f} seconds")
        await asyncio.sleep(sleep_time)
    
    LAST_REQUEST_TIME = time.time()

def prepare_safe_content(content: str, max_tokens: int = MAX_TOKENS_PER_REQUEST) -> str:
    """Prepare content ensuring it's within token limits."""
    token_count = count_tokens(content)
    
    if token_count > max_tokens:
        logger.warning(f"Content exceeds token limit: {token_count} > {max_tokens}")
        return truncate_by_tokens(content, max_tokens)
    
    return content


def get_text_splitter():
    global TEXT_SPLITTER
    if TEXT_SPLITTER is None:
        # Reduced chunk size to stay within token limits
        TEXT_SPLITTER = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=MAX_CHUNK_SIZE, 
            chunk_overlap=100
        )
    return TEXT_SPLITTER


async def setup_mcp_client():
    global MCP_CLIENT
    if MCP_CLIENT is None:
        MCP_CLIENT = MultiServerMCPClient({
            "tavily": {
                "transport": "streamable_http",
                "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
            }
        })
        mcp_tools = await MCP_CLIENT.get_tools()  # Load FIRST
        
        # Now debug schema
        tavily_tool = next((t for t in mcp_tools if t.name == "tavily_search"), None)
        if tavily_tool:
            # Log schema (adjust attr if needed)
            schema_info = getattr(tavily_tool, 'args_schema', getattr(tavily_tool, 'schema', 'Unknown'))
            logger.info(f"tavily_search schema: {schema_info}")
        
        return [t for t in mcp_tools if t.name in ["tavily_search", "tavily_extract"]]
    return MCP_TOOLS  # Cache existing


@tool
async def retrieve_relevant_documents(query: str):
    """Search internal database for relevant documents using dense and sparse search."""
    return await retrieve_documents_helper(query)


async def download_and_parse_document(state: State):
    session_id = state.get("session_id", "default")
    
    if state.get("doc_contents") and len(state["doc_contents"]) > 0:
        return {
            "has_documents": True,
            "messages": [HumanMessage(content=f"Documents already in memory: {len(state['doc_contents'])} files")]
        }
    
    doc_filepaths = state.get("doc_filepaths", [])
    if not doc_filepaths:
        return {
            "doc_contents": [],
            "has_documents": False,
            "messages": [HumanMessage(content="No documents provided")]
        }
    
    doc_contents = []
    errors = []
    
    try:
        access_key = AWS_ACCESS_KEY
        secret_key = AWS_SECRET_KEY
        auth = AWS4Auth(access_key, secret_key, 'us-east-1', 's3')
        
        for doc_filepath in doc_filepaths:
            try:
                logger.info(f"Processing document: {doc_filepath}")
                
                response = requests.get(doc_filepath, auth=auth)
                if response.status_code != 200:
                    errors.append(f"Failed to download {doc_filepath}: {response.status_code}")
                    continue
                
                filename = doc_filepath.split('/')[-1]
                temp_path = f"/tmp/{filename}"
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                
                try:
                    all_text = await document_parser(temp_path)
                    
                    if all_text.strip():
                        doc_contents.append({
                            "filepath": doc_filepath,
                            "filename": filename,
                            "content": all_text
                        })
                        logger.info(f"✅ Parsed {filename}: {len(all_text)} characters")
                    else:
                        errors.append(f"No text extracted from {filename}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            except Exception as e:
                logger.error(f"Error processing {doc_filepath}: {str(e)}")
                errors.append(f"Error processing {doc_filepath}: {str(e)}")
        
        message = f"Parsed {len(doc_contents)} documents successfully"
        if errors:
            message += f". Errors: {'; '.join(errors[:3])}"
        
        return {
            "doc_contents": doc_contents,
            "has_documents": len(doc_contents) > 0,
            "messages": [HumanMessage(content=message)]
        }
    
    except Exception as e:
        logger.error(f"Error in document processing: {str(e)}")
        return {
            "doc_contents": [],
            "has_documents": False,
            "messages": [HumanMessage(content=f"Error: {str(e)}")]
        }


async def orchestration_agent(state: State):
    has_documents = state.get("has_documents", False)
    if has_documents:
        return {
            "has_documents": True,
            "messages": [HumanMessage(content=f"Documents detected: {len(state.get('doc_contents', []))} files")]
        }
    
    # Enforce rate limiting
    await enforce_rate_limit()
    
    all_tools = [retrieve_relevant_documents]
    if MCP_TOOLS:
        all_tools.extend(MCP_TOOLS)
    
    agent = create_agent(
        orchestration_model,
        tools=all_tools,
        system_prompt="""
You are SafeClause.ai, an expert Indian Legal AI Agent providing accurate, practical guidance.

Keep responses concise and under 4000 tokens. Be direct and to the point.

──────── TOOL USAGE ────────
• VectorDB (RAG) — PRIMARY: Statutes, sections, definitions, case law.
• WebSearch — SECONDARY: Recent judgments, amendments.

──────── CORE LEGAL RULES ────────
1. Old vs New Codes: Always provide BNS/BNSS/BSA equivalents for IPC/CrPC/Evidence Act.
2. Accuracy: Cite exact Acts and sections; flag uncertainty.
3. Tone: Professional, neutral, precise.

──────── RESPONSE STRUCTURE ────────
1. Direct Answer
2. Legal Provisions (Acts and sections)
3. Case Law (if relevant)
4. Practical Notes
5. Disclaimer: "Information is for educational purposes only."

Keep responses focused and under token limits.
"""
    )
    
    # Trim messages and ensure token limits
    messages = state.get("messages", [])
    messages = trim_message_history(messages)
    
    # Prepare safe query content
    safe_query = prepare_safe_content(state['query'], 1000)  # Limit query to 1000 tokens
    
    if not messages:
        messages = [{"role": "user", "content": safe_query}]
    else:
        messages.append({"role": "user", "content": safe_query})
    
    try:
        result = await agent.ainvoke({"messages": messages})
        final_answer = result["messages"][-1].content if result["messages"] else "No answer generated"
        
        # Truncate response if needed
        final_answer = prepare_safe_content(final_answer, MAX_TOKENS_PER_REQUEST)
        
        return {
            "has_documents": False,
            "final_report": final_answer,
            "messages": [HumanMessage(content=final_answer)]
        }
    except Exception as e:
        logger.error(f"Error in orchestration_agent: {str(e)}")
        return {
            "has_documents": False,
            "final_report": f"Error processing request: {str(e)}",
            "messages": [HumanMessage(content=f"Error: {str(e)}")]
        }


async def route_after_orchestration(state: State) -> Literal["chunk_document", "answer_from_cache", "END"]:
    has_documents = state.get("has_documents", False)
    if not has_documents:
        return "END"
    if state.get("completed_sections") and len(state["completed_sections"]) > 0:
        return "answer_from_cache"
    return "chunk_document"


def truncate_text(text: str, max_chars: int = 25000) -> str:
    """
    Truncates text to ensure it fits within LLM context window.
    25,000 chars is roughly 6,000-7,000 tokens.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [CONTENT TRUNCATED DUE TO LENGTH] ..."


async def answer_from_cache(state: State):
    # Enforce rate limiting
    await enforce_rate_limit()
    
    cached_sections = state.get("completed_sections", [])
    combined_context = "\n\n---\n\n".join(cached_sections)
    
    # Use token-based truncation instead of character-based
    safe_context = prepare_safe_content(combined_context, MAX_CONTEXT_TOKENS)
    
    doc_info = "\n".join([f"- {doc['filename']}" for doc in state.get("doc_contents", [])])
    
    all_tools = [retrieve_relevant_documents]
    if MCP_TOOLS:
        all_tools.extend(MCP_TOOLS)
    
    agent = create_agent(
        model,
        tools=all_tools,
        system_prompt="""
SafeClause.ai: Indian Legal AI Agent. Keep responses under 4000 tokens.

**TOOL USAGE**
1. **VectorDB (RAG):** PRIMARY for Acts, sections, definitions, case law.
2. **WebSearch:** SECONDARY for recent judgments, amendments.

**CORE RULES**
1. Old vs New Law: Provide BNS/BNSS/BSA equivalents for IPC/CrPC/Evidence Act.
2. Accuracy: Cite exact Acts and sections.
3. Tone: Professional, concise.

**RESPONSE FORMAT**
1. Direct Answer
2. Legal Provisions (Acts and sections)
3. Case Law (if relevant)
4. Disclaimer: "Educational purposes only."

Keep responses focused and under token limits.
"""
    )
    
    messages = state.get("messages", [])
    messages = trim_message_history(messages)
    
    # Prepare safe prompt content
    safe_query = prepare_safe_content(state['query'], 800)  # Limit query
    prompt_content = f"Documents:\n{doc_info}\n\nAnalysis:\n{safe_context}\n\nQuestion: {safe_query}"
    prompt_content = prepare_safe_content(prompt_content, MAX_CONTEXT_TOKENS)
    
    if not messages:
        messages = [{"role": "user", "content": prompt_content}]
    else:
        messages.append({"role": "user", "content": prompt_content})
    
    try:
        result = await agent.ainvoke({"messages": messages})
        final_answer = result["messages"][-1].content if result["messages"] else "No answer generated"
        
        # Truncate response if needed
        final_answer = prepare_safe_content(final_answer, MAX_TOKENS_PER_REQUEST)
        
        return {
            "final_report": final_answer,
            "messages": [HumanMessage(content=final_answer)]
        }
    except Exception as e:
        logger.error(f"Error in answer_from_cache: {str(e)}")
        return {
            "final_report": f"Error processing cached analysis: {str(e)}",
            "messages": [HumanMessage(content=f"Error: {str(e)}")]
        }


async def chunk_document(state: State):
    text_splitter = get_text_splitter()
    chunks = []
    
    for doc_idx, doc_info in enumerate(state.get("doc_contents", [])):
        splits = text_splitter.split_text(doc_info["content"])
        for chunk_idx, split in enumerate(splits):
            chunks.append(
                Section(
                    name=f"{doc_info['filename']} - Section {chunk_idx + 1}",
                    content=split,
                    chunk_index=len(chunks)
                )
            )
    
    return {"chunks": chunks}


async def analyze_chunk(state: WorkerState):
    all_tools = [retrieve_relevant_documents]
    if MCP_TOOLS:
        all_tools.extend(MCP_TOOLS)
    
    agent = create_agent(
        model,
        tools=all_tools,
        system_prompt="""
**ROLE**
Legal Document Analyst.

**TASK**
Analyze the provided document section for legal validity, risks, and compliance.

**MANDATORY TOOL PROTOCOL**
1. **RAG:** Cross-reference the input text with stored Indian laws and standard clauses.
2. **WebSearch:** Verify the clause's current validity against the latest amendments and judgments.
   - example: tavily_search(query: str, max_results=3–10), without using top_n or any other param.

**OUTPUT STRUCTURE**
1. **Original Data:** [Insert the verbatim input section]
2. **Analysis Report:** Provide a detailed breakdown of legal implications, potential loopholes, and compliance status based on insights from RAG and WebSearch.
    """
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
        system_prompt="""
### SYSTEM PROMPT
**TASK**
Synthesize the provided individual section analyses into a comprehensive Final Legal Review Report.

**Web Search**
  example: tavily_search(query: str, max_results=3–10), without using top_n or any other param.

**EXECUTION INSTRUCTIONS**
For *each* section provided in the input, you must generate a structured assessment containing the following five components:

1.  **Original Text:** Display the exact original clause/data.
2.  **Risk Assessment:**
    - **Risk Score:** Assign a score from 1 (Safe) to 10 (Critical Risk).
    - **Justification:** Briefly explain *why* this score was assigned.
3.  Detailed Analysis:** Synthesize the legal implications, referencing the provided analyses.
4.  **Strategic Suggestions:** Provide specific, actionable advice on how to mitigate the identified risks.
5.  **Refined Clause (Redline):** Rewrite the section to be legally watertight, protecting the user's interest while maintaining the original intent.

**FINAL SUMMARY**
At the end of the report, provide an **Overall Document Risk Score** (Average) and a concluding recommendation (e.g., "Safe to Sign," "Negotiation Required," "Do Not Sign").
    """
    )
    
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": f"Combine these {len(completed_sections)} analyses into a comprehensive report:\n\n{combined_analysis}"}]
    })
    
    final_report = result["messages"][-1].content if result["messages"] else combined_analysis
    return {"final_report": final_report}


async def build_graph():
    """Build and compile the LangGraph graph."""
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
        {
            "chunk_document": "chunk_document",
            "answer_from_cache": "answer_from_cache",
            "END": END
        }
    )
    
    builder.add_edge("answer_from_cache", END)

    builder.add_conditional_edges(
        "chunk_document",
        assign_workers,
        ["analyze_chunk"]
    )
    
    builder.add_edge("analyze_chunk", "synthesizer")
    builder.add_edge("synthesizer", END)

    checkpointer = MemorySaver()
    GRAPH = builder.compile(checkpointer=checkpointer)
    logger.info("Graph compiled successfully")


def get_graph():
    """Get the compiled graph."""
    return GRAPH


async def stream_graph_updates(input_state: State, config: dict, stream_mode: str = "updates"):
    """
    Stream updates from the graph as they occur.
    
    Args:
        input_state: Initial state for the graph
        config: Configuration dict with thread_id for checkpointing
        stream_mode: One of "updates", "values", or "messages"
        
    Yields:
        Streamed chunks from the graph
    """
    if GRAPH is None:
        await build_graph()
    
    async for chunk in GRAPH.astream(input_state, config, stream_mode=stream_mode):
        yield chunk
