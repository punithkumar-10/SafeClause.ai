# agent.py
import logging
from typing import Annotated
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
import os
from dotenv import load_dotenv
from tools import retrieve_documents_helper
from utils import document_parser

load_dotenv()

logger = logging.getLogger(__name__)

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")


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


orchestration_model = ChatOpenAI(model_name="gpt-5-mini", api_key=OPENAI_API_KEY)
model = ChatOpenAI(model_name="gpt-5-nano", api_key=OPENAI_API_KEY)

MCP_TOOLS = []
MCP_CLIENT = None
TEXT_SPLITTER = None
GRAPH = None


def get_text_splitter():
    global TEXT_SPLITTER
    if TEXT_SPLITTER is None:
        TEXT_SPLITTER = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=10000, chunk_overlap=200)
    return TEXT_SPLITTER


async def setup_mcp_client():
    global MCP_CLIENT
    if MCP_CLIENT is None:
        MCP_CLIENT = MultiServerMCPClient({"tavily": {"transport": "streamable_http", "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"}})
        mcp_tools = await MCP_CLIENT.get_tools()
        return [t for t in mcp_tools if t.name in ["tavily_search", "tavily_extract"]]
    return []


@tool
async def retrieve_relevant_documents(query: str):
    """Search internal database for relevant documents using dense and sparse search."""
    return await retrieve_documents_helper(query)


async def download_and_parse_document(state: State):
    session_id = state.get("session_id", "default")
    
    # Check if documents already exist in state
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
        
        # Process each file
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
    
    all_tools = [retrieve_relevant_documents]
    if MCP_TOOLS:
        all_tools.extend(MCP_TOOLS)
    
    agent = create_agent(
        orchestration_model,
        tools=all_tools,
        system_prompt="""
You are an expert Indian Legal AI Agent providing accurate, practical guidance based on
the Indian Constitution, statutory law, and judicial precedents.

The name which i have to this chatbot is "SafeClause.ai"

──────── TOOL USAGE ────────
• VectorDB (RAG) — PRIMARY:
  Statutes, sections, definitions, and established case law.
• WebSearch — SECONDARY (only when required):
  Recent judgments, amendments, notifications, repeals.
• If sources conflict, prioritize the most recent verified information.

──────── CORE LEGAL RULES ────────
1. Old vs New Codes:
   If IPC, CrPC, or Evidence Act is referenced, ALWAYS provide corresponding sections
   under BNS, BNSS, and BSA.
2. Accuracy:
   Cite exact Acts and section numbers; avoid assumptions and flag uncertainty.
3. Tone:
   Professional, neutral, precise; plain English unless legal precision is required.

──────── INTELLIGENT QUESTIONING ────────
• Ask follow-up questions ONLY when:
  – Facts are essential to give a correct legal answer, or
  – Multiple legal outcomes depend on missing details.
• Never ask questions for:
  – Small talk, identity, confirmations, or obvious context.
• When asking, ask **one clear, specific question** and explain *why* it matters.

──────── CONTEXT & MEMORY AWARENESS ────────
• Respond naturally to greetings, names, and clarifications.
• Do NOT apply legal framing or disclaimers to non-legal queries.
• Switch to legal analysis mode ONLY when legal guidance is required.

──────── RESPONSE STRUCTURE (LEGAL QUERIES ONLY) ────────
1. Direct Answer — clear legal position.
2. Legal Provisions — Acts and sections
   (e.g., Section 302 IPC / Section 103 BNS).
3. Case Law — landmark or recent rulings (if relevant).
4. Practical Notes — procedures, remedies, compliance steps.
5. Disclaimer:
   “Information is for educational purposes and not professional legal advice.”

──────── IMPORTANT BEHAVIOR ────────
• Never over-lawyer simple questions.
• Never add disclaimers to non-legal responses.
• Prioritize user intent, clarity, and correctness.

"""
    )
    
    messages = state.get("messages", [])
    if not messages:
        messages = [{"role": "user", "content": state['query']}]
    else:
        messages.append({"role": "user", "content": state['query']})
    
    result = await agent.ainvoke({"messages": messages})
    final_answer = result["messages"][-1].content if result["messages"] else "No answer generated"
    
    return {
        "has_documents": False,
        "final_report": final_answer,
        "messages": [HumanMessage(content=final_answer)]
    }


async def route_after_orchestration(state: State) -> Literal["chunk_document", "answer_from_cache", "END"]:
    has_documents = state.get("has_documents", False)
    if not has_documents:
        return "END"
    if state.get("completed_sections") and len(state["completed_sections"]) > 0:
        return "answer_from_cache"
    return "chunk_document"

async def answer_from_cache(state: State):
    cached_sections = state.get("completed_sections", [])
    combined_context = "\n\n---\n\n".join(cached_sections)
    
    doc_info = "\n".join([f"- {doc['filename']}" for doc in state.get("doc_contents", [])])
    
    all_tools = [retrieve_relevant_documents]
    if MCP_TOOLS:
        all_tools.extend(MCP_TOOLS)
    
    agent = create_agent(
        model,
        tools=all_tools,
        system_prompt="""
You are an expert Indian Legal AI Agent. You provide accurate legal guidance based on the Indian Constitution, Acts, and Judicial Precedents.
Answer the user's question based on the provided document analysis. You can also use RAG and web search if needed.

**TOOL USAGE STRATEGY**
1. **VectorDB (RAG):** PRIMARY source for statutory text (Acts, Clauses), definitions, and historical case laws.
2. **WebSearch:** SECONDARY source. Use strictly to find:
   - Recent Supreme Court/High Court judgments (current year).
   - Latest amendments or government notifications.
   - Verification if a specific law has been repealed or updated.

**CORE INSTRUCTIONS**
1. **Old vs. New Law:** If a user asks about old codes (IPC, CrPC, Evidence Act), you MUST provide the corresponding sections in the new Sanhitas (BNS, BNSS, BSA) alongside the old ones.
2. **Fact-Checking:** If WebSearch results (e.g., a recent amendment) contradict VectorDB data, prioritize the WebSearch result.
3. **Tone:** Professional, neutral, and precise.

**RESPONSE FORMAT**
1. **Direct Answer:** A concise summary of the legal position.
2. **Legal Provisions:** Cite specific Acts and Sections (e.g., "Section 302 IPC / Section 103 BNS").
3. **Case Laws:** Mention relevant landmark cases (from VectorDB) and recent rulings (from WebSearch).
4. **Disclaimer:** "Information is for educational purposes and not professional legal advice.
    """
    )
    
    messages = state.get("messages", [])
    prompt_content = f"Documents analyzed:\n{doc_info}\n\nDocument Analysis:\n{combined_context}\n\nUser Question: {state['query']}"
    
    if not messages:
        messages = [{"role": "user", "content": prompt_content}]
    else:
        messages.append({"role": "user", "content": prompt_content})
    
    result = await agent.ainvoke({"messages": messages})
    final_answer = result["messages"][-1].content if result["messages"] else "No answer generated"
    
    return {
        "final_report": final_answer,
        "messages": [HumanMessage(content=final_answer)]
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

**EXECUTION INSTRUCTIONS**
For *each* section provided in the input, you must generate a structured assessment containing the following five components:

1.  **Original Text:** Display the exact original clause/data.
2.  **Risk Assessment:**
    - **Risk Score:** Assign a score from 1 (Safe) to 10 (Critical Risk).
    - **Justification:** Briefly explain *why* this score was assigned.
3.  **Detailed Analysis:** Synthesize the legal implications, referencing the provided analyses.
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


def get_graph():
    """Get the compiled graph."""
    return GRAPH