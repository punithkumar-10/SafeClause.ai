from typing import Annotated, List
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


orchestration_model = ChatOpenAI(
    model_name="gpt-5-mini",
    api_key="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"
)

model = ChatOpenAI(
    model_name="gpt-5-nano",
    api_key="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"
)

MCP_TOOLS = []
MCP_CLIENT = None
PINECONE_CLIENT = None
TEXT_SPLITTER = None

def get_text_splitter():
    """Lazy load and cache text splitter"""
    global TEXT_SPLITTER
    if TEXT_SPLITTER is None:
        TEXT_SPLITTER = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=10000,
            chunk_overlap=200,
        )
    return TEXT_SPLITTER

def get_pinecone_client():
    """Get or create global Pinecone client"""
    global PINECONE_CLIENT
    if PINECONE_CLIENT is None:
        PINECONE_CLIENT = Pinecone(api_key="pcsk_2wFprz_FWmjj2zwhCuAZNqheBR4mtP8FU7VUggLqQQUwZhJaWFMcCK2NXaSC5h26LZzVap")
    return PINECONE_CLIENT


@tool
async def retrieve_relevant_documents(query: str):
    """Search internal database for relevant documents using dense and sparse search."""
    
    pc = get_pinecone_client()
    dense_index = pc.Index(host="https://safeclause-dc5puwa.svc.aped-4627-b74a.pinecone.io")
    sparse_index = pc.Index(host="https://safeclause-sparse-dc5puwa.svc.aped-4627-b74a.pinecone.io")

    dense_task = asyncio.create_task(asyncio.to_thread(
        lambda: dense_index.search(
            namespace="Acts_and_Clause_Namespace",
            query={"top_k": 35, "inputs": {"text": query}}
        )
    ))
    
    sparse_task = asyncio.create_task(asyncio.to_thread(
        lambda: sparse_index.search(
            namespace="Acts_and_Clause_Namespace",
            query={"top_k": 35, "inputs": {"text": query}}
        )
    ))
    
    dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

    combined_results = {}
    for match in dense_results['result']['hits']:
        combined_results[match['_id']] = match

    for match in sparse_results['result']['hits']:
        if match['_id'] not in combined_results:
            combined_results[match['_id']] = match

    documents = [
        {
            "id": match['_id'],
            "text": match['fields']['text'],
            "act_name": match['fields']['act_name']  
        }
        for match in combined_results.values()
    ]

    reranked = pc.inference.rerank(
        model="pinecone-rerank-v0",
        query=query,
        documents=documents,
        rank_fields=["text"], 
        top_n=30,
        return_documents=True
    )

    return str(reranked)


async def setup_mcp_client():
    """Connect to Tavily MCP server once and reuse."""
    global MCP_CLIENT
    
    if MCP_CLIENT is None:
        MCP_CLIENT = MultiServerMCPClient({
            "tavily": {
                "transport": "streamable_http",
                "url": "https://mcp.tavily.com/mcp/?tavilyApiKey=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",
            }
        })
        mcp_tools = await MCP_CLIENT.get_tools()
        return [t for t in mcp_tools if t.name in ["tavily_search", "tavily_extract"]]
    
    return []

class Section(BaseModel):
    name: str = Field(description="Section name")
    content: str = Field(description="Chunked content")
    chunk_index: int = Field(description="Chunk index")

class State(TypedDict):
    query: str
    doc_content: str
    has_document: bool
    chunks: list[Section]
    completed_sections: Annotated[list, operator.add]
    final_report: str
    messages: Annotated[list, operator.add]

class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]



async def orchestration_agent(state: State):
    """Orchestration agent with RAG and web search tools"""
    
    has_document = bool(state.get("doc_content") and len(state["doc_content"].strip()) > 0)
    
    if has_document:
        return {
            "has_document": True,
            "messages": [HumanMessage(content=f"Document detected. Processing query: {state['query']}")],
        }
    
    all_tools = [retrieve_relevant_documents]
    if MCP_TOOLS:
        all_tools.extend(MCP_TOOLS)
    
    agent = create_agent(
        orchestration_model,
        tools=all_tools,
        system_prompt="""You are an expert Indian Legal AI Agent. Provide accurate legal guidance based on the Indian Constitution, Acts, and Judicial Precedents.

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
4. **Disclaimer:** "Information is for educational purposes and not professional legal advice."
"""
    )
    
    result = await agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": state['query']
        }]
    })
    
    final_answer = result["messages"][-1].content if result["messages"] else "No answer generated"
    
    return {
        "has_document": False,
        "final_report": final_answer,
        "messages": [HumanMessage(content=final_answer)],
    }

async def route_after_orchestration(state: State) -> Literal["chunk_document", "answer_from_cache", "END"]:
    """Route based on whether document is PROCESSED"""
    
    if not state.get("has_document"):
        return "END"
    
    if state.get("completed_sections") and len(state["completed_sections"]) > 0:
        return "answer_from_cache"
    
    return "chunk_document"

async def answer_from_cache(state: State):
    """Use cached results with fresh agent"""
    
    cached_sections = state.get("completed_sections", [])
    combined_context = "\n\n---\n\n".join(cached_sections)
    
    all_tools = [retrieve_relevant_documents]
    if MCP_TOOLS:
        all_tools.extend(MCP_TOOLS)
    
    agent = create_agent(
        model,
        tools=all_tools,
        system_prompt="""You are an expert Indian Legal AI Agent. You provide accurate legal guidance based on the Indian Constitution, Acts, and Judicial Precedents.
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
4. **Disclaimer:** "Information is for educational purposes and not professional legal advice."
"""
    )
    
    result = await agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": f"Document Analysis:\n{combined_context}\n\nUser Question: {state['query']}"
        }]
    })
    
    final_answer = result["messages"][-1].content if result["messages"] else "No answer generated"
    
    return {
        "final_report": final_answer,
        "messages": [HumanMessage(content=final_answer)],
    }

async def chunk_document(state: State):
    """Split document into chunks"""
    
    text_splitter = get_text_splitter()
    splits = text_splitter.split_text(state["doc_content"])
    
    chunks = [
        Section(
            name=f"Section {i+1}",
            content=splits[i],
            chunk_index=i
        )
        for i in range(len(splits))
    ]
    
    return {"chunks": chunks}

async def analyze_chunk(state: WorkerState):
    """Analyze chunk - fresh agent for each chunk"""
    
    all_tools = [retrieve_relevant_documents]
    if MCP_TOOLS:
        all_tools.extend(MCP_TOOLS)
    
    agent = create_agent(
        model,
        tools=all_tools,
        system_prompt="""**ROLE**
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
        "messages": [{
            "role": "user",
            "content": f"Section {state['section'].chunk_index}:\n{state['section'].content}\n\nProvide a detailed analysis of this section."
        }]
    })
    
    analysis = result["messages"][-1].content if result["messages"] else "No analysis generated"
    
    return {"completed_sections": [analysis]}

async def assign_workers(state: State):
    """Dispatch chunks to parallel workers"""
    return [Send("analyze_chunk", {"section": chunk}) for chunk in state["chunks"]]

async def synthesizer(state: State):
    """Combine all analyses - fresh agent"""
    
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
        "messages": [{
            "role": "user",
            "content": f"Combine these {len(completed_sections)} section analyses into a comprehensive report:\n\n{combined_analysis}"
        }]
    })
    
    final_report = result["messages"][-1].content if result["messages"] else combined_analysis
    
    return {"final_report": final_report}


async def build_graph():
    """Build the graph"""
    
    global MCP_TOOLS
    
    MCP_TOOLS = await setup_mcp_client()
    
    builder = StateGraph(State)

    builder.add_node("orchestration_agent", orchestration_agent)
    builder.add_node("answer_from_cache", answer_from_cache)
    builder.add_node("chunk_document", chunk_document)
    builder.add_node("analyze_chunk", analyze_chunk)
    builder.add_node("synthesizer", synthesizer)

    builder.add_edge(START, "orchestration_agent")

    builder.add_conditional_edges(
        "orchestration_agent",
        route_after_orchestration,
        {
            "chunk_document": "chunk_document",
            "answer_from_cache": "answer_from_cache",
            "END": END,
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
    return builder.compile(checkpointer=checkpointer)



orchestrator_worker = asyncio.run(build_graph())
config = {"configurable": {"thread_id": "session_1"}}