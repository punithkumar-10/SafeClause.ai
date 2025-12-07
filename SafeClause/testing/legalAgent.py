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
import requests
from requests_aws4auth import AWS4Auth
from llama_cloud_services import LlamaParse
import os


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


# NODE 1: Download and parse document
async def download_and_parse_document(state: State):
    """Download file from Storj, parse it with LlamaParse, and store content"""
    
    # Check if document is already in state memory
    if state.get("doc_content") and len(state["doc_content"].strip()) > 0:
        # Document already parsed, skip download and parsing
        return {
            "has_document": True,
            "messages": [HumanMessage(content="Document already in memory, skipping download")],
        }
    
    doc_filepath = state.get("doc_filepath")
    
    if not doc_filepath:
        return {
            "doc_content": "",
            "has_document": False,
            "messages": [HumanMessage(content="No document filepath provided")],
        }
    
    try:
        # Storj credentials
        access_key = 'jxv52ooceheejwc2njollbeo7gea'
        secret_key = 'jyfrizf7g7vekzje7f4wwxa5frdplzbqkt3dao5nabvsbvaxqm46w'
        
        # Create signed request
        auth = AWS4Auth(access_key, secret_key, 'us-east-1', 's3')
        
        # Download file from Storj
        response = requests.get(doc_filepath, auth=auth)
        
        if response.status_code != 200:
            return {
                "doc_content": "",
                "has_document": False,
                "messages": [HumanMessage(content=f"Failed to download document: {response.status_code}")],
            }
        
        # Extract filename from URL
        filename = doc_filepath.split('/')[-1]
        temp_path = f"/tmp/{filename}"
        
        # Save to temporary file
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        
        try:
            # Parse with LlamaParse
            parser = LlamaParse(
                api_key="llx-oECNsOjRTaimHRD4iBjqHiASRCnnDPj2vLlCNgqejvNIgXbz",
                num_workers=4,
                verbose=True,
                language="en",
            )
            
            # Parse the document
            result = parser.parse(temp_path)
            
            # Extract all text content
            all_text = ""
            for page_object in result.pages:
                page_number = page_object.page
                page_text = page_object.text.strip()
                all_text += f"\n\n=== Page {page_number} ===\n{page_text}\n"
            
            return {
                "doc_content": all_text,
                "has_document": True if all_text.strip() else False,
                "messages": [HumanMessage(content="Document downloaded and parsed successfully")],
            }
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        return {
            "doc_content": "",
            "has_document": False,
            "messages": [HumanMessage(content=f"Error downloading/parsing document: {str(e)}")],
        }


# NODE 2: Orchestration agent
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


# ROUTING FUNCTION: After orchestration agent
async def route_after_orchestration(state: State) -> Literal["chunk_document", "answer_from_cache", "END"]:
    """Route based on whether document content is available"""
    
    # Check if we have document content
    has_document = bool(state.get("doc_content") and len(state["doc_content"].strip()) > 0)
    
    if not has_document:
        # No document, end the flow
        return "END"
    
    if state.get("completed_sections") and len(state["completed_sections"]) > 0:
        # Document already analyzed, use cached results
        return "answer_from_cache"
    
    # Document available but not yet analyzed, chunk it
    return "chunk_document"


# NODE 3: Answer from cache
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


# NODE 4: Chunk document
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


# NODE 5: Analyze chunk (worker node for parallel execution)
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


# ROUTING FUNCTION: Assign workers
async def assign_workers(state: State):
    """Dispatch chunks to parallel workers"""
    return [Send("analyze_chunk", {"section": chunk}) for chunk in state["chunks"]]


# NODE 6: Synthesizer
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


# BUILD GRAPH
async def build_graph():
    """Build the LangGraph workflow"""
    
    global MCP_TOOLS
    
    # Setup MCP tools
    MCP_TOOLS = await setup_mcp_client()
    
    builder = StateGraph(State)

    # Add all nodes
    builder.add_node("download_and_parse_document", download_and_parse_document)
    builder.add_node("orchestration_agent", orchestration_agent)
    builder.add_node("answer_from_cache", answer_from_cache)
    builder.add_node("chunk_document", chunk_document)
    builder.add_node("analyze_chunk", analyze_chunk)
    builder.add_node("synthesizer", synthesizer)

    # Define edges
    builder.add_edge(START, "download_and_parse_document")
    
    # After downloading/parsing, go to orchestration agent
    builder.add_edge("download_and_parse_document", "orchestration_agent")

    # Conditional edges from orchestration_agent
    builder.add_conditional_edges(
        "orchestration_agent",
        route_after_orchestration,
        {
            "chunk_document": "chunk_document",
            "answer_from_cache": "answer_from_cache",
            "END": END,
        }
    )

    # answer_from_cache ends the graph
    builder.add_edge("answer_from_cache", END)

    # chunk_document dispatches to analyze_chunk workers in parallel
    builder.add_conditional_edges(
        "chunk_document",
        assign_workers,
        ["analyze_chunk"]
    )

    # analyze_chunk goes to synthesizer
    builder.add_edge("analyze_chunk", "synthesizer")
    
    # synthesizer ends the graph
    builder.add_edge("synthesizer", END)

    # Compile with checkpointer for state persistence
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# MAIN EXECUTION
async def main():
    """Main execution function"""
    
    # Build the graph
    orchestrator_worker = await build_graph()
    
    # Configuration for session persistence
    config = {"configurable": {"thread_id": "session_1"}}

    # Example: First request - downloads and analyzes document
    print("=" * 80)
    print("FIRST REQUEST: Download and analyze document")
    print("=" * 80)
    
    first_input = {
        "query": "Analyze the legal document for risks and compliance issues",
        "doc_filepath": "https://gateway.storjshare.io/safeclause-ai/demo1-user/lawsimpl-document.pdf",
        "doc_content": "",
        "has_document": False,
        "chunks": [],
        "completed_sections": [],
        "final_report": "",
        "messages": []
    }

    # Run the graph
    result1 = await orchestrator_worker.ainvoke(first_input, config)
    
    print("\nFirst Request - Final Report:")
    print(result1.get("final_report", "No report generated"))
    print("\nDocument Content Length:", len(result1.get("doc_content", "")))
    print("Completed Sections:", len(result1.get("completed_sections", [])))

    # Example: Second request - uses cached document
    print("\n" + "=" * 80)
    print("SECOND REQUEST: Same document, different question (uses cached content)")
    print("=" * 80)
    
    second_input = {
        "query": "What are the key clauses and their implications?",
        "doc_filepath": "https://gateway.storjshare.io/safeclause-ai/demo1-user/lawsimpl-document.pdf",
        "doc_content": result1.get("doc_content", ""),  # Reuse from first request
        "has_document": result1.get("has_document", False),
        "chunks": result1.get("chunks", []),
        "completed_sections": result1.get("completed_sections", []),
        "final_report": result1.get("final_report", ""),
        "messages": result1.get("messages", [])
    }

    # Run the graph again - will skip download since doc_content exists
    result2 = await orchestrator_worker.ainvoke(second_input, config)
    
    print("\nSecond Request - Final Report:")
    print(result2.get("final_report", "No report generated"))
    print("\nMessage from download node:", result2["messages"][0].content if result2["messages"] else "No message")


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())