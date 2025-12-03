# from typing import List, Dict, Optional, TypedDict, Annotated
# import operator
# from langchain.agents import create_agent
# from langchain_openai import ChatOpenAI
# from llama_cloud_services import LlamaParse

# # ------------------------------
# # Define State
# # ------------------------------

# class Chunk(TypedDict):
#     page_range: str
#     content: str
#     tag: Optional[str]

# class LegalState(TypedDict):
#     user_query: str
#     document_path: Optional[str]                  # Path if user uploaded a document
#     parsed_pages: Optional[List[Dict]]           # [{"page": int, "text": str}]
#     intent: Optional[str]                         # Detected intent
#     chunks: Optional[List[Chunk]]                 # List of chunked pages
#     worker_outputs: Optional[Annotated[List[Dict], operator.add]]  # Parallel-safe
#     final_answer: Optional[str]                   # Synthesized output
#     next: Optional[str]                           # "worker_phase" | "stop"

# # ------------------------------
# # Build Orchestrator Agent
# # ------------------------------

# llm = ChatOpenAI(model="gpt-5")

# orchestrator_agent = create_agent(
#     llm=llm,
#     tools=[],  # Add RAG or other tools later if needed
#     system_prompt="""
# You are the ORCHESTRATOR AGENT for a legal-document analysis workflow.

# Responsibilities:

# 1. Understand user intent (review, risk scoring, drafting, general query, summarization, rewrite)
# 2. If parsed_pages is empty, answer query directly (no chunking)
# 3. If document exists, perform intelligent chunking (2 pages per chunk)
# 4. Output JSON strictly as:
# {
#     "intent": "<detected intent>",
#     "chunks": [ ...chunk list... ],
#     "next": "worker_phase"
# }
# """
# )

# # ------------------------------
# # Orchestrator Node
# # ------------------------------

# def orchestrator_node(state: LegalState) -> LegalState:
#     user_query = state.get("user_query", "")
#     document_path = state.get("document_path", None)

#     if document_path:
#         # --- Document uploaded ---
#         parser = LlamaParse(
#             api_key="llx-oECNsOjRTaimHRD4iBjqHiASRCnnDPj2vLlCNgqejvNIgXbz",
#             num_workers=4,
#             verbose=True,
#             language="en",
#         )

#         # Parse the document
#         result = parser.parse(document_path)
#         parsed_pages = [{"page": p.page, "text": p.text} for p in result.pages]
#         state["parsed_pages"] = parsed_pages

#         # Call orchestrator agent to detect intent and chunk pages
#         messages = [
#             {
#                 "role": "user",
#                 "content": f"User Query:\n{user_query}\nParsed Pages:\n{parsed_pages}"
#             }
#         ]
#         response = orchestrator_agent.invoke(messages)

#         # Update state with agent response
#         state["intent"] = response.get("intent")
#         state["chunks"] = response.get("chunks", [])
#         state["next"] = response.get("next", "worker_phase")
#         state["final_answer"] = response.get("final_answer", None)

#     else:
#         # --- No document uploaded, direct query ---
#         messages = [{"role": "user", "content": user_query}]
#         response = orchestrator_agent.invoke(messages)

#         state["intent"] = response.get("intent", "general_query")
#         state["chunks"] = []
#         state["next"] = "stop"
#         state["final_answer"] = response.get("final_answer", response.get("output", ""))

#     return state

# # ------------------------------
# # Worker State
# # ------------------------------

# class WorkerState(TypedDict):
#     chunk: Chunk
#     intent: str
#     worker_outputs: Optional[Annotated[List[Dict], operator.add]]  # Parallel-safe

# # ------------------------------
# # Initialize Agent for Worker Node
# # ------------------------------

# # Define your tools here
# tools = [
#     # Tool(name="Search", func=lambda query: f"Search results for {query}", description="Searches relevant laws"),
# ]

# worker_agent = create_agent(
#     llm=llm,
#     tools=tools,
#     system_prompt="""
# You are a WORKER AGENT for legal-document analysis.

# Responsibilities:
# 1. Read the given chunk content carefully.
# 2. Process the chunk based on the intent:
#    - "review": provide detailed line-by-line review
#    - "risk_score": assess legal risks and highlight issues
#    - "draft": suggest drafting improvements or missing clauses
#    - "general_query": summarize or answer questions
# 3. Use available tools if needed (search, calculation, references).
# 4. Always preserve original content, never hallucinate.
# 5. Output strictly as:
# {
#     "chunk_range": "<page_range>",
#     "output": "<analysis or answer>"
# }
# """
# )

# # ------------------------------
# # Worker Node Function
# # ------------------------------

# def worker_node(state: WorkerState, agent=worker_agent) -> WorkerState:
#     """
#     Process a single chunk using the agent and tools.
#     Append output to worker_outputs safely in parallel.
#     """

#     chunk = state["chunk"]
#     intent = state["intent"]
#     worker_outputs = state.get("worker_outputs", [])

#     # Prepare agent input
#     input_text = f"""
# Chunk Range: {chunk['page_range']}
# Tag: {chunk.get('tag', '')}
# Content: {chunk['content']}

# Intent: {intent}
# Process the chunk accordingly and output JSON.
# """

#     # Call agent with tools
#     response = agent.run(input=input_text)

#     # Append response to worker_outputs
#     worker_outputs.append({
#         "chunk_range": chunk["page_range"],
#         "output": response
#     })

#     state["worker_outputs"] = worker_outputs
#     return state

# # ------------------------------
# # Synthesizer State
# # ------------------------------

# class SynthesizerState(TypedDict):
#     worker_outputs: List[Dict]       # [{"chunk_range": ..., "output": ...}]
#     intent: str                      # Original intent from orchestrator
#     final_answer: Optional[str]      # Output from synthesizer

# # ------------------------------
# # Initialize Synthesizer Agent
# # ------------------------------

# synthesizer_agent = create_agent(
#     llm=llm,
#     tools=[],
#     system_prompt="""
# You are a SYNTHESIZER AGENT for legal-document analysis.

# Responsibilities:
# 1. Receive processed chunk outputs from worker agents.
# 2. Adapt dynamically to the user intent provided in the input.
#    - Analyze the worker outputs according to the intent.
#    - Ensure output matches the user’s goal (review, draft, risk scoring, query, etc.).
# 3. Preserve original content integrity; do not hallucinate.
# 4. Output strictly as:
# {
#     "final_answer": "<synthesized answer based on all chunks and intent>"
# }
# """
# )

# # ------------------------------
# # Synthesizer Node Function
# # ------------------------------

# def synthesizer_node(state: SynthesizerState, agent=synthesizer_agent) -> SynthesizerState:
#     """
#     Synthesize a final answer based on worker outputs and dynamically detected intent.
#     """

#     worker_outputs = state["worker_outputs"]
#     intent = state["intent"]

#     # Prepare input for agent
#     input_text = f"""
# Worker Outputs:
# {worker_outputs}

# User Intent:
# {intent}

# Instructions:
# - Adapt your output dynamically to the intent provided above.
# - Preserve all original content, do not hallucinate.
# - If intent relates to review, draft, risk scoring, or query, act accordingly.
# - Output strictly in JSON format: {{"final_answer": "<your synthesized answer>"}}
# """

#     response = agent.run(input=input_text)
#     state["final_answer"] = response
#     return state



# GROQ_API_KEY="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"

# GROQ_API_KEY="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"



from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from pinecone import Pinecone
from langchain_groq import ChatGroq

model = ChatGroq(
    model_name="openai/gpt-oss-120b",
    api_key="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"
)

@tool
def retrieve_relevant_clauses(query: str):
    """
    RAG tool to retreive relevant legal cluases, acts, laws.
    Returns top 30 reranked results.
    """

    pc = Pinecone(api_key="pcsk_2wFprz_FWmjj2zwhCuAZNqheBR4mtP8FU7VUggLqQQUwZhJaWFMcCK2NXaSC5h26LZzVap")

    dense_index = pc.Index(host="https://safeclause-dc5puwa.svc.aped-4627-b74a.pinecone.io")
    sparse_index = pc.Index(host="https://safeclause-sparse-dc5puwa.svc.aped-4627-b74a.pinecone.io")

    # Get results from both indexes
    dense_results = dense_index.search(
        namespace="Acts_and_Clause_Namespace",
        query={
            "top_k": 45,  
            "inputs": {
                "text": query
            }
        }
    )

    sparse_results = sparse_index.search(
        namespace="Acts_and_Clause_Namespace",
        query={
            "top_k": 45,  
            "inputs": {
                "text": query
            }
        }
    )

    # Combine and deduplicate results
    combined_results = {}
    for match in dense_results['result']['hits']:
        combined_results[match['_id']] = match

    for match in sparse_results['result']['hits']:
        if match['_id'] not in combined_results:
            combined_results[match['_id']] = match

    # Prepare documents for reranking
    documents = []
    for match in combined_results.values():
        documents.append({
            "id": match['_id'],
            "text": match['fields']['text'],
            "act_name": match['fields']['act_name']  
        })

    # Apply reranking 
    reranked = pc.inference.rerank(
        model="pinecone-rerank-v0",
        query=query,
        documents=documents,
        rank_fields=["text"], 
        top_n=30,
        return_documents=True
    )

    return {"output" :reranked}

agent = create_agent(model, tools = [retrieve_relevant_clauses])


result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain pf related acts"}]}
)
print(result["messages"][-1].content)
