import asyncio
import logging
import os
from functools import lru_cache
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

@lru_cache(maxsize=1)
def get_pinecone_client():
    """Initialize Pinecone client once and cache it."""
    return Pinecone(api_key=PINECONE_API_KEY)


async def retrieve_documents_helper(query: str):
    """Helper function that contains all RAG logic."""
    logger.info(f"🔍 RAG HELPER CALLED with query: {query[:100]}")
    
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