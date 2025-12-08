from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_2wFprz_FWmjj2zwhCuAZNqheBR4mtP8FU7VUggLqQQUwZhJaWFMcCK2NXaSC5h26LZzVap")

dense_index = pc.Index(host="https://safeclause-dc5puwa.svc.aped-4627-b74a.pinecone.io")
sparse_index = pc.Index(host="https://safeclause-sparse-dc5puwa.svc.aped-4627-b74a.pinecone.io")

query = "emploment contract termination clause"

# Get results from both indexes
dense_results = dense_index.search(
    namespace="Acts_and_Clause_Namespace",
    query={
        "top_k": 60,  
        "inputs": {
            "text": query
        }
    }
)

sparse_results = sparse_index.search(
    namespace="Acts_and_Clause_Namespace",
    query={
        "top_k": 60,  
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

print(reranked)

