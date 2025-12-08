import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.tools import tool
from pinecone import Pinecone
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model_name="gpt-5-mini",
    api_key="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"
)

@tool
def retrieve_relevant_clauses(query: str):
    """Search internal legal database for acts and clauses."""
    try:
        pc = Pinecone(api_key="pcsk_2wFprz_FWmjj2zwhCuAZNqheBR4mtP8FU7VUggLqQQUwZhJaWFMcCK2NXaSC5h26LZzVap")
        dense_index = pc.Index(host="https://safeclause-dc5puwa.svc.aped-4627-b74a.pinecone.io")
        sparse_index = pc.Index(host="https://safeclause-sparse-dc5puwa.svc.aped-4627-b74a.pinecone.io")

        dense_results = dense_index.search(
            namespace="Acts_and_Clause_Namespace",
            query={"top_k": 45, "inputs": {"text": query}}
        )

        sparse_results = sparse_index.search(
            namespace="Acts_and_Clause_Namespace",
            query={"top_k": 45, "inputs": {"text": query}}
        )

        combined_results = {}
        for match in dense_results['result']['hits']:
            combined_results[match['_id']] = match

        for match in sparse_results['result']['hits']:
            if match['_id'] not in combined_results:
                combined_results[match['_id']] = match

        documents = []
        for match in combined_results.values():
            documents.append({
                "id": match['_id'],
                "text": match['fields']['text'],
                "act_name": match['fields']['act_name']  
            })

        reranked = pc.inference.rerank(
            model="pinecone-rerank-v0",
            query=query,
            documents=documents,
            rank_fields=["text"], 
            top_n=30,
            return_documents=True
        )

        return str(reranked)
    except Exception as e:
        return str(e)


async def setup_mcp_client():
    """Connect to Tavily MCP server."""
    client = MultiServerMCPClient({
        "tavily": {
            "transport": "streamable_http",
            "url": "https://mcp.tavily.com/mcp/?tavilyApiKey=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",
        }
    })
    
    mcp_tools = await client.get_tools()
    return [t for t in mcp_tools if t.name in ["tavily_search", "tavily_extract"]]


async def main():
    mcp_tools = await setup_mcp_client()
    
    all_tools = [retrieve_relevant_clauses] + mcp_tools
    
    agent = create_agent(
        model,
        tools=all_tools,
        system_prompt="You are a legal research assistant. Use retrieve_relevant_clauses for internal database and tavily_search for web search."
    )
    
    query = "Who is the founder of kroolo"
    result = await agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": query
        }]
    })
    
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())