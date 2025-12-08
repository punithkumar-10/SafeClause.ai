# import asyncio
# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langchain.agents import create_agent
# from langchain_groq import ChatGroq

# async def main():
    
#     # Configure the Tavily MCP server connection
#     client = MultiServerMCPClient({
#         "tavily": {
#             "transport": "streamable_http",
#             # Use the Tavily MCP endpoint with your API key
#             "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey=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",
#         }
#     })

#     # Get all available tools from tavily
#     tools = await client.get_tools()
    
#     model = ChatGroq(
#         model_name="openai/gpt-oss-120b",
#         api_key="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"
#     )
    
#     # Create an agent with the tavily tools
#     agent = create_agent(
#         model,  # or any LLM you prefer
#         tools=tools
#     )

#     # Use the agent to invoke Exa tools
#     response = await agent.ainvoke({
#         "messages": [{
#             "role": "user",
#             "content": "Search for information about artificial intelligence"
#         }]
#     })
    
#     print(response)

# if __name__ == "__main__":
#     asyncio.run(main())



import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq

model = ChatGroq(
    model_name="openai/gpt-oss-120b",
    api_key="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"
)

async def setup_mcp_client():
    """Connect to Tavily MCP server and get available tools."""
    client = MultiServerMCPClient({
        "tavily": {
            "transport": "streamable_http",
            "url": "https://mcp.tavily.com/mcp/?tavilyApiKey=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",
        }
    })
    
    mcp_tools = await client.get_tools()
    print(f"MCP Tools loaded: {[tool.name for tool in mcp_tools]}")
    
    # Print tool schemas to see what parameters they expect
    for tool in mcp_tools:
        print(f"\nTool: {tool.name}")
        print(f"Description: {tool.description}")
    
    return mcp_tools


def create_legal_agent(mcp_tools):
    """Create agent with only MCP tools for testing."""
    agent = create_agent(
        model,
        tools=mcp_tools,
        system_prompt="You are a legal research assistant. Search for information about employment laws."
    )
    
    return agent


async def execute_query(agent, query):
    """Execute a query using the legal research agent."""
    result = await agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": query
        }]
    })
    
    response_content = result["messages"][-1].content
    return response_content


async def main():
    print("Setting up MCP...")
    mcp_tools = await setup_mcp_client()
    
    print("\n" + "="*70)
    print("Creating agent...")
    agent = create_legal_agent(mcp_tools)
    
    print("Executing query...")
    query = "who is the founder of kroolo?"
    response = await execute_query(agent, query)
    
    print("\n" + "="*70)
    print("RESPONSE:")
    print("="*70)
    print(response)
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())