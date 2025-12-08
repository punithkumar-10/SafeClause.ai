import logging
import json
import asyncio
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from langchain.messages import HumanMessage

from agent import build_graph, get_graph, State

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str = Field(None, description="User's legal query")
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
    return HealthResponse(status="healthy", message="SafeClause.ai is running!")


@app.post("/query")
async def query(request: QueryRequest):
    try:
        GRAPH = get_graph()
        if not GRAPH:
            raise HTTPException(status_code=500, detail="Graph not initialized")
        
        session_id = request.session_id
        config = {"configurable": {"thread_id": session_id}}
        
        previous_state = GRAPH.get_state(config)
        
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
        
        new_user_message = HumanMessage(content=request.query)
        all_messages = previous_messages + [new_user_message]
        
        input_state = {
            "query": request.query,
            "session_id": session_id,
            "doc_filepath": request.doc_filepath or "",
            "doc_content": previous_doc_content,
            "has_document": bool(previous_doc_content.strip()) if previous_doc_content else False,
            "chunks": previous_chunks,
            "completed_sections": previous_completed_sections,
            "final_report": "",
            "messages": all_messages,
        }
        
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