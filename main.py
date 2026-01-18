import logging
import json
import asyncio
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from langchain.messages import HumanMessage
import uvicorn
import os
import tempfile
from pathlib import Path

from agent import build_graph, get_graph, stream_graph_updates, State
from utils.storage_service import upload_to_storj

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NODE_MESSAGES = {
    "download_and_parse_document": "🧾 Generating results...",
    "orchestration_agent": "🧠 Analyzing query intent & legal context...",
    "retrieve_relevant_documents": "🔍 Searching legal databases & case laws...",
    "chunk_document": "📄 Splitting documents into sections...",
    "analyze_chunk": "⚖️ Reviewing individual clauses for compliance...",
    "synthesizer": "✍️ Drafting final legal assessment...",
    "answer_from_cache": "⚡ Retrieving analysis from memory...",
}

class QueryRequest(BaseModel):
    query: str = Field(..., description="User's legal query")
    doc_filepaths: list[str] = Field(default=[], description="List of S3 document URLs")
    session_id: str = Field(default="default_session", description="Session ID")


class HealthResponse(BaseModel):
    status: str
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize graph on startup."""
    await build_graph()
    logger.info("Graph initialized on startup")
    yield


app = FastAPI(
    title="Legal Document Analysis API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", message="Legal Document Analysis API is running")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file to Storj storage and return the URL.
    """
    try:
        logger.info(f"Uploading file: {file.filename}")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            # Write uploaded content to temp file
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Upload to Storj with original filename
            success, result = upload_to_storj(tmp_file_path, file.filename)
            
            if success:
                logger.info(f"Successfully uploaded {file.filename}")
                return {
                    "success": True,
                    "filename": file.filename,
                    "url": result,
                    "message": f"Successfully uploaded {file.filename}"
                }
            else:
                logger.error(f"Failed to upload {file.filename}: {result}")
                return {
                    "success": False,
                    "filename": file.filename,
                    "error": result,
                    "message": f"Failed to upload {file.filename}"
                }
        
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")
    
    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}", exc_info=True)
        return {
            "success": False,
            "filename": file.filename if file else "unknown",
            "error": str(e),
            "message": "Upload failed due to server error"
        }


@app.post("/query")
async def query(request: QueryRequest):
    """
    Stream legal query analysis in real-time.
    """
    try:
        GRAPH = get_graph()
        if not GRAPH:
            raise HTTPException(status_code=500, detail="Graph not initialized")
        
        session_id = request.session_id
        config = {"configurable": {"thread_id": session_id}}
        
        # Retrieve previous state
        previous_state = GRAPH.get_state(config)
        
        if previous_state and previous_state.values:
            previous_messages = previous_state.values.get("messages", [])
            previous_doc_contents = previous_state.values.get("doc_contents", [])
            previous_chunks = previous_state.values.get("chunks", [])
            previous_completed_sections = previous_state.values.get("completed_sections", [])
        else:
            previous_messages = []
            previous_doc_contents = []
            previous_chunks = []
            previous_completed_sections = []
        
        new_user_message = HumanMessage(content=request.query)
        
        if len(previous_messages) > 3:
            previous_messages = previous_messages[-3:]
        
        all_messages = previous_messages + [new_user_message]
        
        input_state: State = {
            "query": request.query,
            "session_id": session_id,
            "doc_filepaths": request.doc_filepaths,
            "doc_contents": previous_doc_contents,
            "has_documents": len(previous_doc_contents) > 0,
            "chunks": previous_chunks,
            "completed_sections": previous_completed_sections,
            "final_report": "",
            "messages": all_messages,
        }
        
        return await stream_query_response(input_state, config)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /query endpoint: {str(e)}", exc_info=True)
        async def error_generator():
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"
        return StreamingResponse(error_generator(), media_type="application/x-ndjson")


async def stream_query_response(input_state: State, config: dict):
    """
    Stream graph updates using astream with stream_mode="updates".
    """
    async def event_generator():
        try:
            node_count = 0
            final_report = ""
            
            async for chunk in stream_graph_updates(
                input_state, 
                config, 
                stream_mode="updates"
            ):
                node_count += 1
                
                for node_name, state_updates in chunk.items():
                    logger.info(f"Node {node_count}: {node_name}")

                    friendly_message = NODE_MESSAGES.get(node_name, f"Processing step: {node_name}...")
                    
                    yield json.dumps({
                        "type": "progress", 
                        "node": node_name,
                        "content": friendly_message
                    }) + "\n"


                    if "final_report" in state_updates and state_updates["final_report"]:
                        final_report = state_updates["final_report"]
                        yield json.dumps({
                            "type": "report",
                            "node": node_name,
                            "content": final_report
                        }) + "\n"
            
            yield json.dumps({
                "type": "complete",
                "message": "Query processing completed",
                "total_nodes": node_count
            }) + "\n"
        
        except Exception as e:
            logger.error(f"Error in stream_query_response: {str(e)}", exc_info=True)
            yield json.dumps({
                "type": "error",
                "error": str(e)
            }) + "\n"
    
    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
