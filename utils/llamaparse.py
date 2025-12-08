# utils/document_parser.py
import os
import logging
from llama_cloud_services import LlamaParse
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY")


async def document_parser(file_path: str) -> str:
    """Helper function to parse documents using LlamaParse."""
    logger.info(f"📄 LLAMAPARSE CALLED with file: {file_path}")
    
    try:
        parser = LlamaParse(
            api_key=LLAMAPARSE_API_KEY,
            num_workers=4,
            verbose=False,
            language="en"
        )
        
        result = parser.parse(file_path)
        
        # Combine all pages into one text
        all_text = ""
        for page_object in result.pages:
            all_text += f"\n\n=== Page {page_object.page} ===\n{page_object.text.strip()}\n"
        
        logger.info(f"✅ LLAMAPARSE RETURNED {len(result.pages)} pages, {len(all_text)} characters")
        
        return all_text
    except Exception as e:
        logger.error(f"❌ LLAMAPARSE ERROR: {str(e)}", exc_info=True)
        raise