# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # Step 1: Load PDF
# loader = PyPDFLoader("SafeClause/Datasets/a1881-26.pdf")
# pages = loader.load()

# # Combine all pages into one string (optional)
# full_text = "\n".join([p.page_content for p in pages])

# # Step 2: Split the text
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=3000,
#     chunk_overlap=400
# )

# chunks = text_splitter.split_text(full_text)

# # Step 3: Print chunks
# for i, chunk in enumerate(chunks, start=1):
#     print(f"\n==== Chunk {i} ====\n")
#     print(chunk)




import os
import json
import uuid
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "SafeClause/Datasets/THE ARBITRATION AND CONCILIATION ACT, 1996.pdf"   # your single file
OUTPUT_DIR = "SafeClause/Chunks"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_RECORDS_PER_JSON = 96

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_text(filepath):
    """Extract text using PyPDFLoader"""
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    return "\n".join([p.page_content for p in pages])

def chunk_pdf(filepath):
    """Extract text + split into chunks + attach metadata."""
    act_name = os.path.basename(filepath).replace(".pdf", "")
    text = extract_text(filepath)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    raw_chunks = splitter.split_text(text)

    chunks = []
    for chunk in raw_chunks:
        chunks.append({
            "id": str(uuid.uuid4()),
            "text": chunk,
            "act_name": act_name
        })

    return chunks

def save_in_batches(chunks, act_name):
    """Writes chunk batches to JSON files (96 per file)."""
    batch_num = 1
    for i in range(0, len(chunks), MAX_RECORDS_PER_JSON):
        batch = chunks[i:i + MAX_RECORDS_PER_JSON]
        filename = f"{act_name}_chunks_{batch_num}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(batch, f, indent=2, ensure_ascii=False)

        batch_num += 1

def main():
    act_name = os.path.basename(PDF_PATH).replace(".pdf", "")
    print(f"Processing PDF: {PDF_PATH}")

    chunks = chunk_pdf(PDF_PATH)
    save_in_batches(chunks, act_name)

    print("Finished. Output JSON files saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
