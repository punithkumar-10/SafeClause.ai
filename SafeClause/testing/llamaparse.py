from llama_cloud_services import LlamaParse

parser = LlamaParse(
    api_key="llx-oECNsOjRTaimHRD4iBjqHiASRCnnDPj2vLlCNgqejvNIgXbz",
    num_workers=4,
    verbose=True,
    language="en",
)

# 1. Execute the parsing job.
result = parser.parse("1.1.2.pdf")

# --- SIMPLIFIED EXTRACTION CODE ---
print("--- Extracted Page and Text Content ---")

# Access the list of Page objects directly via the '.pages' attribute,
# as confirmed by your working output structure.
for page_object in result.pages:
    # Directly access the 'page' and 'text' attributes of the Page object.
    page_number = page_object.page
    page_text = page_object.text

    print(f"**Page:** {page_number}")
    print(f"**Text:**")
    print(page_text.strip())
    print("-" * 30)

# NOTE: If the output structure ever changes back to a list of LlamaIndex 
# Document objects, this code will break, and you'd need the previous 
# complex logic or simply: 'for doc in result: doc.metadata["page_label"]'