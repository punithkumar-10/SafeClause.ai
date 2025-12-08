import os
import uuid
import time
import json
from pathlib import Path
import tempfile

import requests
import streamlit as st

from utils.storage_service import upload_to_storj

# ---------------- Configuration ----------------
st.set_page_config(
    page_title="SafeClause.ai",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
    /* 1. Header Cleanup */
    [data-testid="stHeader"] {
        background-color: transparent; 
    }
    
    .stAppDeployButton {
        display: none;
    }

    .block-container {
        padding-top: 2rem;
    }
    
    /* 2. Title Styling */
    .centered-title {
        text-align: center;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 300;
        font-size: 3.5rem;
        color: #000000;
        margin-top: 6vh;
        margin-bottom: 0.5rem;
    }
    
    .centered-subtitle {
        text-align: center;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 400;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }

    /* 3. Button Styling (Perfectly Aligned) */
    div.stButton > button {
        width: 100%;
        /* Force a fixed height so all boxes are identical */
        height: 70px !important; 
        margin: 0px;
        padding: 0px 20px; /* Horizontal padding */
        
        background-color: #ffffff;
        color: #1f1f1f;
        border: 1px solid #dcdcdc;
        border-radius: 10px; 
        
        /* Typography */
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 15px;
        text-align: left;
        
        /* Flex alignment to keep text perfectly vertically centered */
        display: flex;
        align-items: center; 
        justify-content: flex-start;
        
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background-color: #f9f9f9;
        border-color: #b0b0b0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    div.stButton > button:focus {
        box-shadow: none;
        border-color: #000;
    }
    
    /* Remove default column gaps to give us manual control */
    [data-testid="column"] {
        padding: 0px 10px; 
    }

    /* Input box styling */
    .stChatInputContainer {
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Session State ----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [] 

if "docs" not in st.session_state:
    st.session_state.docs = []

if "files_locked" not in st.session_state:
    st.session_state.files_locked = False

if "api_url" not in st.session_state:
    st.session_state.api_url = os.getenv("API_URL", "http://localhost:8000")

# ---------------- Helper Functions ----------------
@st.dialog("Sources Locked")
def upload_locked_dialog():
    st.markdown("Sources are locked for this session. Clear the thread to start over.")
    if st.button("Clear Thread"):
        reset_session()

def reset_session():
    st.session_state.messages = []
    st.session_state.docs = []
    st.session_state.files_locked = False
    st.session_state.session_id = str(uuid.uuid4())
    st.rerun()

# ---------------- Sidebar (Sources) ----------------
with st.sidebar:
    st.markdown("### 📚 Sources")
    
    if st.session_state.docs:
        for doc in st.session_state.docs:
            with st.container(border=True):
                st.markdown(f"**📄 {doc['name']}**")
    else:
        st.info("No sources attached.")

    st.divider()

    if not st.session_state.files_locked:
        uploaded_files = st.file_uploader(
            "Add Knowledge", 
            type=["pdf", "docx", "txt", "doc"], 
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            if st.button("Process Sources", use_container_width=True, type="primary"):
                progress_bar = st.progress(0.0)
                temp_paths = []
                try:
                    for f in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
                            tmp.write(f.getbuffer())
                            temp_paths.append(tmp.name)
                    
                    for i, p in enumerate(temp_paths, 1):
                        progress_bar.progress(i / len(temp_paths))
                        ok, url = upload_to_storj(p)
                        if ok:
                            st.session_state.docs.append({"name": Path(p).name, "url": url})
                        else:
                            st.error(f"Failed to upload {Path(p).name}")
                    
                    if st.session_state.docs:
                        st.session_state.files_locked = True
                        st.rerun()
                finally:
                    for p in temp_paths:
                        try:
                            os.unlink(p)
                        except: pass
    else:
        if st.button("➕ Add more sources", use_container_width=True):
            upload_locked_dialog()

    st.divider()
    if st.button("New Thread", icon="🔄", use_container_width=True):
        reset_session()

# ---------------- Main Interface ----------------

# 1. Empty State (Centered & Aligned)
if not st.session_state.messages:
    # TITLE
    st.markdown('<div class="centered-title">SafeClause.ai</div>', unsafe_allow_html=True)
    st.markdown('<div class="centered-subtitle">Where legal knowledge begins</div>', unsafe_allow_html=True)
    
    st.write("") 
    st.write("") 

    # --- ALIGNMENT FIX ---
    # We use 5 columns: [spacer, box1, box2, box3, spacer]
    # Ratios: 1 (spacer) : 2 (box) : 2 (box) : 2 (box) : 1 (spacer)
    # This centers the boxes and prevents them from stretching too wide.
    _, col1, col2, col3, _ = st.columns([0.5, 1, 1, 1, 0.5])
    
    prompt_input = None

    with col1:
        if st.button("📝 Summarize Contract", help="Get a quick overview"):
            prompt_input = "Summarize the key clauses, obligations, and dates in this contract."

    with col2:
        if st.button("⚖️ Identify Liabilities", help="Find potential risks"):
            prompt_input = "Identify all indemnity clauses and potential financial liabilities in this document."

    with col3:
        if st.button("✍️ Draft NDA for ...", help="Generate a new document"):
            prompt_input = "Draft a standard Mutual Non-Disclosure Agreement (NDA) for two tech companies."

else:
    prompt_input = None

# 2. Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. Input & Processing
if prompt := (st.chat_input("Ask a legal question...") or prompt_input):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "query": prompt,
        "doc_filepaths": [d["url"] for d in st.session_state.docs],
        "session_id": st.session_state.session_id,
    }

    # Assistant Response
    with st.chat_message("assistant"):
        status_container = st.status("Searching & Analyzing...", expanded=True)
        response_placeholder = st.empty()
        final_report = ""
        
        try:
            response = requests.post(
                f"{st.session_state.api_url}/query",
                json=payload,
                stream=True,
                timeout=300,
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if not line: continue
                    try:
                        event = json.loads(line)
                    except: continue

                    etype = event.get("type")
                    
                    if etype == "progress":
                        status_container.write(f"⚙️ {event.get('message', 'Processing...')}")
                    elif etype == "message":
                        status_container.markdown(f"*{event.get('content')}*")
                    elif etype == "report":
                        final_report = event.get("content", "")
                        response_placeholder.markdown(final_report)
                    elif etype == "complete":
                        status_container.update(label="Analysis Complete", state="complete", expanded=False)
                    elif etype == "error":
                        status_container.update(label="Error Occurred", state="error")
                        st.error(event.get("error"))
            else:
                status_container.update(label="Server Error", state="error")
                st.error(f"API returned status code: {response.status_code}")

        except Exception as e:
            status_container.update(label="Connection Error", state="error")
            st.error(f"Could not connect to API: {str(e)}")

    if final_report:
        st.session_state.messages.append({"role": "assistant", "content": final_report})