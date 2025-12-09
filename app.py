import os
import uuid
import time
import json
from pathlib import Path
import tempfile

import requests
import streamlit as st
import streamlit.components.v1 as components # Import components for JS

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
        margin-top: 15vh; /* Increased top margin for vertical centering */
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

    /* Input box styling */
    .stChatInputContainer {
        padding-bottom: 2rem;
    }

    /* Hide the placeholder text immediately when the user clicks (focuses) 
       on the input box.
    */
    textarea[data-testid="stChatInputTextArea"]:focus::placeholder {
        color: transparent !important;
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

# ---------------- Typewriter Effect JavaScript ----------------
# This script injects JS to find the chat input and animate the placeholder
typewriter_js = """
<script>
    const phrases = [
        "Draft a Non-Disclosure Agreement for...",
        "Summarize the liability clauses in this contract...",
        "Identify the termination conditions...",
        "Explain the indemnity obligations..."
    ];
    
    let currentPhraseIndex = 0;
    let currentCharIndex = 0;
    let isDeleting = false;
    let typeSpeed = 100;

    function typeWriter() {
        const inputField = window.parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
        
        if (!inputField) {
            setTimeout(typeWriter, 100);
            return;
        }

        const currentPhrase = phrases[currentPhraseIndex];

        if (isDeleting) {
            // Remove a character
            inputField.placeholder = currentPhrase.substring(0, currentCharIndex - 1);
            currentCharIndex--;
            typeSpeed = 50; // Faster when deleting
        } else {
            // Add a character
            inputField.placeholder = currentPhrase.substring(0, currentCharIndex + 1);
            currentCharIndex++;
            typeSpeed = 100; // Normal typing speed
        }

        if (!isDeleting && currentCharIndex === currentPhrase.length) {
            // Finished typing phrase, pause before deleting
            isDeleting = true;
            typeSpeed = 2000; 
        } else if (isDeleting && currentCharIndex === 0) {
            // Finished deleting, move to next phrase
            isDeleting = false;
            currentPhraseIndex = (currentPhraseIndex + 1) % phrases.length;
            typeSpeed = 500;
        }

        setTimeout(typeWriter, typeSpeed);
    }

    // Start the loop
    typeWriter();
</script>
"""

# ---------------- Main Interface ----------------

# 1. Empty State (Centered & Aligned)
if not st.session_state.messages:
    # TITLE
    st.markdown('<div class="centered-title">SafeClause.ai</div>', unsafe_allow_html=True)
    st.markdown('<div class="centered-subtitle">Where legal knowledge begins</div>', unsafe_allow_html=True)
    
    # Inject the JS only on the empty screen
    components.html(typewriter_js, height=0, width=0)

# 2. Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. Input & Processing
# Note: Initial placeholder is set, but JS will immediately overwrite it
if prompt := st.chat_input("Ask a legal question..."):
    
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