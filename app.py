import os
import uuid
import time
import json
import threading
import logging
import base64
from pathlib import Path
import tempfile

import requests
import streamlit as st
import streamlit.components.v1 as components 

from utils.storage_service import upload_to_storj

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        
        /* CHANGED: Increased from 15vh to 30vh to move everything down to center */
        margin-top: 30vh; 
        
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 2; /* Ensure text is above image */
    }
    
    .centered-subtitle {
        text-align: center;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 400;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 1rem; 
        position: relative;
        z-index: 2;
    }

    /* 3. Background Justice Element (Flow-based) */
    .justice-bg {
        display: block;
        width: 100%;   
        height: 60vh;
        
        /* Positioning logic */
        margin-top: -2rem; /* Pull it closer to the text */
        margin-left: auto;
        margin-right: auto;
        
        /* Image handling */
        background-repeat: no-repeat;
        background-position: top center; 
        background-size: 400px; 
        
        opacity: 0.1; /* Translucent effect */
        z-index: 1;    /* Behind text */
        pointer-events: none;
    }

    /* Input box styling */
    .stChatInputContainer {
        padding-bottom: 2rem;
    }

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

# --- API URL ---
if "api_url" not in st.session_state:
    st.session_state.api_url = os.getenv("API_URL", "https://safeclause-ai.onrender.com")

# ---------------- Background Wake-Up Call ----------------
def wake_up_api(url):
    try:
        requests.get(f"{url}/health", timeout=5)
    except Exception:
        pass

if "api_woken" not in st.session_state:
    st.session_state.api_woken = True
    thread = threading.Thread(target=wake_up_api, args=(st.session_state.api_url,))
    thread.start()

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
            if st.button("Upload Files", use_container_width=True, type="primary"):
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                temp_paths = []
                uploaded_file_names = []
                
                try:
                    # Create temporary files and store original names
                    for f in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
                            tmp.write(f.getbuffer())
                            temp_paths.append(tmp.name)
                            uploaded_file_names.append(f.name)
                    
                    # Upload each file with original filename
                    upload_success_count = 0
                    for i, (temp_path, original_name) in enumerate(zip(temp_paths, uploaded_file_names), 1):
                        status_text.text(f"Uploading {original_name}... ({i}/{len(temp_paths)})")
                        progress_bar.progress(i / len(temp_paths))
                        
                        ok, result = upload_to_storj(temp_path, original_name)
                        if ok:
                            st.session_state.docs.append({"name": original_name, "url": result})
                            upload_success_count += 1
                            st.success(f"✅ Uploaded: {original_name}")
                        else:
                            st.error(f"❌ Failed to upload {original_name}: {result}")
                    
                    # Show final status
                    if upload_success_count > 0:
                        if upload_success_count == len(uploaded_files):
                            st.success(f"🎉 All {upload_success_count} files uploaded successfully!")
                        else:
                            st.warning(f"⚠️ {upload_success_count}/{len(uploaded_files)} files uploaded successfully")
                        st.session_state.files_locked = True
                        status_text.empty()
                        time.sleep(1)  # Brief pause before rerun
                        st.rerun()
                    else:
                        st.error("❌ No files were uploaded successfully. Please check the error messages above.")
                        
                finally:
                    for p in temp_paths:
                        try:
                            os.unlink(p)
                        except: 
                            pass
    else:
        if st.button("➕ Add more sources", use_container_width=True):
            upload_locked_dialog()

    st.divider()
    if st.button("New Thread", icon="🔄", use_container_width=True):
        reset_session()

# ---------------- Typewriter Effect JavaScript ----------------
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
            inputField.placeholder = currentPhrase.substring(0, currentCharIndex - 1);
            currentCharIndex--;
            typeSpeed = 50; 
        } else {
            inputField.placeholder = currentPhrase.substring(0, currentCharIndex + 1);
            currentCharIndex++;
            typeSpeed = 100; 
        }

        if (!isDeleting && currentCharIndex === currentPhrase.length) {
            isDeleting = true;
            typeSpeed = 2000; 
        } else if (isDeleting && currentCharIndex === 0) {
            isDeleting = false;
            currentPhraseIndex = (currentPhraseIndex + 1) % phrases.length;
            typeSpeed = 500;
        }

        setTimeout(typeWriter, typeSpeed);
    }

    typeWriter();
</script>
"""

# ---------------- Main Interface ----------------

if not st.session_state.messages:
    
    # Image Loading Logic
    image_path = "img/justice_icon.png"
    bg_style = ""
    
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            bg_style = f'style="background-image: url(data:image/png;base64,{data});"'
        except Exception as e:
            st.error(f"Error loading image: {e}")
    else:
        # Fallback
        default_url = "https://cdn-icons-png.flaticon.com/512/924/924953.png"
        bg_style = f'style="background-image: url({default_url}); opacity: 0.05;"'

    st.markdown('<div class="centered-title">SafeClause.ai</div>', unsafe_allow_html=True)
    st.markdown('<div class="centered-subtitle">Where legal knowledge begins</div>', unsafe_allow_html=True)
    
    # The Background Image Div
    st.markdown(f'<div class="justice-bg" {bg_style}></div>', unsafe_allow_html=True)
    
    components.html(typewriter_js, height=0, width=0)

# 2. Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. Input & Processing
if prompt := st.chat_input("Ask a legal question..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "query": prompt,
        "doc_filepaths": [d["url"] for d in st.session_state.docs],
        "session_id": st.session_state.session_id,
    }

    with st.chat_message("assistant"):
        status_container = st.status("Initializing...", expanded=False)
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
                        status_container.update(label=event.get("content", "Processing..."), state="running")
                    elif etype == "report":
                        final_report = event.get("content", "")
                        response_placeholder.markdown(final_report)
                    elif etype == "complete":
                        status_container.update(label="Analysis Complete", state="complete", expanded=False)
                    elif etype == "error":
                        status_container.update(label="Error", state="error")
                        st.error(event.get("error"))
            else:
                status_container.update(label="Server Error", state="error")
                st.error(f"Status code: {response.status_code}")

        except Exception as e:
            status_container.update(label="Connection Error", state="error")
            st.error(f"Error: {str(e)}")

    if final_report:
        st.session_state.messages.append({"role": "assistant", "content": final_report})
