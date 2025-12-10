# ⚖️ SafeClause.ai
### *Where legal knowledge begins*


## 🇮🇳 Built for the Indian Legal System

**SafeClause.ai** is a specialized AI-powered legal assistant designed specifically for the Indian legal landscape. Unlike general-purpose AI models, SafeClause is engineered to understand, analyze, and generate content strictly adhering to Indian laws, Acts, and Constitutional clauses.

Powered by **OpenAI models**, **Model Context Protocol (MCP)**, and **Hybrid RAG systems**, it delivers fast, accurate, and reliable responses, serving as an intelligent bridge between complex legal jargon and actionable insights for lawyers, law students, and businesses.

---

## 🚀 What It Does

SafeClause.ai goes beyond simple chat. It is a full-stack legal agent capable of:

* **🔍 Precision Legal Queries:** Ask complex questions about Indian law (e.g., IPC, CrPC, Contract Act) and receive citations backed by actual sections and clauses.
* **📄 Intelligent Document Analysis:** Upload legal documents (contracts, petitions, notices) to identify risks, liabilities, and missing clauses based on Indian compliance standards.
* **✍️ Automated Drafting:** Generate legally compliant drafts for NDAs, Rental Agreements, Affidavits, and more, customized to specific Indian states and jurisdictions.
* **⚖️ Case Law & Act Retrieval:** Instantly retrieve relevant sections from the Indian Constitution and major Acts without hallucinated data.

---

## 💡 Why SafeClause.ai? (The "Why It's Better" Factor)

In the era of Generative AI, tools like ChatGPT are powerful but often dangerous for legal work. SafeClause.ai addresses the critical gaps that general-purpose LLMs cannot fill.

### No More "Hallucinations"
General AI models often cite non-existent laws because they prioritize sounding natural over being factual.
> **The SafeClause Difference:** We use a Hybrid RAG architecture grounded strictly in Indian legal databases. If a section doesn't exist in the Indian Penal Code, SafeClause won't invent it.

### End-to-End Workflow vs. Simple Chat
While ChatGPT is great for emails, it cannot manage a legal workflow. It doesn't know how to cross-reference a draft contract against the latest Supreme Court judgment.
> **The SafeClause Difference:** SafeClause.ai is designed for the legal lifecycle. From **researching** a point of law to **reviewing** a contract for loopholes and **drafting** the final agreement, it handles the entire pipeline in one secure workspace.

---
## 🧠 Under the Hood: Architecture & Workflow

### Agent Orchestration Workflow
![Agent Workflow](img/agent_workflow.png)
*An overview of SafeClause.ai's multi-agent orchestration workflow. The system intelligently decides between simple query, quick memory-based answers and complex document processing pipelines involving parallel chunk analysis and synthesis using LangGraph.*

### Hybrid RAG System
![Hybrid RAG System](img/hybrid_rag.png)
*Our Hybrid RAG mechanism combines the precision of keyword-based sparse search with the semantic understanding of dense vector search. The combined results are passed through a reranker to ensure only the most relevant legal context reaches the final answer generation step.*
---

## 💻 Tech Stack

* **Frontend:** Streamlit
* **LLM Orchestration:** LangChain, LangGraph.
* **Vector Database:** Pinecone
* **MCP:** Tavily MCP
* **Object Storage:** Storj (AWS S3 compatiable)
* **Language Model:** openai/gpt-oss-120b, openai/gpt-oss-20b via Groq
* **Backend:** Python, FastAPI, Docker

---

## 📥 Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/your-username/SafeClause.ai.git](https://github.com/your-username/SafeClause.ai.git)
    cd SafeClause.ai
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables**
    Create a `.env` file and add your API keys.

4.  **Run the FastAPI Server**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 10000
    ```

6.  **Run the Frontend Application**
    ```bash
    streamlit run app.py
    ```

---

## 🛡️ Disclaimer

*SafeClause.ai is an AI-powered legal assistant intended to support legal professionals and individuals. It does not provide binding legal advice and should not replace a qualified human attorney. Always verify AI-generated insights with professional legal counsel.*

---

**SafeClause.ai** — *Where legal knowledge begins.*





