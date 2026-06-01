<p align="center">
  <!-- ADD YOUR IMAGE LINK IN THE SRC BELOW -->
  <img width="256" height="256" alt="ChatGPT Image Jun 1, 2026, 09_45_05 AM" src="https://github.com/user-attachments/assets/2841101c-083a-4fef-8ee7-08cd4ccf6557" />
</p>

<h1 align="center">HelixTelemetry | Enterprise Clinical Command Center</h1>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-0284c7.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/LLM-Llama_3.1_8B-22c55e.svg" alt="LLM">
  <img src="https://img.shields.io/badge/Embeddings-PubMedBERT-eab308.svg" alt="Embeddings">
  <img src="https://img.shields.io/badge/Telemetry-Ragas_Async-8b5cf6.svg" alt="Telemetry">
</p>

**HelixTelemetry** is a production-grade, clinically aligned Medical RAG Architecture engineered for high-velocity data ingestion, deterministic safety routing, and automated hallucination tracking. Built to transcend basic LLM wrappers, this system introduces a multi-threaded, non-blocking telemetry engine and an enterprise 3-pane UI, deployed globally to Web and Native Android.

---

##  Core Architecture & Engineering Highlights

1. **Deterministic Safety Router (Zero-Temperature)**
   Prevents dangerous LLM extrapolations by passing raw queries through a strict gatekeeper. Malicious prompts or acute emergency symptoms bypass the generator entirely, immediately rendering hardcoded clinical safety interventions.
2. **High-Velocity Async Token Streaming**
   Bridges LangChain's asynchronous generators directly into Streamlit's synchronous loop, unlocking multi-threaded execution. Renders responses at hundreds of tokens per second using Groq's LPU hardware.
3. **Automated Hallucination Telemetry (Ragas)**
   Every RAG transaction is asynchronously graded in the background for *Faithfulness*, *Context Precision*, and *Answer Relevancy*.
4. **Non-Blocking I/O Logging**
   Protected by `asyncio.Lock()` and managed via `aiofiles`, system metrics (TTFT, tokens/sec, Ragas scores) are successfully written to persistent CSV storage simultaneously during generation without freezing the Global Interpreter Lock (GIL) or the UI.
5. **Domain-Specific Embeddings (PubMedBERT)**
   Leverages `NeuML/pubmedbert-base-embeddings` alongside ChromaDB for hyper-accurate local vector similarity search against the MedQA USMLE clinical corpus.

---

##  UI / UX Design

Streamlit's default aesthetic was entirely overridden via aggressive CSS DOM injection (`src/ui/themes.py`). 
* **Dark-Mode Enterprise Matrix:** Styled to mimic Tier-1 EHR systems (Cerner/Epic) to reduce clinical eye strain.
* **Live Plotly Integration:** Visualizes the backend `system_metrics.csv` asynchronous logs dynamically on execution completion.
* **Persistent State Management:** Seamlessly retains chat history and telemetry scores through strict `st.session_state` singleton rules.

---

##  Tech Stack
* **Inference Model:** Meta Llama-3.1-8b-instant (via Groq API)
* **Embedding Model:** PubMedBERT (HuggingFace)
* **Vector Store:** ChromaDB (Persistent SQLite)
* **Pipeline/Agent Framework:** LangChain Core
* **Telemetry & Grading:** Ragas, Plotly, Pandas, Aiofiles
* **Frontend:** Streamlit & Custom CSS
* **Mobile Portability:** Google Chrome Labs Bubblewrap (TWA)

---

##  Local Installation & Setup

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/HelixTelemetry.git
cd HelixTelemetry
```

**2. Create the Python Virtual Environment:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Configure Environment Variables:**
Create a `.env` file in the root directory and securely add your Groq LPU Key:
```env
GROQ_API_KEY=gsk_your_api_key_here
```

**5. Boot the Engine:**
```bash
streamlit run app.py
```

---

##  Deployment

* **Cloud Web Application:** Hosted seamlessly via [Streamlit Community Cloud](https://share.streamlit.io/). Secrets securely managed in the Streamlit advanced settings portal.
* **Native Android (.APK):** The live Streamlit URL is wrapped in a Trusted Web Activity (TWA) compiled using Median (GoNative) / Bubblewrap CLI.

---

* **Disclaimer:** HelixTelemetry is a portfolio engineering project demonstrating advanced ML architecture. It is not an FDA-approved medical device and should not be deployed in real-world clinical environments or used for actual medical diagnostic decision-making.*
