import sys
import os
import asyncio
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

# Expose backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from ui.themes import ClinicalThemeInjector
from ui.views import StateManager, UIRenderer
from ui.components import TelemetryVisualizer
from pipeline.router import EnterpriseClinicalRouter
from pipeline.stream_engine import EnterpriseAsyncStreamEngine
from pipeline.bridge import StreamlitAsyncBridge

# --- INITIALIZATION ---
ClinicalThemeInjector.enforce_enterprise_styling()
StateManager.initialize_session()

# Load Backend Engines into cache so they don't reload on every keystroke
@st.cache_resource
def load_engines():
    # Attempt to load from OS environment first, fallback to Streamlit Secrets
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key and "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
        
    router = EnterpriseClinicalRouter(api_key=api_key)
    streamer = EnterpriseAsyncStreamEngine(api_key=api_key)
    visualizer = TelemetryVisualizer()
    return router, streamer, visualizer

router, streamer, visualizer = load_engines()

# --- UI LAYOUT ---
UIRenderer.render_sidebar()
chat_col, telemetry_col = st.columns([7, 3], gap="large")

# Right Pane: Render Live Plotly Metrics
with telemetry_col:
    st.markdown("### LIVE TELEMETRY")
    st.markdown("---")
    
    # Fetch the absolute latest metrics from the CSV log
    latest_stats = visualizer.get_latest_metrics()
    
    st.plotly_chart(visualizer.render_gauge(latest_stats['faithfulness'], "RAGAS Faithfulness"), use_container_width=True)
    st.plotly_chart(visualizer.render_gauge(latest_stats['relevance'], "Answer Relevance"), use_container_width=True)
    st.plotly_chart(visualizer.render_gauge(latest_stats['ttft_sec'], "Time To First Token", is_time=True), use_container_width=True)

# Center Pane: Clinical Interface
with chat_col:
    st.markdown("### CLINICAL RAG INTERFACE")
    
    # Draw chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "route" in msg:
                st.markdown(f"<div class='source-card'>ROUTE: {msg['route']}</div>", unsafe_allow_html=True)

    # User Input
    if prompt := st.chat_input("Enter clinical query here..."):
        
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            # 1. Triage the Query (Day 5 Router)
            triage_decision = router.triage_query(prompt)
            intent = triage_decision["route"]
            
            if not triage_decision["is_safe_for_rag"]:
                # Hard block - Render fallback
                fallback = router.get_hardcoded_fallback(intent)
                st.markdown(fallback)
                st.markdown(f"<div class='source-card'>BLOCKED: {intent}</div>", unsafe_allow_html=True)
                st.session_state.chat_history.append({"role": "assistant", "content": fallback, "route": intent})
                
            else:
                # 2. Safe Query - Execute Async Stream (Day 4 & Day 2 Vector Core logic)
                # In full production, inject vectorstore retrieval context here. 
                # For UI integration, we stream the engine.
                response_placeholder = st.empty()
                full_response = ""
                
                clinical_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are the HelixTelemetry Command Center AI. Answer accurately based on medical context."),
                    ("human", "{query}")
                ])
                
                # Fetch async generator and bridge it to sync Streamlit
                async_gen = streamer.execute_clinical_stream(clinical_prompt, {"query": prompt})
                sync_gen = StreamlitAsyncBridge.run_async_generator(async_gen)
                
                # Render tokens instantly
                for packet in sync_gen:
                    if "token" in packet and hasattr(packet["token"], "content"):
                        full_response += packet["token"].content
                    elif "token" in packet and isinstance(packet["token"], str):
                         full_response += packet["token"]
                    elif isinstance(packet, str):
                         full_response += packet
                         
                    response_placeholder.markdown(full_response + "▌")
                    
                response_placeholder.markdown(full_response)
                st.markdown(f"<div class='source-card'>RAG PATHWAY: {intent}</div>", unsafe_allow_html=True)
                
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": full_response, 
                    "route": intent
                })
                
                # Trigger a background re-run to update the telemetry gauges immediately
                st.rerun()
