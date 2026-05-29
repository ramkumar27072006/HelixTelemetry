import streamlit as st
import uuid

class StateManager:
    @staticmethod
    def initialize_session():
        """Bootstraps the memory matrix for the Streamlit session."""
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
            
        if "chat_history" not in st.session_state:
            # Load with an initial AI greeting
            st.session_state.chat_history = [
                {"role": "assistant", "content": "System Online. Awaiting clinical telemetry or literature query."}
            ]
            
        if "latest_telemetry" not in st.session_state:
            st.session_state.latest_telemetry = {
                "faithfulness": 0.0,
                "relevance": 0.0,
                "ttft": 0.0
            }

class UIRenderer:
    @staticmethod
    def render_sidebar():
        """Left Pane: Data Engine Status"""
        with st.sidebar:
            st.markdown("### Engine Status")
            st.markdown("---")
            
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Vector Core</div>
                <div class="metric-value" style="color: #22c55e;">ONLINE</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Active Corpus</div>
                <div class="metric-value">MedQA USMLE</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.caption(f"Session ID: {st.session_state.session_id[:8]}")

    @staticmethod
    def render_telemetry_pane():
        """Right Pane: Live Guardrail Metrics (Placeholder for Day 9 Plotly)"""
        st.markdown("### Live Telemetry")
        st.markdown("---")
        st.info("Awaiting Plotly Engine Injection (Day 9)")
        
        # Render static HTML placeholders for now based on session state
        metrics = st.session_state.latest_telemetry
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Last TTFT</div>
            <div class="metric-value">{metrics['ttft']}s</div>
        </div>
        """, unsafe_allow_html=True)
