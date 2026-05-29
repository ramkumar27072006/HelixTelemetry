import streamlit as st

class ClinicalThemeInjector:
    @staticmethod
    def enforce_enterprise_styling():
        """Injects heavy CSS to override standard Streamlit aesthetics."""
        
        # Must be called first to expand the canvas
        st.set_page_config(
            page_title="HelixTelemetry Command Center", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        custom_css = """
        <style>
            /* Global Background & Font */
            .stApp {
                background-color: #0b0f19;
                color: #e2e8f0;
                font-family: 'Inter', -apple-system, sans-serif;
            }
            
            /* Hide Default Streamlit Clutter */
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}

            /* Sidebar Restyling (The Data Engine Panel) */
            [data-testid="stSidebar"] {
                background-color: #111827 !important;
                border-right: 1px solid #1f2937;
            }
            
            /* Chat Message Bubbles */
            [data-testid="stChatMessage"] {
                background-color: #1e293b !important;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 10px;
            }
            
            /* User Message Differentiation */
            [data-testid="stChatMessage"][data-baseweb="block"]:nth-child(odd) {
                background-color: #0f172a !important;
                border-left: 4px solid #38bdf8;
            }
            
            /* AI Message Differentiation */
            [data-testid="stChatMessage"][data-baseweb="block"]:nth-child(even) {
                border-left: 4px solid #22c55e;
            }

            /* Custom Metric Cards */
            .metric-card {
                background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 12px;
                text-align: center;
                margin-bottom: 10px;
            }
            .metric-title { color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;}
            .metric-value { color: #38bdf8; font-size: 1.5rem; font-weight: bold; margin-top: 5px;}
            
            /* Citation Source Card */
            .source-card {
                background-color: #0284c7;
                color: white;
                padding: 8px;
                font-size: 0.75rem;
                border-radius: 4px;
                margin-top: 10px;
                display: inline-block;
            }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)
