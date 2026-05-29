import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

class TelemetryVisualizer:
    def __init__(self, log_path: str = "./data/telemetry/system_metrics.csv"):
        self.log_path = log_path

    def get_latest_metrics(self) -> dict:
        """Safely fetches the most recent system metrics from the disk log."""
        if not os.path.exists(self.log_path):
            return {"faithfulness": 0.0, "relevance": 0.0, "ttft_sec": 0.00}
            
        try:
            # Read only the last few rows to save memory
            df = pd.read_csv(self.log_path)
            if df.empty:
                return {"faithfulness": 0.0, "relevance": 0.0, "ttft_sec": 0.00}
                
            latest = df.iloc[-1]
            return {
                "faithfulness": float(latest.get("faithfulness", 0.0)),
                "relevance": float(latest.get("relevance", 0.0)),
                "ttft_sec": float(latest.get("ttft_sec", 0.0))
            }
        except Exception:
            return {"faithfulness": 0.0, "relevance": 0.0, "ttft_sec": 0.00}

    @staticmethod
    def render_gauge(value: float, title: str, is_time: bool = False) -> go.Figure:
        """Constructs a dark-themed Plotly gauge matching the UI CSS."""
        
        # Invert color logic for time (lower is better for TTFT)
        if is_time:
            range_max = 2.0
            steps = [
                {'range': [0, 0.5], 'color': "#22c55e"},   # Green (Fast)
                {'range': [0.5, 1.5], 'color': "#eab308"}, # Yellow (Warning)
                {'range': [1.5, 2.0], 'color': "#ef4444"}  # Red (Slow)
            ]
            value_format = f"{value:.2f}s"
        else:
            range_max = 1.0
            steps = [
                {'range': [0, 0.75], 'color': "#ef4444"},  # Red (Hallucination risk)
                {'range': [0.75, 1.0], 'color': "#22c55e"} # Green (Safe)
            ]
            value_format = f"{value:.2f}"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            number={'valueformat': '.2f', 'font': {'color': '#38bdf8', 'size': 30}},
            title={'text': title, 'font': {'color': '#94a3b8', 'size': 14}},
            gauge={
                'axis': {'range': [0, range_max], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': "#0f172a", 'thickness': 0.1},
                'bgcolor': "#1e293b",
                'borderwidth': 1,
                'bordercolor': "#334155",
                'steps': steps
            }
        ))
        
        # Make the background transparent to blend with Streamlit CSS
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=200,
            margin=dict(l=15, r=15, t=40, b=15)
        )
        return fig
