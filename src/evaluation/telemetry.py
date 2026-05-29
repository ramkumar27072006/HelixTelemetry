import os
import asyncio
import aiofiles
from datetime import datetime
from typing import Dict, Any

# GLOBAL CONFIGURATION
NAME_OF_WEBSITE = "HelixTelemetry"

class AsyncTelemetryLogger:
    def __init__(self, log_dir: str = "../../data/telemetry", filename: str = "system_metrics.csv"):
        """Initializes the non-blocking telemetry logger with thread-safe file locks."""
        self.log_dir = log_dir
        self.log_path = os.path.join(self.log_dir, filename)
        
        # Ensure the telemetry directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Asyncio lock prevents race conditions if multiple users query the system simultaneously
        self._lock = asyncio.Lock()
        
        print(f"[{NAME_OF_WEBSITE} OBSERVE] Async Telemetry Engine spinning up at {self.log_path}...")
        self._initialize_log_file()

    def _initialize_log_file(self):
        """Creates the CSV with strict headers if it does not already exist."""
        headers = (
            "timestamp,session_id,query_length,intent_route,router_latency_sec,"
            "ttft_sec,tokens_per_sec,total_tokens,faithfulness,relevance,precision\n"
        )
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', encoding='utf-8') as f:
                f.write(headers)
            print(f"[{NAME_OF_WEBSITE} OBSERVE] Initialized new telemetry matrix headers.")

    async def log_transaction(self, metrics: Dict[str, Any]):
        """
        Asynchronously appends a transaction record to the CSV.
        Awaits the lock to prevent file corruption during high-traffic spikes.
        """
        timestamp = datetime.utcnow().isoformat()
        
        # Sanitize data to prevent CSV delimiter breaking
        session_id = str(metrics.get("session_id", "anon_01"))
        query_length = str(metrics.get("query_length", 0))
        intent_route = str(metrics.get("intent_route", "UNKNOWN")).replace(",", "")
        
        # Extract numerical metrics safely
        router_latency = f"{metrics.get('router_latency_sec', 0.0):.3f}"
        ttft = f"{metrics.get('ttft_sec', 0.0):.3f}"
        tps = f"{metrics.get('tokens_per_sec', 0.0):.1f}"
        total_tokens = str(metrics.get("total_tokens", 0))
        
        # Ragas safety scores
        faithfulness = f"{metrics.get('faithfulness', 0.0):.2f}"
        relevance = f"{metrics.get('relevance', 0.0):.2f}"
        precision = f"{metrics.get('precision', 0.0):.2f}"
        
        # Construct the exact CSV row
        csv_row = (
            f"{timestamp},{session_id},{query_length},{intent_route},{router_latency},"
            f"{ttft},{tps},{total_tokens},{faithfulness},{relevance},{precision}\n"
        )
        
        # Non-blocking file write protected by an async lock
        async with self._lock:
            async with aiofiles.open(self.log_path, mode='a', encoding='utf-8') as f:
                await f.write(csv_row)
