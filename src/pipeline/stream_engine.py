import os
import time
import asyncio
from typing import AsyncGenerator, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# GLOBAL CONFIGURATION
NAME_OF_WEBSITE = "HelixTelemetry"
DEFAULT_MODEL = "llama-3.1-8b-instant"

class EnterpriseAsyncStreamEngine:
    def __init__(self, api_key: str = None, model_name: str = DEFAULT_MODEL):
        """Initializes the ultra-high-speed asynchronous inference wrapper."""
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(f"[{NAME_OF_WEBSITE} CRITICAL] GROQ_API_KEY environment variable is unassigned.")
            
        self.model_name = model_name
        
        print(f"[{NAME_OF_WEBSITE} CORE] Hot-linking asynchronous LPU pipelines via Groq hardware target: {self.model_name}")
        # Initialize streaming-enabled model with exact zero-temperature configuration
        self.llm = ChatGroq(
            temperature=0.0,
            model_name=self.model_name,
            groq_api_key=self.api_key,
            streaming=True
        )
        self.parser = StrOutputParser()

    async def execute_clinical_stream(self, prompt_template: ChatPromptTemplate, variables: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Executes a non-blocking token stream across network connection threads.
        Yields granular metrics alongside text blocks for real-time dashboard instrumentation.
        """
        # Formulate execution sequence using LangChain Expression Language (LCEL)
        chain = prompt_template | self.llm | self.parser
        
        start_time = time.perf_counter()
        token_count = 0
        time_to_first_token = None
        
        print(f"[{NAME_OF_WEBSITE} ASYNC] Network call dispatched. Collecting incoming token telemetry...")
        
        # Stream chunks asynchronously as they drop from Groq's LPUs
        async for chunk in chain.astream(variables):
            current_time = time.perf_counter()
            token_count += 1
            
            # Record Time To First Token (TTFT) performance metric
            if time_to_first_token is None:
                time_to_first_token = current_time - start_time
            
            elapsed_overall = current_time - start_time
            tokens_per_second = token_count / elapsed_overall if elapsed_overall > 0 else 0
            
            # Package text output along with real-time performance telemetry
            yield {
                "token": chunk,
                "telemetry": {
                    "token_index": token_count,
                    "ttft_seconds": round(time_to_first_token, 4),
                    "elapsed_seconds": round(elapsed_overall, 4),
                    "tokens_per_second": round(tokens_per_second, 2)
                }
            }
