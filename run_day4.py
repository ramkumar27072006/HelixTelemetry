import sys
import os
import asyncio

# Fix path resolution for linters by inserting the src directory at index 0
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from dotenv import load_dotenv
from src.pipeline.stream_engine import EnterpriseAsyncStreamEngine
from langchain_core.prompts import ChatPromptTemplate

async def run_stress_test():
    print("===================================================")
    print("  STARTING DAY 4: ASYNC STREAM & LPU INFERENCE TEST ")
    print("===================================================")
    
    # 1. Load environment variables cautiously
    load_dotenv()
    
    # 2. Fetch key parameters securely
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[CRITICAL ERROR] GROQ_API_KEY not found in environment or .env file.")
        print("Please create a .env file at the project root with the following format:")
        print("GROQ_API_KEY=gsk_your_actual_api_key_here")
        sys.exit(1)

    # 3. Instantiate the stream engine core
    engine = EnterpriseAsyncStreamEngine(api_key=api_key)
    
    # 3. Formulate a dense mock clinical prompt setup to evaluate constraints
    clinical_prompt_structure = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert clinical oncology consultant. Evaluate the provided patient case context "
            "and output structural diagnostic step pathways. If the context details are insufficient, "
            "refuse to comment. Respond with maximum scientific precision."
        )),
        ("human", "Patient presents with a 3cm mass in left upper lobe of lung. Heavy smoker for 20 years. Outline primary actions.")
    ])
    
    # 4. Trigger the stream capture generator loop
    print("\n[STREAM STARTING - LIVE LPU OUTPUT]")
    print("---------------------------------------------------")
    
    final_metrics = {}
    
    async for packet in engine.execute_clinical_stream(prompt_template=clinical_prompt_structure, variables={}):
        # Direct character printing to stdout chunk by chunk to simulate real UI streaming
        sys.stdout.write(packet["token"])
        sys.stdout.flush()
        # Cache changing telemetry records
        final_metrics = packet["telemetry"]

    print("\n---------------------------------------------------")
    print("[STREAM COMPLETE - GENERATION METRICS LOGGED]")
    print(f"-> Total Tokens Rendered: {final_metrics.get('token_index')}")
    print(f"-> Time-to-First-Token (TTFT): {final_metrics.get('ttft_seconds')} seconds")
    print(f"-> Gross Stream Duration: {final_metrics.get('elapsed_seconds')} seconds")
    print(f"-> Real-Time Engine Velocity: {final_metrics.get('tokens_per_second')} tokens/sec")
    print("===================================================")

if __name__ == "__main__":
    # Initialize the modern native asynchronous engine loop manually
    asyncio.run(run_stress_test())
