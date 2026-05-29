import sys
import os
import asyncio
import time
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from evaluation.telemetry import AsyncTelemetryLogger
# In a full deployment, we would also import EnterpriseClinicalRouter, EnterpriseAsyncStreamEngine, etc.

async def mock_clinical_generation(query: str):
    """Simulates the async streaming LLM generating tokens."""
    print(f"   [LLM] Streaming clinical response for: '{query}'...")
    await asyncio.sleep(1.5) # Simulating generation time
    return {
        "ttft_sec": 0.18,
        "tokens_per_sec": 245.5,
        "total_tokens": 368
    }

async def mock_ragas_evaluation():
    """Simulates the heavy Ragas AI judge running in the background."""
    print(f"   [RAGAS] Analyzing context precision and faithfulness...")
    await asyncio.sleep(2.0) # Ragas takes time because it makes multiple LLM calls
    return {
        "faithfulness": 0.98,
        "relevance": 0.92,
        "precision": 0.88
    }

async def process_clinical_query(query: str, logger: AsyncTelemetryLogger):
    """The master asynchronous orchestrator for a single user interaction."""
    session_id = str(uuid.uuid4())[:8]
    print(f"\n--- INITIATING TRANSACTION [{session_id}] ---")
    
    # 1. Routing (Synchronous fast-path)
    print("   [ROUTER] Classifying intent...")
    intent_route = "CLINICAL_RAG"
    router_latency = 0.23
    
    # 2. Generation & Evaluation running CONCURRENTLY
    # The user gets the text stream immediately while Ragas grades it in the background
    generation_task = asyncio.create_task(mock_clinical_generation(query))
    evaluation_task = asyncio.create_task(mock_ragas_evaluation())
    
    # Wait for both critical backend processes to finish
    gen_metrics, eval_metrics = await asyncio.gather(generation_task, evaluation_task)
    
    # 3. Compile Master Telemetry Payload
    master_metrics = {
        "session_id": session_id,
        "query_length": len(query),
        "intent_route": intent_route,
        "router_latency_sec": router_latency,
        **gen_metrics,
        **eval_metrics
    }
    
    # 4. Fire-and-forget the async log write
    print("   [TELEMETRY] Dispatching metrics to persistent disk log...")
    await logger.log_transaction(master_metrics)
    print(f"--- TRANSACTION [{session_id}] COMPLETE ---")

async def execute_day_7_pipeline():
    print("===================================================")
    print(" STARTING DAY 7: ASYNC ORCHESTRATION & TELEMETRY ")
    print("===================================================")
    
    # Initialize the Logger
    logger = AsyncTelemetryLogger(log_dir="./data/telemetry", filename="system_metrics.csv")
    
    queries = [
        "What is the first-line treatment for essential hypertension?",
        "Explain the mechanism of action for SSRIs.",
        "What are the diagnostic criteria for Rheumatoid Arthritis?"
    ]
    
    # Run multiple queries concurrently to test the file locking and async queues
    tasks = [process_clinical_query(q, logger) for q in queries]
    
    start_time = time.perf_counter()
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start_time
    
    print("\n===================================================")
    print(f" DAY 7 COMPLETE: PROCESSED 3 CONCURRENT QUERIES IN {elapsed:.2f}s")
    print(" CHECK ./data/telemetry/system_metrics.csv FOR LOGS  ")
    print("===================================================")

if __name__ == "__main__":
    asyncio.run(execute_day_7_pipeline())
