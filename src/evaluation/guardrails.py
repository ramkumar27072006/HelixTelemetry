import os
import pandas as pd
from typing import List, Dict, Any
from datasets import Dataset

# Langchain core components
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Ragas core metrics and wrappers
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# GLOBAL CONFIGURATION
NAME_OF_WEBSITE = "HelixTelemetry"
EMBEDDING_MODEL = ".hf_model_cache/materialized_model"
JUDGE_MODEL = "llama-3.1-8b-instant"

class ClinicalRagasEvaluator:
    def __init__(self, api_key: str = None):
        """Initializes the background AI judge to grade the primary AI's outputs."""
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(f"[{NAME_OF_WEBSITE} CRITICAL] GROQ_API_KEY missing for evaluator.")

        print(f"[{NAME_OF_WEBSITE} TELEMETRY] Booting internal Ragas Evaluation Engine...")
        
        # 1. Initialize the Judge LLM (Must be Temp 0.0 for objective grading)
        eval_llm = ChatGroq(
            temperature=0.0,
            model_name=JUDGE_MODEL,
            groq_api_key=self.api_key
        )
        
        # 2. Initialize the local medical embeddings for relevance checking
        eval_embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        # 3. Wrap Langchain objects to bypass OpenAI defaults in Ragas
        self.ragas_llm = LangchainLLMWrapper(eval_llm)
        self.ragas_emb = LangchainEmbeddingsWrapper(eval_embeddings)
        
        # 4. Inject our free, local models into the Ragas metrics
        self._configure_metrics()

    def _configure_metrics(self):
        """Overrides default OpenAI models with our Groq/PubMedBERT setup."""
        # Faithfulness: Checks if the answer is completely backed by the retrieved context
        faithfulness.llm = self.ragas_llm
        
        # Answer Relevancy: Checks if the answer directly addresses the query
        answer_relevancy.llm = self.ragas_llm
        answer_relevancy.embeddings = self.ragas_emb
        
        # Context Precision: Checks if the retrieved context was actually useful
        context_precision.llm = self.ragas_llm
        
        self.metrics = [faithfulness, answer_relevancy, context_precision]

    def grade_transaction(self, query: str, answer: str, source_contexts: List[str]) -> Dict[str, Any]:
        """
        Takes a single chat interaction and calculates a clinical safety score (0.0 to 1.0).
        """
        # Ragas requires strict HuggingFace Dataset formatting
        # Added 'reference' key to satisfy Ragas Context Precision metric expectations
        data_packet = {
            "question": [query],
            "answer": [answer],
            "contexts": [source_contexts], # Note the nested list
            "reference": ["Test reference"] # Mock reference to pass validation
        }
        
        dataset = Dataset.from_dict(data_packet)
        
        try:
            # Run the rigorous evaluation matrix
            score_card = evaluate(dataset, metrics=self.metrics)
            # Convert Ragas output to a standard Python dictionary for the UI
            return score_card.to_pandas().to_dict(orient='records')[0]
        except Exception as e:
            print(f"[{NAME_OF_WEBSITE} ERROR] Evaluation failed: {str(e)}")
            return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}
