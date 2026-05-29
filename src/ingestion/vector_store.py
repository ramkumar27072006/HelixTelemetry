import os
import pickle
import pandas as pd
from typing import List
from tqdm import tqdm
from huggingface_hub import snapshot_download
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

# GLOBAL CONFIGURATION
NAME_OF_WEBSITE = "HelixTelemetry"
EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"


class EnterpriseVectorStoreManager:
    def __init__(self, data_dir: str = "../../data/processed", db_dir: str = "../../chroma_db"):
        self.data_dir = data_dir
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)
        self.model_cache_dir = os.path.abspath(os.path.join(self.db_dir, "..", ".hf_model_cache"))
        os.makedirs(self.model_cache_dir, exist_ok=True)

        # Force model/cache paths into project-local directories to avoid Windows user-path issues.
        os.environ["HF_HOME"] = self.model_cache_dir
        os.environ["HF_HUB_CACHE"] = os.path.join(self.model_cache_dir, "hub")
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(self.model_cache_dir, "transformers")
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(self.model_cache_dir, "sentence_transformers")
        os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)
        os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
        os.makedirs(os.environ["SENTENCE_TRANSFORMERS_HOME"], exist_ok=True)

        self.materialized_model_dir = os.path.join(self.model_cache_dir, "materialized_model")
        os.makedirs(self.materialized_model_dir, exist_ok=True)

        # Materialize the full model repository locally so sentence-transformers can read real files.
        snapshot_download(
            repo_id=EMBEDDING_MODEL,
            local_dir=self.materialized_model_dir,
            local_dir_use_symlinks=False,
        )

        print(f"[{NAME_OF_WEBSITE} INIT] Downloading and spinning up clinical weights: {EMBEDDING_MODEL}...")
        # Initialize PubMedBERT embeddings. Run on GPU if available, else fallback to CPU.
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.materialized_model_dir,
            cache_folder=self.model_cache_dir,
            model_kwargs={"device": "cpu"},  # Change to 'cuda' or 'mps' if hardware allows
            encode_kwargs={"normalize_embeddings": True},
        )

    def load_child_documents(self, filename: str = "child_documents.pkl") -> List[Document]:
        """Loads the fragmented highly precise text chunks from Day 2."""
        import_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(import_path):
            raise FileNotFoundError(f"[{NAME_OF_WEBSITE} ERROR] Child documents missing at {import_path}.")

        print(f"[{NAME_OF_WEBSITE} SYSTEM] Importing target search chunks from {import_path}...")
        return pd.read_pickle(import_path).tolist()

    def batch_vector_ingestion(self, documents: List[Document], batch_size: int = 500):
        """
        Pushes thousands of vector documents to ChromaDB in windows.
        Prevents Out-Of-Memory (OOM) errors and thread pooling blockages.
        """
        total_docs = len(documents)
        if total_docs == 0:
            raise ValueError(f"[{NAME_OF_WEBSITE} ERROR] No documents provided for ingestion.")

        print(f"[{NAME_OF_WEBSITE} CORE] Beginning incremental vector migration for {total_docs:,} blocks...")

        # Initialize the baseline persistent db connection with the initial batch
        initial_batch = documents[0:batch_size]
        vector_db = Chroma.from_documents(
            documents=initial_batch,
            embedding=self.embeddings,
            persist_directory=self.db_dir,
        )

        # Loop over remaining chunks safely
        for i in tqdm(range(batch_size, total_docs, batch_size), desc="Ingesting Vectors to ChromaDB"):
            batch = documents[i:i + batch_size]
            vector_db.add_documents(documents=batch)

        print(f"[{NAME_OF_WEBSITE} SUCCESS] Vector matrix fully committed to disk storage at {self.db_dir}")

    def build_and_serialize_lexical_index(self, documents: List[Document], filename: str = "bm25_index.pkl"):
        """
        Builds the keyword match system (BM25) to run alongside our semantic vector engine.
        Serializes to disk via pickle to completely avoid runtime recalculation bottlenecks.
        """
        print(f"[{NAME_OF_WEBSITE} CORE] Compiling lexical BM25 database layers...")
        bm25_retriever = BM25Retriever.from_documents(documents)

        export_path = os.path.join(self.data_dir, filename)
        with open(export_path, "wb") as f:
            pickle.dump(bm25_retriever, f)

        print(f"[{NAME_OF_WEBSITE} SUCCESS] Lexical tracker indexed and written to {export_path}")


if __name__ == "__main__":
    # Test path configurations if running locally within the sub-module directory
    manager = EnterpriseVectorStoreManager(data_dir="../../data/processed", db_dir="../../chroma_db")
    docs = manager.load_child_documents()
    manager.batch_vector_ingestion(docs)
    manager.build_and_serialize_lexical_index(docs)
