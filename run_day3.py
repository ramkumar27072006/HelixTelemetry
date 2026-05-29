import os
import sys

# Ensure Python can find the src module
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.chdir(os.path.dirname(__file__))

from src.ingestion.vector_store import EnterpriseVectorStoreManager


def execute_day_3_pipeline():
    print("===================================================")
    print("  STARTING DAY 3: VECTOR CORE MATRIX EMBEDDING ")
    print("===================================================")

    # Define absolute persistence directories
    processed_data_directory = "./data/processed"
    chroma_persistence_directory = "./chroma_db"

    # Initialize the engine
    store_manager = EnterpriseVectorStoreManager(
        data_dir=processed_data_directory,
        db_dir=chroma_persistence_directory,
    )

    # 1. Fetch data arrays prepared during the chunking phase
    child_docs = store_manager.load_child_documents(filename="child_documents.pkl")

    # 2. Ingest vectors through safety windows into local disk-space
    store_manager.batch_vector_ingestion(documents=child_docs, batch_size=500)

    # 3. Build parallel keyword retriever layer for structural keywords
    store_manager.build_and_serialize_lexical_index(documents=child_docs, filename="bm25_index.pkl")

    print("===================================================")
    print("   DAY 3 COMPLETE: DATA TIERS LOADED AND INDEXED   ")
    print("===================================================")


if __name__ == "__main__":
    execute_day_3_pipeline()
