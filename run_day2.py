import os
import sys

# Ensure Python can find the src module
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.ingestion.splitter import EnterpriseHierarchicalSplitter


def execute_day_2_pipeline():
    print("===================================================")
    print("  STARTING DAY 2: HIERARCHICAL SEGMENTATION PIPELINE ")
    print("===================================================")

    # Initialize the architecture engine mapping paths to data architecture
    splitter = EnterpriseHierarchicalSplitter(
        input_dir="./data/processed",
        output_dir="./data/processed",
    )

    # 1. Fetch the master corpus array
    base_docs = splitter.load_corpus(filename="full_clinical_corpus.pkl")

    # 2. Run the chunk segmentation loops
    parents, children = splitter.process_hierarchical_splits(base_documents=base_docs)

    # 3. Write structured tokens back to the local database staging area
    splitter.serialize_chunks(parents=parents, children=children)

    print("===================================================")
    print("  DAY 2 COMPLETE: VECTOR DATA LAYERS READY TO EMBED ")
    print("===================================================")


if __name__ == "__main__":
    execute_day_2_pipeline()
