import os
import sys

# Ensure Python can find the src module
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.ingestion.parser import EnterpriseClinicalParser


def execute_day_1_pipeline():
    print("===================================================")
    print("   STARTING DAY 1: MASSIVE DATA INGESTION PIPELINE ")
    print("===================================================")

    # 1. Initialize the parser (pointing output to the data/processed folder)
    medical_parser = EnterpriseClinicalParser(output_dir="./data/processed")

    # 2. Trigger the massive download (Requires internet connection)
    medical_parser.download_full_corpus()

    # 3. Clean, structure, and convert thousands of rows into LLM-ready Documents
    medical_parser.sanitize_and_structure()

    # 4. Serialize to disk
    medical_parser.export_processed_data(filename="full_clinical_corpus.pkl")

    print("===================================================")
    print(" DAY 1 COMPLETE. PROCESSED DATA READY FOR CHUNKING ")
    print("===================================================")


if __name__ == "__main__":
    execute_day_1_pipeline()
