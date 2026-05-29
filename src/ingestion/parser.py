import os
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm
from langchain_core.documents import Document

# GLOBAL CONFIGURATION
NAME_OF_WEBSITE = "HelixTelemetry"
DATASET_ID = "medalpaca/medical_meadow_medqa"


class EnterpriseClinicalParser:
    def __init__(self, output_dir: str = "../../data/processed"):
        """Initializes the parser and ensures output directories exist."""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.cache_dir = os.path.abspath(os.path.join(self.output_dir, "..", ".hf_cache"))
        os.makedirs(self.cache_dir, exist_ok=True)
        self.full_dataframe = None
        self.documents = []

    def download_full_corpus(self) -> pd.DataFrame:
        """
        Connects to HuggingFace and pulls the ENTIRE medical dataset.
        No sampling. No limits.
        """
        print(f"[{NAME_OF_WEBSITE} INIT] Commencing full dataset download from {DATASET_ID}...")

        try:
            # Primary path: use datasets loader for direct split access
            dataset = load_dataset(DATASET_ID, split="train", cache_dir=self.cache_dir)

            # Convert to a massive Pandas DataFrame for rapid vectorized cleaning
            self.full_dataframe = dataset.to_pandas()
        except Exception as exc:
            print(
                f"[{NAME_OF_WEBSITE} WARN] datasets.load_dataset failed: {exc}"
            )
            print(
                f"[{NAME_OF_WEBSITE} FALLBACK] Switching to direct parquet download from Hugging Face Hub..."
            )
            self.full_dataframe = self._download_via_hub_files()

        total_rows = len(self.full_dataframe)
        print(f"[{NAME_OF_WEBSITE} SUCCESS] Downloaded {total_rows:,} raw clinical records.")
        return self.full_dataframe

    def _download_via_hub_files(self) -> pd.DataFrame:
        """Fallback path that downloads the dataset files directly from the Hub."""
        api = HfApi()
        repo_files = api.list_repo_files(repo_id=DATASET_ID, repo_type="dataset")
        parquet_files = [file for file in repo_files if file.endswith(".parquet")]
        json_files = [file for file in repo_files if file.endswith(".json")]

        if parquet_files:
            local_repo_dir = snapshot_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                allow_patterns=parquet_files,
                local_dir=os.path.join(self.cache_dir, "snapshot"),
                local_dir_use_symlinks=False,
            )
            dataframes = []
            for parquet_file in tqdm(parquet_files, desc="Downloading parquet shards"):
                local_path = os.path.join(local_repo_dir, parquet_file)
                dataframes.append(pd.read_parquet(local_path))

            return pd.concat(dataframes, ignore_index=True)

        if json_files:
            # This dataset currently publishes a single JSON artifact.
            local_repo_dir = snapshot_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                allow_patterns=[json_files[0]],
                local_dir=os.path.join(self.cache_dir, "snapshot"),
                local_dir_use_symlinks=False,
            )
            local_path = os.path.join(local_repo_dir, json_files[0])
            return pd.read_json(local_path)

        raise RuntimeError(
            f"No parquet or json files found in dataset repo '{DATASET_ID}'."
        )

    def sanitize_and_structure(self):
        """
        Cleans the dataset of NaN values and structures it into LangChain Documents
        with high-density metadata for traceable RAG retrieval.
        """
        if self.full_dataframe is None:
            raise ValueError("Dataset not loaded. Run download_full_corpus() first.")

        print(f"[{NAME_OF_WEBSITE} SYSTEM] Sanitizing data and dropping null values...")
        # Drop any corrupted or empty rows
        df_clean = self.full_dataframe.dropna(subset=["input", "output"]).copy()

        print(
            f"[{NAME_OF_WEBSITE} SYSTEM] Compiling {len(df_clean):,} rows into LangChain Document objects..."
        )

        # Wrap in tqdm for a progress bar so you can monitor the 16-hour grind
        for index, row in tqdm(
            df_clean.iterrows(), total=len(df_clean), desc="Structuring Documents"
        ):
            # Combine the medical context/question and the clinical answer
            clinical_text = (
                f"Clinical Context/Query: {row.get('input', '')}\n"
                f"Clinical Decision/Answer: {row.get('output', '')}"
            )

            # Pack rich metadata so the UI can cite its sources later
            metadata = {
                "source_dataset": DATASET_ID,
                "record_id": f"medqa_{index}",
                "instruction_type": row.get("instruction", "medical_qa"),
                "system_tag": NAME_OF_WEBSITE,
            }

            # Create the LangChain Document
            doc = Document(page_content=clinical_text, metadata=metadata)
            self.documents.append(doc)

    def export_processed_data(self, filename: str = "full_clinical_corpus.pkl"):
        """Serializes the processed documents so Day 2 can instantly load them."""
        export_path = os.path.join(self.output_dir, filename)

        # We save as a pickle file to preserve the LangChain Document objects perfectly
        pd.Series(self.documents).to_pickle(export_path)
        print(f"[{NAME_OF_WEBSITE} SUCCESS] Entire structured corpus saved to {export_path}")
        print(
            f"[{NAME_OF_WEBSITE} METRICS] Total Viable Clinical Documents: {len(self.documents):,}"
        )


if __name__ == "__main__":
    # This allows testing the parser directly
    parser = EnterpriseClinicalParser(output_dir="../../data/processed")
    parser.download_full_corpus()
    parser.sanitize_and_structure()
    parser.export_processed_data()
