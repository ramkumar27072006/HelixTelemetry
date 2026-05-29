import os
import uuid
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# GLOBAL CONFIGURATION
NAME_OF_WEBSITE = "HelixTelemetry"


class EnterpriseHierarchicalSplitter:
    def __init__(self, input_dir: str = "../../data/processed", output_dir: str = "../../data/processed"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Define the structural boundaries
        # Parent chunks ensure the complete clinical case is readable
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Child chunks ensure semantic search pinpoint accuracy
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_corpus(self, filename: str = "full_clinical_corpus.pkl") -> List[Document]:
        """Loads the raw document objects generated during Day 1."""
        import_path = os.path.join(self.input_dir, filename)
        if not os.path.exists(import_path):
            raise FileNotFoundError(
                f"[{NAME_OF_WEBSITE} ERROR] Missing corpus file at {import_path}. Did Day 1 run successfully?"
            )

        print(f"[{NAME_OF_WEBSITE} SYSTEM] Loading structured corpus from {import_path}...")
        return pd.read_pickle(import_path).tolist()

    def process_hierarchical_splits(self, base_documents: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        Deconstructs base medical documents into connected Parent and Child structures.
        Maintains rigorous ID lineage for backend mapping.
        """
        parent_docs = []
        child_docs = []

        print(
            f"[{NAME_OF_WEBSITE} COMPILING] Commencing multi-tier text segmentation on {len(base_documents):,} base files..."
        )

        for base_doc in tqdm(base_documents, desc="Executing Hierarchical Splits"):
            # 1. Segment base text into distinct Parent Chunks
            parents = self.parent_splitter.split_text(base_doc.page_content)

            for parent_text in parents:
                # Generate an immutable tracking ID for this specific parent block
                parent_id = str(uuid.uuid4())

                parent_metadata = base_doc.metadata.copy()
                parent_metadata["parent_id"] = parent_id
                parent_metadata["chunk_type"] = "parent"

                parent_document = Document(page_content=parent_text, metadata=parent_metadata)
                parent_docs.append(parent_document)

                # 2. Sub-segment this specific Parent Block into precise Child Chunks
                children = self.child_splitter.split_text(parent_text)

                for child_text in children:
                    child_metadata = base_doc.metadata.copy()
                    # Tie the child directly to its biological parent ID
                    child_metadata["parent_id"] = parent_id
                    child_metadata["child_id"] = str(uuid.uuid4())
                    child_metadata["chunk_type"] = "child"

                    child_document = Document(page_content=child_text, metadata=child_metadata)
                    child_docs.append(child_document)

        return parent_docs, child_docs

    def serialize_chunks(self, parents: List[Document], children: List[Document]):
        """Saves both document layers cleanly to disk to prevent reprocessing overhead."""
        parent_path = os.path.join(self.output_dir, "parent_chunks.pkl")
        child_path = os.path.join(self.output_dir, "child_documents.pkl")

        print(f"[{NAME_OF_WEBSITE} EXPORT] Writing generated blocks to storage...")
        pd.Series(parents).to_pickle(parent_path)
        pd.Series(children).to_pickle(child_path)

        print(f"[{NAME_OF_WEBSITE} SUCCESS] Structural compilation complete.")
        print(f"-> Created Parent Units (Context Layer): {len(parents):,}")
        print(f"-> Created Child Units (Search Vector Layer): {len(children):,}")


if __name__ == "__main__":
    # Internal module testing validation
    splitter = EnterpriseHierarchicalSplitter(
        input_dir="../../data/processed", output_dir="../../data/processed"
    )
    docs = splitter.load_corpus()
    p, c = splitter.process_hierarchical_splits(docs)
    splitter.serialize_chunks(p, c)
