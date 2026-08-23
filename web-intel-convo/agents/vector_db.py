import logging
import time
import uuid
import hashlib
from datetime import datetime, timedelta
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import config
import os

logger = logging.getLogger("VectorDB")

class VectorDBAgent:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=config.db_persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name="web_intel_knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        os.environ["GOOGLE_API_KEY"] = config.gemini_api_key
        self.embedding_model = GoogleGenerativeAIEmbeddings(model=config.embeddings_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len
        )

    def _generate_document_hash(self, text: str) -> str:
        """Generates a hash for deduplication."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def process_and_index(self, documents: list[dict]) -> dict:
        """Chunks, embeds, deduplicates, and upserts documents into VectorDB."""
        logger.info(f"Processing and indexing {len(documents)} documents...")
        
        report = {
            "total_documents_processed": len(documents),
            "chunks_created": 0,
            "chunks_inserted": 0,
            "duplicates_skipped": 0,
            "timestamp": datetime.now().isoformat()
        }

        # Check existing hashes for deduplication
        existing_data = self.collection.get(include=["metadatas"])
        existing_hashes = set()
        if existing_data and existing_data["metadatas"]:
            for meta in existing_data["metadatas"]:
                if meta and "doc_hash" in meta:
                    existing_hashes.add(meta["doc_hash"])

        current_date = datetime.now().isoformat()

        for doc in documents:
            doc_hash = self._generate_document_hash(doc["content"])
            if doc_hash in existing_hashes:
                logger.info(f"Skipping duplicate document from URL: {doc['url']}")
                report["duplicates_skipped"] += 1
                continue

            chunks = self.text_splitter.split_text(doc["content"])
            report["chunks_created"] += len(chunks)

            ids = []
            metadatas = []
            documents_text = []

            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                ids.append(chunk_id)
                documents_text.append(chunk)
                
                metadata = {
                    "url": doc["url"],
                    "title": doc["title"],
                    "doc_hash": doc_hash,
                    "date_added": current_date,
                    "chunk_index": i
                }
                metadatas.append(metadata)

            if chunks:
                # Generate embeddings using Google API with rate limit handling
                encoded_embeddings = []
                batch_size = 5
                for j in range(0, len(documents_text), batch_size):
                    batch_texts = documents_text[j:j+batch_size]
                    try:
                        batch_embeds = self.embedding_model.embed_documents(batch_texts)
                        encoded_embeddings.extend(batch_embeds)
                        time.sleep(1.5) # Prevent aggressive rate limit hitting
                    except Exception as e:
                        logger.warning(f"Embedding error: {e}. Sleeping 45s due to possible rate limit...")
                        time.sleep(45)
                        batch_embeds = self.embedding_model.embed_documents(batch_texts)
                        encoded_embeddings.extend(batch_embeds)
                
                try:
                    self.collection.upsert(
                        ids=ids,
                        embeddings=encoded_embeddings,
                        metadatas=metadatas,
                        documents=documents_text
                    )
                    report["chunks_inserted"] += len(chunks)
                    existing_hashes.add(doc_hash)
                except Exception as e:
                    logger.error(f"Failed to insert chunks into ChromaDB: {e}")

        logger.info(f"Indexing complete. Inserted {report['chunks_inserted']} chunks. Skipped {report['duplicates_skipped']} duplicates.")
        return report

    def purge_expired(self):
        """Purges documents older than retention_days."""
        logger.info(f"Purging documents older than {config.retention_days} days...")
        try:
            existing_data = self.collection.get(include=["metadatas"])
            if not existing_data or not existing_data["metadatas"]:
                return

            cutoff_date = datetime.now() - timedelta(days=config.retention_days)
            ids_to_delete = []

            for i, meta in enumerate(existing_data["metadatas"]):
                if meta and "date_added" in meta:
                    doc_date = datetime.fromisoformat(meta["date_added"])
                    if doc_date < cutoff_date:
                        ids_to_delete.append(existing_data["ids"][i])

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Purged {len(ids_to_delete)} expired chunks.")
            else:
                logger.info("No expired chunks found.")
                
        except Exception as e:
            logger.error(f"Error purging expired documents: {e}")
