import os
import uuid
from datetime import datetime
from config import Config
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed.embedding import DefaultEmbedding

class MemoryManager:
    def __init__(self):
        # Initialize Qdrant client with local storage
        self.client = QdrantClient(path=Config.QDRANT_PATH)
        self.collection_name = "research_memory"
        self.embedding_model = DefaultEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # Ensure collection exists
        self._init_collection()

    def _init_collection(self):
        """Initialize the collection if it doesn't exist."""
        try:
            self.client.get_collection(self.collection_name)
            print(f"✅ Collection '{self.collection_name}' already exists.")
        except Exception as e:
            # Collection doesn't exist, create it
            print(f"📝 Creating collection '{self.collection_name}'...")
            vector_size = 384  # BAAI/bge-small-en-v1.5 embedding dimension
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            print(f"✅ Collection '{self.collection_name}' created successfully.")

    def add_memory(self, text: str, metadata: dict = None):
        """
        Save a text blob (e.g., report, interaction) to memory.
        """
        if metadata is None:
            metadata = {}
        
        metadata["timestamp"] = str(datetime.now())
        
        try:
            # Generate embedding
            embeddings = list(self.embedding_model.embed([text]))
            embedding = embeddings[0] if embeddings else [0.0] * 384
            
            point_id = str(uuid.uuid4()).replace("-", "")[:20]
            
            # Add to collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=int(point_id[:10], 16) % (2**31 - 1),  # Convert to valid int ID
                        vector=embedding,
                        payload={"text": text, **metadata}
                    )
                ]
            )
            print(f"✅ Saved to memory: {text[:50]}...")
        except Exception as e:
            print(f"⚠️ Error saving to memory: {e}")

    def get_context(self, query: str, n_results: int = 2):
        """
        Retrieve relevant context for a query.
        """
        try:
            # Generate query embedding
            embeddings = list(self.embedding_model.embed([query]))
            query_embedding = embeddings[0] if embeddings else [0.0] * 384
            
            # Search collection using scroll for safety
            try:
                # Try new API first
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=n_results,
                    score_threshold=0.0
                )
                documents = [hit.payload.get("text", "") for hit in search_result if hit.payload]
            except AttributeError:
                # Fallback for older Qdrant versions or empty collections
                documents = []
            
            return documents if documents else []
        except Exception as e:
            print(f"⚠️ Error retrieving context: {e}")
            return []

# Singleton instance
memory = MemoryManager()
