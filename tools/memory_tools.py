# tools/memory_tools.py
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from langchain_core.schema import Document as LC_Document
from config import Config
from fastembed.embedding import DefaultEmbedding

# Use fastembed DefaultEmbedding for local embeddings
embeddings = DefaultEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Initialize Qdrant client
client = QdrantClient(path=Config.QDRANT_PATH)

# Initialize Qdrant vectorstore with LangChain
vectorstore = Qdrant(
    client=client,
    collection_name="research_history",
    embeddings=embeddings
)

def save_research_to_memory(query: str, report: str):
    """Saves the final research report for future context."""
    doc = Document(
        page_content=report,
        metadata={"query": query, "type": "final_report"}
    )
    vectorstore.add_documents([doc])
    # Note: Qdrant persists automatically, no need for explicit persist()


def get_vectorstore():
    """Return the initialized Qdrant vectorstore backed by FastEmbed embeddings."""
    return vectorstore


def get_relevant_context(query: str, k: int = 2):
    """Convenience wrapper to return raw contexts from the vectorstore."""
    docs = vectorstore.similarity_search(query, k=k)
    if not docs:
        return []
    return docs


__all__ = [
    "embeddings",
    "client",
    "vectorstore",
    "save_research_to_memory",
    "retrieve_past_context",
    "get_vectorstore",
    "get_relevant_context",
]

def retrieve_past_context(query: str):
    """Retrieves relevant history to avoid repeating basics."""
    # Search for the top 2 most relevant past research pieces
    docs = vectorstore.similarity_search(query, k=2)
    if not docs:
        return "No prior research history found."
    
    context = "\n---\n".join([d.page_content[:500] + "..." for d in docs])
    return f"Relevant Past Research Found:\n{context}"