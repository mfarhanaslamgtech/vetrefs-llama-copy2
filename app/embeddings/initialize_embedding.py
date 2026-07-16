from app.config.config import Config
from app.core.errors import BackendServiceError
from langchain_community.embeddings import OpenAIEmbeddings

try:
    from langchain_pinecone import PineconeVectorStore
except Exception:  # pragma: no cover - depends on installed dependencies
    PineconeVectorStore = None


def initialize_embeddings():
    """Initialize the embedding model and Pinecone vector store."""
    if PineconeVectorStore is None:
        raise BackendServiceError(
            code="pinecone_sdk_missing",
            message="Pinecone support is not installed. Install project dependencies and retry.",
            service="pinecone",
            model=Config.EMBEDDING_MODEL_NAME,
        )

    if not Config.PINECONE_API_KEY:
        raise BackendServiceError(
            code="pinecone_auth_error",
            message="PINECONE_API_KEY is not set.",
            service="pinecone",
            model=Config.EMBEDDING_MODEL_NAME,
        )

    if not Config.OPENAI_API_KEY:
        raise BackendServiceError(
            code="openai_auth_error",
            message="OPENAI_API_KEY is not set.",
            service="openai",
            model=Config.OPENAI_EMBEDDING_MODEL,
        )

    embeddings = OpenAIEmbeddings(
        model=Config.OPENAI_EMBEDDING_MODEL,
        api_key=Config.OPENAI_API_KEY,
    )
    vectordb = PineconeVectorStore.from_existing_index(
        index_name=Config.PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=Config.PINECONE_NAMESPACE,
    )
    return vectordb
