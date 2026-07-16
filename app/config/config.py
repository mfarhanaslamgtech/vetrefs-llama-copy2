import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / '.env', override=True)


class Config:
    # MongoDB configuration
    MONGO_URI = os.environ.get('MONGO_URI')
    MONGO_DB_NAME = str(os.environ.get('MONGO_DB_NAME')).strip()

    # Redis configuration for rate limiting
    REDIS_URI = os.environ.get('REDIS_URI')

    # OpenAI configuration for the chat model
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    LLM_NAME = os.environ.get('LLM_NAME', 'gpt-4o-mini')

    # OpenAI configuration for embeddings
    OPENAI_EMBEDDING_MODEL = os.environ.get('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
    EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL_NAME', OPENAI_EMBEDDING_MODEL)

    # Pinecone configuration for vector storage
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME', 'vetrefs-llama')
    PINECONE_NAMESPACE = os.environ.get('PINECONE_NAMESPACE', 'vetref')
