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
    LLM_NAME = os.environ['LLM_NAME']

    # OpenAI configuration for embeddings
    OPENAI_EMBEDDING_MODEL = os.environ['OPENAI_EMBEDDING_MODEL']
    EMBEDDING_MODEL_NAME = os.environ['EMBEDDING_MODEL_NAME']

    # Pinecone configuration for vector storage
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = os.environ['PINECONE_INDEX_NAME']
    PINECONE_NAMESPACE = os.environ['PINECONE_NAMESPACE']
