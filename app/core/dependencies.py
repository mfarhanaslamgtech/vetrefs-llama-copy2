from functools import lru_cache

from app.chatbot.chatbot import Chatbot
from app.database.db_operations import DatabaseHandler


@lru_cache(maxsize=1)
def get_database_handler() -> DatabaseHandler:
    """Return a process-local MongoDB handler."""
    return DatabaseHandler()


@lru_cache(maxsize=1)
def get_chatbot() -> Chatbot:
    """Return a process-local chatbot service."""
    return Chatbot()
