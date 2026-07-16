from app.config.config import Config
from app.core.errors import BackendServiceError
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatOpenAI


def initialize_llm():
    if not Config.OPENAI_API_KEY:
        raise BackendServiceError(
            code="openai_auth_error",
            message="OPENAI_API_KEY is not set.",
            service="openai",
            model=Config.LLM_NAME,
        )

    try:
        return ChatOpenAI(
            model=Config.LLM_NAME,
            api_key=Config.OPENAI_API_KEY,
            temperature=0.4,
            verbose=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
    except ImportError as exc:
        raise BackendServiceError(
            code="openai_sdk_missing",
            message="OpenAI SDK is not installed. Install project dependencies and retry.",
            service="openai",
            model=Config.LLM_NAME,
        ) from exc
    except Exception as exc:
        raise BackendServiceError(
            code="openai_client_error",
            message="Failed to initialize the OpenAI chat client.",
            service="openai",
            model=Config.LLM_NAME,
        ) from exc
