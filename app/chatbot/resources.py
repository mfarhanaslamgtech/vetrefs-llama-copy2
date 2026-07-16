import asyncio
import json
import logging
import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.chatbot.chatbot import Chatbot
from app.config.config import Config
from app.core.dependencies import get_chatbot, get_database_handler
from app.core.errors import BackendServiceError
from app.database.db_operations import DatabaseHandler
from app.logs.logger import configure_logging

logging = configure_logging()
router = APIRouter(prefix="/v1/chatbot", tags=["chatbot"])


class ChatRequest(BaseModel):
    user_id: int = Field(..., ge=1, description="Unique user identifier")
    chat_session_id: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional chat session identifier. If omitted, a new one is created.",
    )
    question: str = Field(..., min_length=1, description="User question")

    @field_validator("chat_session_id", mode="before")
    @classmethod
    def normalize_chat_session_id(cls, value):
        if value in (None, "", "null", "None"):
            return None
        return value

    @field_validator("question", mode="before")
    @classmethod
    def normalize_question(cls, value):
        if isinstance(value, str):
            value = value.strip()
        return value


class RenameChatTitleRequest(BaseModel):
    user_id: int = Field(..., ge=1)
    chat_session_id: int = Field(..., ge=1)
    chat_title: str = Field(..., min_length=1, max_length=50)


_OPENAI_AUTH_ERROR_PATTERN = re.compile(
    r"api key|authentication|unauthorized|invalid api key|openai api key",
    re.IGNORECASE,
)
_OPENAI_RATE_LIMIT_PATTERN = re.compile(
    r"rate limit|too many requests",
    re.IGNORECASE,
)


def _is_openai_auth_error(exc: Exception) -> bool:
    return bool(_OPENAI_AUTH_ERROR_PATTERN.search(str(exc)))


def _is_openai_rate_limit_error(exc: Exception) -> bool:
    return bool(_OPENAI_RATE_LIMIT_PATTERN.search(str(exc)))


def _is_openai_sdk_missing_error(exc: Exception) -> bool:
    error_text = str(exc).lower()
    return isinstance(exc, (ImportError, ModuleNotFoundError)) or "could not import openai python package" in error_text


def _raise_model_http_error(exc: Exception) -> None:
    if _is_openai_sdk_missing_error(exc):
        raise BackendServiceError(
            code="openai_sdk_missing",
            message="OpenAI SDK is not installed. Install project dependencies and retry.",
            service="openai",
            model=Config.LLM_NAME,
        ) from exc

    if _is_openai_auth_error(exc):
        raise BackendServiceError(
            code="openai_auth_error",
            message="OpenAI API key is missing or invalid.",
            service="openai",
            model=Config.LLM_NAME,
        ) from exc

    if _is_openai_rate_limit_error(exc):
        raise BackendServiceError(
            code="openai_rate_limited",
            message="OpenAI rate limit reached.",
            service="openai",
            model=Config.LLM_NAME,
        ) from exc


@router.post("")
async def answer_chat(
    request_data: ChatRequest,
    chatbot: Chatbot = Depends(get_chatbot),
    db_handler: DatabaseHandler = Depends(get_database_handler),
):
    try:
        chat_session_id = request_data.chat_session_id
        if not isinstance(chat_session_id, int):
            chat_session_id = db_handler.generate_chat_session_id(request_data.user_id)

        try:
            chatbot_response = chatbot.answer_question(
                request_data.user_id,
                chat_session_id,
                request_data.question,
            )
        except Exception as exc:
            logging.error(
                "Chat completion failed (%s on %s): %s",
                exc.__class__.__name__,
                Config.LLM_NAME,
                str(exc),
                exc_info=True,
            )
            _raise_model_http_error(exc)
            raise

        generated_chat_title = db_handler.generate_chat_title(
            request_data.user_id,
            chat_session_id,
        )
        db_handler.save_chat_title(
            request_data.user_id,
            chat_session_id,
            generated_chat_title,
        )

        if isinstance(chatbot_response, dict):
            answer = chatbot_response.get("answer", "")
            sources = chatbot_response.get("sources", [])
        else:
            answer = chatbot_response
            sources = []

        payload = {"answer": answer, "chat_session_id": chat_session_id}
        if sources:
            payload["sources"] = sources
        return payload
    except HTTPException:
        raise
    except BackendServiceError:
        raise
    except Exception:
        logging.error("An error occurred while processing a request", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request.",
        )


@router.post("/stream")
async def answer_chat_stream(
    request_data: ChatRequest,
    chatbot: Chatbot = Depends(get_chatbot),
    db_handler: DatabaseHandler = Depends(get_database_handler),
):
    async def event_stream():
        def encode(event):
            return json.dumps(event, ensure_ascii=False) + "\n"

        chat_session_id = request_data.chat_session_id
        if not isinstance(chat_session_id, int):
            chat_session_id = db_handler.generate_chat_session_id(request_data.user_id)

        yield encode({"type": "session", "chat_session_id": chat_session_id})
        yield encode({"type": "status", "message": "Retrieving context"})

        try:
            chatbot_response = chatbot.answer_question(
                request_data.user_id,
                chat_session_id,
                request_data.question,
            )

            generated_chat_title = db_handler.generate_chat_title(
                request_data.user_id,
                chat_session_id,
            )
            db_handler.save_chat_title(
                request_data.user_id,
                chat_session_id,
                generated_chat_title,
            )

            if isinstance(chatbot_response, dict):
                answer = chatbot_response.get("answer", "")
                sources = chatbot_response.get("sources", [])
            else:
                answer = chatbot_response
                sources = []

            yield encode({"type": "sources", "sources": sources})
            yield encode({"type": "status", "message": "Generating answer"})

            words = answer.split(" ")
            for index in range(0, len(words), 4):
                chunk = " ".join(words[index:index + 4])
                if index + 4 < len(words):
                    chunk += " "
                yield encode({"type": "chunk", "text": chunk})
                await asyncio.sleep(0.015)

            yield encode({
                "type": "done",
                "answer": answer,
                "chat_session_id": chat_session_id,
                "sources": sources,
            })
        except Exception as exc:
            logging.error(
                "Streaming chat completion failed (%s on %s): %s",
                exc.__class__.__name__,
                Config.LLM_NAME,
                str(exc),
                exc_info=True,
            )
            yield encode({
                "type": "error",
                "error": {
                    "code": "stream_error",
                    "message": str(exc) or "An error occurred while processing your request.",
                    "service": "api",
                },
            })

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@router.get("/history/{user_id}/{chat_session_id}")
def get_chat_history(
    user_id: int,
    chat_session_id: int,
    db_handler: DatabaseHandler = Depends(get_database_handler),
):
    try:
        return db_handler.get_chat_history(user_id, chat_session_id)
    except Exception as exc:
        logging.error("Chat history not found: %s", str(exc))
        raise HTTPException(status_code=404, detail="Chat history not found!.")


@router.get("/chat_list/{user_id}")
def get_chat_list(
    user_id: int,
    db_handler: DatabaseHandler = Depends(get_database_handler),
):
    try:
        chat_sessions = db_handler.get_chat_sessions(user_id)
        chat_sessions_with_titles = []

        for chat_session_id in chat_sessions:
            chat_title = db_handler.get_chat_title(user_id, chat_session_id)
            chat_sessions_with_titles.append(
                {"chat_session_id": chat_session_id, "chat_title": chat_title}
            )

        if not chat_sessions_with_titles:
            raise HTTPException(status_code=404, detail="Chat list not found!!.")

        return {"chat_sessions": chat_sessions_with_titles}
    except HTTPException:
        raise
    except Exception as exc:
        logging.error("Chat list not found!: %s", str(exc))
        raise HTTPException(status_code=404, detail="Chat list not found!!.")


@router.put("/rename_chat_title")
def rename_chat_title(
    request_data: RenameChatTitleRequest,
    db_handler: DatabaseHandler = Depends(get_database_handler),
):
    try:
        if not db_handler.does_chat_title_exist(
            request_data.user_id,
            request_data.chat_session_id,
            request_data.chat_title,
        ):
            raise HTTPException(status_code=404, detail="Chat session does not exist!")

        db_handler.rename_chat_title(
            request_data.user_id,
            request_data.chat_session_id,
            request_data.chat_title,
        )
        return {"success": "Chat title updated successfully!"}
    except HTTPException:
        raise
    except Exception as exc:
        logging.error("An unexpected error occured: %s", str(exc))
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occured.",
        )


@router.delete("/delete_chat_session/{user_id}/{chat_session_id}")
def delete_chat_session(
    user_id: int,
    chat_session_id: int,
    db_handler: DatabaseHandler = Depends(get_database_handler),
):
    try:
        if not db_handler.check_session_existence(user_id, chat_session_id):
            raise HTTPException(status_code=404, detail="Chat does not exist!")

        db_handler.delete_chat_session(user_id, chat_session_id)
        return {"message": "Deleted successfully"}
    except HTTPException:
        raise
    except Exception as exc:
        logging.error("An unexpected error occured: %s", str(exc))
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occured.",
        )


@router.delete("/delete_all_chats/{user_id}")
def delete_all_chats(
    user_id: int,
    db_handler: DatabaseHandler = Depends(get_database_handler),
):
    try:
        if not db_handler.check_user_existence(user_id):
            raise HTTPException(status_code=404, detail="Data does not exist")

        db_handler.delete_all_chats(user_id)
        return {"message": "Deleted successfully!"}
    except HTTPException:
        raise
    except Exception as exc:
        logging.error("An error occured while processing your request :%s", str(exc))
        return {"message": "An error occured while processing your request"}
