from typing import Any, Dict, Tuple


from app.config.config import Config


def check_openai_health() -> Tuple[bool, Dict[str, Any]]:
    try:
        import openai  # noqa: F401
        sdk_available = True
    except Exception:
        sdk_available = False

    api_key_present = bool(Config.OPENAI_API_KEY)
    is_healthy = sdk_available and api_key_present
    return is_healthy, {
        "service": "openai",
        "status": "up" if is_healthy else "down",
        "model": Config.LLM_NAME,
        "message": (
            None
            if is_healthy
            else (
                "OpenAI SDK is not installed"
                if not sdk_available
                else "OPENAI_API_KEY is not set"
            )
        ),
    }


