import sys
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.chatbot.resources import router as chatbot_router
from app.config.config import Config
from app.core.errors import BackendServiceError
from app.core.health import check_openai_health


def create_app() -> FastAPI:
    app = FastAPI(title="VetPedia", version="1.0.0")
    templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

    @app.middleware("http")
    async def add_request_metrics(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.exception_handler(BackendServiceError)
    async def backend_service_error_handler(request: Request, exc: BackendServiceError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail.__dict__},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed.",
                    "service": "api",
                    "details": exc.errors(),
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "http_error",
                    "message": str(exc.detail),
                    "service": "api",
                }
            },
        )

    @app.get("/", response_class=HTMLResponse)
    def welcome(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/health")
    def health():
        openai_ok, openai_info = check_openai_health()
        payload = {
            "status": "ok" if openai_ok else "degraded",
            "service": "vetpedia-api",
            "checks": {
                "openai": openai_info,
            },
        }
        if not openai_ok:
            return JSONResponse(status_code=503, content=payload)
        return payload

    @app.get("/config")
    def config_summary():
        return {
            "service": "vetpedia-api",
            "llm_provider": "openai",
            "llm_model_name": Config.LLM_NAME,
            "openai_api_key_configured": bool(Config.OPENAI_API_KEY),
            "embedding_model_name": Config.OPENAI_EMBEDDING_MODEL,
            "pinecone_index_name": Config.PINECONE_INDEX_NAME,
            "pinecone_namespace": Config.PINECONE_NAMESPACE,
        }

    app.include_router(chatbot_router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.app:app", host="0.0.0.0", port=5000, reload=False)
