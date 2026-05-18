"""FastAPI application entrypoint for Alcohol Intelligence backend."""

from __future__ import annotations

from typing import Any, Mapping

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.health import router as health_router
from api.logging_utils import structured_error
from api.routes import router as pipeline_router

app = FastAPI(title="Alcohol Intelligence API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(pipeline_router)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    detail = "request validation failed"
    if exc.errors():
        first = exc.errors()[0]
        message = first.get("msg")
        if isinstance(message, str) and message.strip():
            detail = message.strip()
    return JSONResponse(status_code=400, content=structured_error(detail, "request_validation"))


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, Mapping):
        payload = dict(detail)
        payload.setdefault("error", True)
        payload.setdefault("stage", "http")
        payload.setdefault("message", "request failed")
        return JSONResponse(status_code=exc.status_code, content=payload)
    return JSONResponse(status_code=exc.status_code, content=structured_error(str(detail), "http"))


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content=structured_error(str(exc), "unhandled"))


@app.get("/")
def root() -> Mapping[str, Any]:
    return {"service": "alcohol-intelligence-api", "status": "ok"}
