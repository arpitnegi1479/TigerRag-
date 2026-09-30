from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.documents import router as documents_router
from app.api.routes.evaluation import router as evaluation_router
from app.api.routes.graph import router as graph_router
from app.api.routes.health import router as health_router
from app.api.routes.metrics import router as metrics_router
from app.api.routes.queries import router as query_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.rate_limit import RateLimitMiddleware

logger = setup_logging()


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting %s in %s mode", settings.app_name, settings.environment)
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Backend for the comparative Agentic GraphRAG system.",
    debug=False,
    lifespan=lifespan,
)
app.add_middleware(
    RateLimitMiddleware,
    max_requests=settings.rate_limit_requests_per_minute,
    window_seconds=settings.rate_limit_window_seconds,
)

app.include_router(health_router)
app.include_router(documents_router)
app.include_router(query_router)
app.include_router(graph_router)
app.include_router(evaluation_router)
app.include_router(metrics_router)


@app.get("/")
def root() -> dict:
    return {"message": "Agentic GraphRAG backend is running."}
