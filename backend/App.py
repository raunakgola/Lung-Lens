from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from global_variables.model_store import load_model
from logger import get_logger
from middleware import add_middleware, add_exception_handlers
from routes import health_router, predict_router, explain_router

log = get_logger("lunglens.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="LungLens API",
    description="Fine-tuned ResNet-18 COVID radiography classifier with Grad-CAM explainability",
    version="1.0.0",
    lifespan=lifespan,
)

add_middleware(app)
add_exception_handlers(app)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(explain_router)

# Serve the built React frontend if present (frontend/dist).
# API routes above take precedence; everything else falls through to the SPA.
frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    log.info(f"Serving frontend from {frontend_dist}")
else:
    log.info("frontend/dist not found — API only mode (run 'npm run build' in frontend/)")
