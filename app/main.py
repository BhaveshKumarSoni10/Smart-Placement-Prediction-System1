from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import FRONTEND_DIR
from app.database import Base, engine
from app.ml.train import train_and_save_model
from app.routes.api import router as api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising SQLite database tables...")
    Base.metadata.create_all(bind=engine)

    logger.info("Loading or training placement ML model...")
    try:
        app.state.ml_artifact = train_and_save_model()
    except Exception as exc:
        logger.exception("Startup ML initialization error: %s", exc)
        app.state.ml_artifact = None

    yield


app = FastAPI(
    title="Smart Placement Intelligence System",
    description="Educational decision-support dashboard for student profile analytics & placement readiness.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
app.mount("/js", StaticFiles(directory=FRONTEND_DIR / "js"), name="js")


@app.get("/", include_in_schema=False)
def serve_dashboard():
    return FileResponse(FRONTEND_DIR / "index.html")
