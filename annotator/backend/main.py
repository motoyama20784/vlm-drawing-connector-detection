from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from annotator.backend.routes import images, annotations, infer

app = FastAPI(title="Annotation Tool")

app.include_router(images.router, prefix="/api")
app.include_router(annotations.router, prefix="/api")
app.include_router(infer.router, prefix="/api")

@app.get("/api/health")
def health():
    return {"status": "ok"}

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
