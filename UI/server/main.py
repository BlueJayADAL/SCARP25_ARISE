from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from ARISE_llm import router as llm_router
from ARISE_tts import router as tts_router
from ARISE_stt import router as stt_router

# from ARISE_vision_API.yolo_vision import arise_vision
from ARISE_vision_API.ws_pose import websocket_endpoint, websocket_space_invaders

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change this to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(llm_router, prefix="/api/llm")
app.include_router(tts_router, prefix="/api/tts")
app.include_router(stt_router, prefix="/api/stt")
app.add_api_websocket_route("/ws/pose", websocket_endpoint)
app.add_api_websocket_route("/ws/spaceinvaders", websocket_space_invaders)

# Mount static files at /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html for the root route
@app.get("/", response_class=HTMLResponse)
async def serve_root():
    with open(os.path.join("static", "index.html"), encoding = "utf8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

# Example API endpoint
@app.get("/api/hello")
async def hello():
    return {"message": "Hello from FastAPI!"}

# Serve index.html for React Router routes (non-API, non-static)
@app.get("/{path:path}", response_class=HTMLResponse)
async def serve_react_app(path: str):
    if path.startswith("api/") or path.startswith("static/"):
        return JSONResponse(content={"error": "Not found"}, status_code=404)
    with open(os.path.join("static", "index.html"),encoding = 'utf8') as f:
        return HTMLResponse(content=f.read(), status_code=200)
