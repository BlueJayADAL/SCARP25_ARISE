import asyncio
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketDisconnect
#import mediapipe as mp
import cv2
import json
import os

from ARISE_llm import router as llm_router
from ARISE_tts import router as tts_router
from ARISE_stt import router as stt_router


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

#mp_pose = mp.solutions.pose # type: ignore
#pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

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


clients = set()

async def pose_streamer():
    while True:
        """
        success, frame = cap.read()
        if not success:
            continue

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        landmarks = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                })

        # Broadcast to all connected clients
        if clients:
            data = json.dumps({"landmarks": landmarks})
            await asyncio.gather(*[client.send_text(data) for client in clients])
        """
        await asyncio.sleep(0.03)  # roughly 30 FPS

@app.websocket("/ws/pose")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # Just to keep connection alive
    except WebSocketDisconnect:
        clients.remove(websocket)

# Start streaming when the app starts
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(pose_streamer())  # Start the WebSocket endpoint