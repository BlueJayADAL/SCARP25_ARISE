import asyncio
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketDisconnect
import cv2
import json
import os
import sys
import time
import base64
import numpy as np
from ultralytics import YOLO

from ARISE_llm import router as llm_router
from ARISE_tts import router as tts_router
from ARISE_stt import router as stt_router

# from ARISE_vision_API.yolo_vision import arise_vision
from ARISE_vision_API.ws_pose import websocket_endpoint

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


# clients = set()
# yolo_model = YOLO("../../models/yolo11n-pose_openvino_model_320")

# @app.websocket("/ws/pose")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     clients.add(websocket)
#     try:
#         while True:
#             data = await websocket.receive_text()
#             # Parse base64 data and time of capture
#             if ";" in data:
#                 capture_time_ms, data = data.split(";", 1)
#                 if time.time()*1000 - int(capture_time_ms) > 500:
#                     print('discarding frame, timed out')
#                     continue
#             else:
#                 continue
#             # Parse base64 header off of incoming data package
#             if "," in data:
#                 data = data.split(",", 1)[1]
#             img_bytes = base64.b64decode(data)
#             npimg = np.frombuffer(img_bytes, dtype=np.uint8)
#             img = cv2.imdecode(npimg, 1)
#             results = yolo_model(img, verbose=False, conf=0.2)

#             pose = results[0].keypoints # only focuses on one person at a time
#             kps = []
#             if len(pose.data) > 0:
#                 keypoints = pose.data[0].cpu().numpy().reshape(-1, 3)
#                 for kp in keypoints:
#                     kps.append({
#                         "x": int(kp[0]),
#                         "y": int(kp[1]),
#                         "z": 0,
#                         "visibility": int(kp[2]*100)
#                     })
#             if clients:
#                 data = json.dumps({"keypoints" : kps,
#                                    "capture_time_ms" : capture_time_ms})
#                 await asyncio.gather(*[client.send_text(data) for client in clients])
#     except WebSocketDisconnect:
#         clients.remove(websocket)
