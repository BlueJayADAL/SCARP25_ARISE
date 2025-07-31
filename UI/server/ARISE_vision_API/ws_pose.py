from fastapi import WebSocket, WebSocketDisconnect
import asyncio, json, time, base64, cv2, numpy as np
from ultralytics import YOLO

from ARISE_vision_API.yolo_vision import arise_vision, init_yolo
from ARISE_vision_API.games import *

clients = set()

async def websocket_endpoint(websocket: WebSocket):
    '''
    WebSocket endpoint for pose estimation and exercise feedback.
    Receives base64-encoded frames, processes them, and sends results to all clients.
    '''
    await websocket.accept()
    clients.add(websocket)
    try:
        init_yolo()
        while True:
            data = await websocket.receive_text()

            if clients:
                # Parse base64 data and time of capture
                if ";" in data:
                    capture_time_ms, data = data.split(";", 1)
                    # Discard frame if too old (>500ms)
                    if time.time()*1000 - int(capture_time_ms) > 500:
                        print('discarding frame, timed out')
                        continue
                else:
                    continue
                # Parse base64 header off of incoming data package
                if "," in data:
                    data = data.split(",", 1)[1]
                img_bytes = base64.b64decode(data)
                npimg = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(npimg, 1)

                # Make copy of returned 'SharedState' json object
                api_results = arise_vision(img).get_all_data()

                # Format keypoints for frontend
                kps = []
                if len(api_results['keypoints']) > 0:
                    for kp in api_results['keypoints']:
                        kps.append({
                            "x": int(kp[0]),
                            "y": int(kp[1]),
                            "z": 0,
                            "visibility": int(kp[2]*100)
                        })
            
                # Attach keypoints and camera image capture timestamp for frontend reference
                api_results["capture_time_ms"] = capture_time_ms
                api_results["keypoints"] = kps
                data = json.dumps(api_results)
                # Send results to all connected clients (frontend)
                await asyncio.gather(*[client.send_text(data) for client in clients])
    except WebSocketDisconnect:
        clients.remove(websocket)

async def websocket_space_invaders(websocket: WebSocket):
    '''
    WebSocket endpoint for Space Invaders game.
    Receives base64-encoded frames, processes them, and sends game state to all clients.
    '''
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()

            if clients:
                # Parse base64 data and time of capture
                if ";" in data:
                    capture_time_ms, data = data.split(";", 1)
                    # Discard frame if too old (>500ms)
                    if time.time()*1000 - int(capture_time_ms) > 500:
                        print('discarding frame, timed out')
                        continue
                else:
                    continue
                # Parse base64 header off of incoming data package
                if "," in data:
                    data = data.split(",", 1)[1]
                img_bytes = base64.b64decode(data)
                npimg = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(npimg, 1)

                # Call Space Invaders game logic
                game_state = run_game_frame(img)
                # Format keypoints for frontend
                kps = []
                if len(game_state['keypoints']) > 0:
                    for kp in game_state['keypoints']:
                        kps.append({
                            "x": int(kp[0]),
                            "y": int(kp[1]),
                            "z": 0,
                            "visibility": int(kp[2]*100)
                        })

                # Attach keypoints and camera image capture timestamp for frontend reference
                game_state["capture_time_ms"] = capture_time_ms
                game_state["keypoints"] = kps
                data = json.dumps(game_state)
                # Send game state to all connected clients (frontend)
                await asyncio.gather(*[client.send_text(data) for client in clients])
    except WebSocketDisconnect:
        clients.remove(websocket)