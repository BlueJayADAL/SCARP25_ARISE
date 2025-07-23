from fastapi import WebSocket, WebSocketDisconnect
import asyncio, json, time, base64, cv2, numpy as np
from ultralytics import YOLO

clients = set()
yolo_model = YOLO("../../../models/yolo11n-pose_openvino_model_320")

async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Parse base64 data and time of capture
            if ";" in data:
                capture_time_ms, data = data.split(";", 1)
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
            results = yolo_model(img, verbose=False, conf=0.2)

            pose = results[0].keypoints # only focuses on one person at a time
            kps = []
            if len(pose.data) > 0:
                keypoints = pose.data[0].cpu().numpy().reshape(-1, 3)
                for kp in keypoints:
                    kps.append({
                        "x": int(kp[0]),
                        "y": int(kp[1]),
                        "z": 0,
                        "visibility": int(kp[2]*100)
                    })
            if clients:
                data = json.dumps({"keypoints" : kps,
                                   "capture_time_ms" : capture_time_ms})
                await asyncio.gather(*[client.send_text(data) for client in clients])
    except WebSocketDisconnect:
        clients.remove(websocket)