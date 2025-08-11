import React, { useEffect, useRef, useState } from "react";

const COCO_CONNECTIONS = [
  [0, 1], [0, 2],
  [1, 3], [2, 4],
  [5, 6],
  [5, 7], [7, 9],
  [6, 8], [8, 10],
  [5, 11], [6, 12],
  [11, 12],
  [11, 13], [13, 15],
  [12, 14], [14, 16]
];

const WS_URL = "ws://localhost:8000/ws/pose"; // Update if needed

export default function FlappyBird() {
  const [keypoints, setKeypoints] = useState([]);
  const [fps, setFps] = useState(0);
  const [score, setScore] = useState(0);
  const [incomingCaptureTimeMs, setCaptureTime] = useState(0);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const ws = useRef(null);
  const lastFrameTime = useRef(performance.now());
  const framesDict = useRef({});

  // --- Setup webcam and send frames via WebSocket ---
  useEffect(() => {
    let sendInterval;
    let stream;

    // 1. Open webcam
    navigator.mediaDevices.getUserMedia({ video: true }).then((mediaStream) => {
      const videoTrack = mediaStream.getVideoTracks()[0];
      const imageCapture = new ImageCapture(videoTrack);

      // 2. Connect WebSocket after webcam is ready
      ws.current = new window.WebSocket(WS_URL);
      ws.current.onopen = () => {
        // 3. Periodically send frames to the backend
        sendInterval = setInterval(() => {
          imageCapture.grabFrame().then((imageBitmap) => {
            // Set up invisible canvas to draw camera to and retrieve frame from
            const canvas = captureCanvasRef.current;
            if (!canvas) return;
            const ctx = canvas.getContext("2d");
            
            ctx.drawImage(imageBitmap, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL("image/jpeg", 0.7); // compress a little bit

            // Send frame and time stamp to backend, and save frame in dict
            let captureTimeMs = (new Date()).getTime();
            ws.current.send(String(captureTimeMs)+";"+dataUrl);
            framesDict.current[captureTimeMs] = dataUrl;
          }).catch((err) => {
            console.error("Failed to grab frame", err);
          });

        }, 67); // 15 FPS
      };
      ws.current.onmessage = (event) => {
        // 4. Receive and parse keypoints, update state
        const now = performance.now();
        const elapsed = now - lastFrameTime.current;
        lastFrameTime.current = now;
        setFps(Math.round(1000 / elapsed));

        const data = JSON.parse(event.data);
        setKeypoints(data.keypoints || []);
        setCaptureTime(data.capture_time_ms);
      };
      // Clean up interval
      ws.current.onclose = () => {
        clearInterval(sendInterval);
        console.log("WebSocket closed");
      };
    });

    // Cleanup function
    return () => {
      clearInterval(sendInterval);
      if (ws.current) ws.current.close();
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    // Skip processing for newest frame if too far behind realtime camera
    let captureTimeMs = (new Date()).getTime();
    if (captureTimeMs - incomingCaptureTimeMs > 1000) { 
      console.log("Skipping frame, too long delay");
      return;
    }

    // Draw on canvas with stored image
    var img = new Image;
    img.src = framesDict.current[incomingCaptureTimeMs];
    // Clean up old frames from memory
    Object.keys(framesDict.current).forEach(key => {
        if (key + 1000 <= incomingCaptureTimeMs) delete framesDict.current[key];
    });
    img.onload = () => { 
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-canvas.width, 0);

      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      // Draw pose connections
      ctx.strokeStyle = "#00ff88";
      ctx.lineWidth = 3;

      COCO_CONNECTIONS.forEach(([start, end]) => {
        // Check keypoints are high enough confidence
        if (keypoints[start] && keypoints[end] && keypoints[start].visibility > 50 && keypoints[end].visibility > 50) {
          ctx.beginPath();
          ctx.moveTo(keypoints[start].x, keypoints[start].y);
          ctx.lineTo(keypoints[end].x, keypoints[end].y);
          ctx.stroke();
        }
      });

      // Draw keypoints
      keypoints.forEach((lm) => {
        // Check keypoints are high enough confidence
        if (lm.visibility > 50){
          ctx.beginPath();
          ctx.arc(lm.x, lm.y, 6, 0, 2 * Math.PI);
          ctx.fillStyle = "#ff3366";
          ctx.fill();
        }
      });

      // Score logic
      if (keypoints[10] && keypoints[0] && keypoints[10].y < keypoints[0].y) {
        setScore((prev) => prev + 1);
      }
      ctx.restore();
    };

  }, [incomingCaptureTimeMs]);

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Flappy Pose Game</h1>

      <div style={styles.stats}>
        <span>FPS: {fps}</span>
        <span>Score: {score}</span>
      </div>

      <div style={styles.videoWrapper}>
        <video
          ref={videoRef}
          width={640}
          height={480}
          autoPlay
          muted
          style={{display: "none"}}
        />
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          style={styles.canvas}
        />
        <canvas
          ref={captureCanvasRef}
          width={640}
          height={480}
          style={{ display: "none" }}
        />
      </div>

      <div style={styles.footer}>Raise your right hand to earn points!</div>
    </div>
  );
}

const styles = {
  container: {
    fontFamily: "Arial, sans-serif",
    backgroundColor: "#121212",
    color: "#fff",
    textAlign: "center",
    minHeight: "100vh",
    padding: "1rem"
  },
  title: {
    fontSize: "2rem",
    marginBottom: "1rem"
  },
  stats: {
    display: "flex",
    justifyContent: "center",
    gap: "2rem",
    fontSize: "1.2rem",
    marginBottom: "1rem"
  },
  videoWrapper: {
    position: "relative",
    width: 640,
    height: 480,
    margin: "0 auto",
    border: "4px solid #00ff88",
    borderRadius: "12px",
    overflow: "hidden"
  },
  video: {
    position: "absolute",
    top: 0,
    left: 0,
    zIndex: 1,
    backgroundColor: "#000",
    transform: "scaleX(-1)"
  },
  canvas: {
    position: "absolute",
    top: 0,
    left: 0,
    zIndex: 2,
    pointerEvents: "none"
  },
  footer: {
    marginTop: "1rem",
    fontSize: "1rem",
    color: "#aaaaaa"
  }
};
