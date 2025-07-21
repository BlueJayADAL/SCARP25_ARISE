import React, { useEffect, useRef, useState } from "react";

const POSE_CONNECTIONS = [
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19], [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 12], [23, 24], [11, 23], [12, 24], [23, 25], [24, 26], [25, 27], [26, 28], [27, 29], [28, 30], [29, 31], [30, 32],
  [27, 31], [28, 32]
];

const WS_URL = "ws://localhost:8000/ws/pose"; // Update if needed

export default function FlappyBird() {
  const [landmarks, setLandmarks] = useState([]);
  const [fps, setFps] = useState(0);
  const [score, setScore] = useState(0);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const ws = useRef(null);
  const lastFrameTime = useRef(performance.now());

  // Webcam access
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      if (videoRef.current) videoRef.current.srcObject = stream;
    });
  }, []);

  // WebSocket setup
  useEffect(() => {
    ws.current = new WebSocket(WS_URL);

    ws.current.onmessage = (event) => {
      const now = performance.now();
      const elapsed = now - lastFrameTime.current;
      lastFrameTime.current = now;
      setFps(Math.round(1000 / elapsed));

      const data = JSON.parse(event.data);
      setLandmarks(data.landmarks || []);
    };

    ws.current.onclose = () => console.log("WebSocket closed");

    return () => ws.current.close();
  }, []);

  // Drawing
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    // Mirror canvas horizontally
    ctx.save();
    ctx.scale(-1, 1);
    ctx.translate(-canvas.width, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw pose connections
    ctx.strokeStyle = "#00ff88";
    ctx.lineWidth = 3;
    POSE_CONNECTIONS.forEach(([start, end]) => {
      if (landmarks[start] && landmarks[end]) {
        ctx.beginPath();
        ctx.moveTo(landmarks[start].x * canvas.width, landmarks[start].y * canvas.height);
        ctx.lineTo(landmarks[end].x * canvas.width, landmarks[end].y * canvas.height);
        ctx.stroke();
      }
    });

    // Draw landmarks
    landmarks.forEach((lm) => {
      ctx.beginPath();
      ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 6, 0, 2 * Math.PI);
      ctx.fillStyle = "#ff3366";
      ctx.fill();
    });

    // Score logic
    if (landmarks[16] && landmarks[0] && landmarks[16].y < landmarks[0].y) {
      setScore((prev) => prev + 1);
    }

    ctx.restore(); 
  }, [landmarks]);


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
          style={styles.video}
        />
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          style={styles.canvas}
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
