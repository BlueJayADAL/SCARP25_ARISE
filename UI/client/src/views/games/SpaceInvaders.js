import React, { useEffect, useRef, useState } from "react";

const WS_URL = "ws://localhost:8000/ws/spaceinvaders"; // Update if needed

export default function SpaceInvaders() {
  const [fps, setFps] = useState(0);
  const [score, setScore] = useState(0);
  const [lives, setLives] = useState(5);
  const [gameOver, setGameOver] = useState(false);
  const [windowSize, setWindowSize] = useState({ width: 640, height: 480 });

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const poseCanvasRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const ws = useRef(null);
  const lastFrameTime = useRef(performance.now());
  const gameStateRef = useRef(null);

  // Dictionary of frames keyed by capture time (ms)
  const framesDict = useRef({});

  // Store latest capture time from backend
  const [incomingCaptureTimeMs, setCaptureTime] = useState(0);

  useEffect(() => {
    let sendInterval;
    let stream;

    navigator.mediaDevices.getUserMedia({ video: true }).then((mediaStream) => {
      const videoTrack = mediaStream.getVideoTracks()[0];
      const imageCapture = new window.ImageCapture(videoTrack);

      ws.current = new window.WebSocket(WS_URL);
      ws.current.onopen = () => {
        sendInterval = setInterval(() => {
          imageCapture.grabFrame().then((imageBitmap) => {
            const canvas = captureCanvasRef.current;
            if (!canvas) return;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(imageBitmap, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
            const captureTimeMs = (new Date()).getTime();
            ws.current.send(String(captureTimeMs) + ";" + dataUrl);
            framesDict.current[captureTimeMs] = dataUrl;
          }).catch(() => {});
        }, 67); // ~15 FPS
      };
      ws.current.onmessage = (event) => {
        const now = performance.now();
        const elapsed = now - lastFrameTime.current;
        lastFrameTime.current = now;
        setFps(Math.round(1000 / elapsed));

        const data = JSON.parse(event.data);
        gameStateRef.current = data;
        setScore(data.score);
        setLives(data.lives);
        setGameOver(data.game_over);
        setWindowSize({
          width: data.window_width,
          height: data.window_height
        });
        setCaptureTime(data.capture_time_ms);

        // Optionally clean up old frames
        Object.keys(framesDict.current).forEach(key => {
          if (parseInt(key) + 1000 <= data.capture_time_ms) delete framesDict.current[key];
        });
      };
      ws.current.onclose = () => {
        clearInterval(sendInterval);
      };
    });

    return () => {
      clearInterval(sendInterval);
      if (ws.current) ws.current.close();
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  // Draw game state on main canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    const state = gameStateRef.current;
    if (!ctx || !state) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw ship
    const shipX = state.ship_x;
    const shipY = state.window_height - 40;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(shipX + 20, shipY);
    ctx.lineTo(shipX, shipY + 40);
    ctx.lineTo(shipX + 40, shipY + 40);
    ctx.closePath();
    ctx.fillStyle = "#00ff00";
    ctx.fill();
    ctx.restore();

    // Draw bullets
    state.bullets.forEach(bullet => {
      ctx.beginPath();
      ctx.arc(bullet.x, bullet.y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "#00aaff";
      ctx.fill();
      ctx.beginPath();
      ctx.arc(bullet.x, bullet.y, 3, 0, 2 * Math.PI);
      ctx.fillStyle = "#fff";
      ctx.fill();
    });

    // Draw enemies
    state.enemies.forEach(enemy => {
      ctx.save();
      ctx.beginPath();
      ctx.rect(enemy.x, enemy.y, 40, 30);
      ctx.fillStyle = "#ff3366";
      ctx.fill();
      ctx.restore();
    });

    // Draw explosions
    state.explosions.forEach(explosion => {
      ctx.save();
      ctx.beginPath();
      ctx.arc(explosion.x, explosion.y, 20, 0, 2 * Math.PI);
      ctx.fillStyle = "orange";
      ctx.globalAlpha = explosion.timer / 10;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.restore();
    });

    // Draw score/lives
    ctx.font = "24px Arial";
    ctx.fillStyle = "#00ff00";
    ctx.fillText(`Score: ${state.score}`, 10, 40);
    ctx.fillStyle = "#ff3366";
    ctx.fillText(`Lives: ${state.lives}`, 10, 80);

    // Draw game over
    if (state.game_over) {
      ctx.font = "48px Arial";
      ctx.fillStyle = "#fff";
      ctx.fillText("GAME OVER", canvas.width / 2 - 150, canvas.height / 2);
    }
  }, [score, lives, gameOver, windowSize, incomingCaptureTimeMs]);

  // Draw pose/camera overlay on its own canvas
  useEffect(() => {
    const poseCanvas = poseCanvasRef.current;
    const poseCtx = poseCanvas?.getContext("2d");
    const state = gameStateRef.current;
    if (!poseCtx || !state) return;

    const frameUrl = framesDict.current[state.capture_time_ms];
    if (frameUrl && state.keypoints) {
      const img = new window.Image();
      img.src = frameUrl;
      img.onload = () => {
        poseCtx.clearRect(0, 0, poseCanvas.width, poseCanvas.height);
        poseCtx.drawImage(img, 0, 0, poseCanvas.width, poseCanvas.height);

        // Draw keypoints and connections
        poseCtx.strokeStyle = "#00ff88";
        poseCtx.lineWidth = 2;
        const kps = state.keypoints;
        const COCO_CONNECTIONS = [
          [0, 1], [0, 2], [1, 3], [2, 4],
          [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
          [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
        ];
        // Scale keypoints to overlay size
        kps.forEach((kp, idx) => {
          if (kp.visibility > 50) {
            poseCtx.beginPath();
            poseCtx.arc(
              kp.x / windowSize.width * poseCanvas.width,
              kp.y / windowSize.height * poseCanvas.height,
              4, 0, 2 * Math.PI
            );
            poseCtx.fillStyle = "#ff3366";
            poseCtx.fill();
          }
        });
        COCO_CONNECTIONS.forEach(([start, end]) => {
          if (
            kps[start] && kps[end] &&
            kps[start].visibility > 50 && kps[end].visibility > 50
          ) {
            poseCtx.beginPath();
            poseCtx.moveTo(
              kps[start].x / windowSize.width * poseCanvas.width,
              kps[start].y / windowSize.height * poseCanvas.height
            );
            poseCtx.lineTo(
              kps[end].x / windowSize.width * poseCanvas.width,
              kps[end].y / windowSize.height * poseCanvas.height
            );
            poseCtx.stroke();
          }
        });

        // Draw border and label
        poseCtx.strokeStyle = "#fff";
        poseCtx.lineWidth = 2;
        poseCtx.strokeRect(0, 0, poseCanvas.width, poseCanvas.height);
        poseCtx.font = "14px Arial";
        poseCtx.fillStyle = "#fff";
        poseCtx.fillText("Pose View", 8, 18);
      };
    }
  }, [incomingCaptureTimeMs, windowSize]);

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Space Invaders Pose Game</h1>
      <div style={styles.stats}>
        <span>FPS: {fps}</span>
        <span>Score: {score}</span>
        <span>Lives: {lives}</span>
      </div>
      <div style={styles.videoWrapper}>
        <video
          ref={videoRef}
          width={windowSize.width}
          height={windowSize.height}
          autoPlay
          muted
          style={{display: "none"}}
        />
        <canvas
          ref={canvasRef}
          width={windowSize.width}
          height={windowSize.height}
          style={styles.canvas}
        />
        <canvas
          ref={poseCanvasRef}
          width={155}
          height={120}
          style={styles.poseCanvas}
        />
        <canvas
          ref={captureCanvasRef}
          width={windowSize.width}
          height={windowSize.height}
          style={{ display: "none" }}
        />
      </div>
      <div style={styles.footer}>
        Move your nose left/right to control the ship. Avoid enemies!
      </div>
      {gameOver && (
        <div style={styles.gameOver}>
          <h2>Game Over</h2>
          <div>Refresh page to restart.</div>
        </div>
      )}
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
  canvas: {
    position: "absolute",
    top: 0,
    left: 0,
    zIndex: 2,
    pointerEvents: "none"
  },
  poseCanvas: {
    position: "absolute",
    top: 10,
    right: 10,
    width: 155,
    height: 120,
    zIndex: 3,
    border: "2px solid #fff",
    background: "#222"
  },
  footer: {
    marginTop: "1rem",
    fontSize: "1rem",
    color: "#aaaaaa"
  },
  gameOver: {
    marginTop: "2rem",
    fontSize: "2rem",
    color: "#ff3366"
  }
};
