import { useEffect, useRef, useState } from 'react';
import { Hands } from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils';

export default function useHandTracking() {
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [indexClicking, setIndexClicking] = useState(false);
  const videoRef = useRef(null);
  const cameraRef = useRef(null);
  const handsRef = useRef(null); // Store Hands instance
  const stablePos = useRef({ x: 0, y: 0 });
  const lastIndexY = useRef(null);
  const clickCooldown = useRef(false);
  const isInitialized = useRef(false); // Prevent multiple initializations

  // Check hand gesture states
  function checkHandGesture(landmarks) {
    const indexTip = landmarks[8];
    const middleTip = landmarks[12];
    const ringTip = landmarks[16];
    const pinkyTip = landmarks[20];
    const wrist = landmarks[0];

    const getDistance = (point1, point2) => {
      const dx = point1.x - point2.x;
      const dy = point1.y - point2.y;
      return Math.sqrt(dx * dx + dy * dy);
    };

    const indexDist = getDistance(indexTip, wrist);
    const middleDist = getDistance(middleTip, wrist);
    const ringDist = getDistance(ringTip, wrist);
    const pinkyDist = getDistance(pinkyTip, wrist);

    const isHandOpen =
      Math.abs(indexDist - middleDist) < 0.05 &&
      Math.abs(indexDist - ringDist) < 0.05 &&
      Math.abs(indexDist - pinkyDist) < 0.05 &&
      indexDist > 0.2;

    const isPointing =
      indexDist > middleDist * 1.5 &&
      indexDist > ringDist * 1.5 &&
      indexDist > pinkyDist * 1.5 &&
      indexDist > 0.2;

    let isClicking = false;
    if (lastIndexY.current !== null && !clickCooldown.current) {
      const yDiff = indexTip.y - lastIndexY.current;
      if ((isPointing || isHandOpen) && yDiff > 0.05) {
        isClicking = true;
        clickCooldown.current = true;
        setTimeout(() => {
          clickCooldown.current = false;
        }, 500);
      }
    }
    lastIndexY.current = indexTip.y;

    return { isHandOpen, isPointing, isClicking };
  }

  useEffect(() => {
    if (isInitialized.current) return; // Prevent multiple initializations
    isInitialized.current = true;

    // Create video element
    if (!videoRef.current) {
      const v = document.createElement('video');
      v.setAttribute('playsinline', '');
      v.style.display = 'none';
      document.body.appendChild(v);
      videoRef.current = v;
    }

    // Initialize MediaPipe Hands
    const hands = new Hands({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 0,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7,
    });

    handsRef.current = hands;

    hands.onResults((res) => {
      if (!res.multiHandLandmarks?.length) return;

      const lm = res.multiHandLandmarks[0];
      const tip = lm[8];

      const CAM_MIN = 0.3;
      const CAM_MAX = 0.7;
      const clamp = (v, a, b) => Math.min(Math.max(v, a), b);

      const rawX = clamp(tip.x, CAM_MIN, CAM_MAX);
      const rawY = clamp(tip.y, CAM_MIN, CAM_MAX);
      const normX = (rawX - CAM_MIN) / (CAM_MAX - CAM_MIN);
      const normY = (rawY - CAM_MIN) / (CAM_MAX - CAM_MIN);
      const screenX = (1 - normX) * window.innerWidth;
      const screenY = normY * window.innerHeight;

      const { isHandOpen, isPointing, isClicking } = checkHandGesture(lm);
      setIndexClicking(isClicking);

      if (isPointing && !isHandOpen) {
        setPosition((prev) => {
          const newPos = {
            x: prev.x * 0.8 + screenX * 0.2,
            y: prev.y * 0.8 + screenY * 0.2,
          };
          stablePos.current = newPos;
          return newPos;
        });
      } else {
        setPosition(stablePos.current);
      }
    });

    // Initialize Camera
    cameraRef.current = new Camera(videoRef.current, {
      width: 1280,
      height: 720,
      onFrame: async () => {
        try {
          if (videoRef.current && handsRef.current) {
            await handsRef.current.send({ image: videoRef.current });
          }
        } catch (error) {
          console.error('Error in hands.send:', error);
        }
      },
    });

    // Start camera
    cameraRef.current.start().catch((error) => {
      console.error('Failed to start camera:', error);
    });

    // Cleanup on unmount
    return () => {
      isInitialized.current = false;
      if (cameraRef.current) {
        cameraRef.current.stop();
        cameraRef.current = null;
      }
      if (handsRef.current) {
        handsRef.current.close().catch((error) => {
          console.error('Error closing hands:', error);
        });
        handsRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.remove();
        videoRef.current = null;
      }
    };
  }, []);

  return { position, indexClicking };
}