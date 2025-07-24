import numpy as np
import random
import math
from ultralytics import YOLO

# Global game state dictionary
GAME_STATE = {
    'initialized': False,
    'score': 0,
    'lives': 5,
    'ship_x': 320,
    'bullets': [],
    'enemies': [],
    'explosions': [],
    'latest_x_value': 0.5,
    'shoot_cooldown': 0,
    'WINDOW_WIDTH': 640,
    'WINDOW_HEIGHT': 480,
    'keypoints': []
}

# Game constants
SHIP_WIDTH = 40
SHIP_HEIGHT = 40
BULLET_SPEED = 20
BULLET_RADIUS = 3
ENEMY_WIDTH = 40
ENEMY_HEIGHT = 30
ENEMY_SPEED = 2

POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# Load YOLO model once
MODEL = None
def init_model():
    global MODEL
    if MODEL is None:
        # From UI/server directory
        MODEL = YOLO("../../models/yolo11n-pose_openvino_model_320")

def get_keypoint_position(results, keypoint_num, axis='x'):
    if not 0 <= keypoint_num <= 16:
        return 0.5
    if axis.lower() not in ['x', 'y']:
        return 0.5
    try:
        keypoint = results[0].keypoints.xyn[0][keypoint_num]
        return keypoint[0].item() if axis.lower() == 'x' else keypoint[1].item()
    except Exception:
        return 0.5

def create_enemy():
    return {'x': random.randint(0, GAME_STATE['WINDOW_WIDTH'] - ENEMY_WIDTH), 'y': 0}

def fire_bullet(x):
    GAME_STATE['bullets'].append({'x': x + SHIP_WIDTH//2, 'y': GAME_STATE['WINDOW_HEIGHT'] - SHIP_HEIGHT - 10})

def run_game_frame(frame):
    # Initialize model and state if needed
    if not GAME_STATE['initialized']:
        init_model()
        GAME_STATE['initialized'] = True

    # Pose detection
    results = MODEL.predict(frame, imgsz=320, verbose=False)
    nose_x = get_keypoint_position(results, 0, 'x')
    GAME_STATE['latest_x_value'] = nose_x

    # Ship position update (nose controls ship)
    target_x = int(np.interp(nose_x, [0.1, 0.9], [GAME_STATE['WINDOW_WIDTH'] - SHIP_WIDTH, 0]))
    GAME_STATE['ship_x'] = int(np.clip(target_x, 0, GAME_STATE['WINDOW_WIDTH'] - SHIP_WIDTH))

    # Enemy spawn
    if random.random() < 0.02:
        GAME_STATE['enemies'].append(create_enemy())

    # Bullet firing (auto-fire)
    GAME_STATE['shoot_cooldown'] -= 1
    if GAME_STATE['shoot_cooldown'] <= 0:
        fire_bullet(GAME_STATE['ship_x'])
        GAME_STATE['shoot_cooldown'] = 12

    # Move bullets
    for bullet in GAME_STATE['bullets'][:]:
        bullet['y'] -= BULLET_SPEED
        if bullet['y'] < 0:
            GAME_STATE['bullets'].remove(bullet)

    # Move enemies and check collisions
    for enemy in GAME_STATE['enemies'][:]:
        enemy['y'] += ENEMY_SPEED
        if enemy['y'] > GAME_STATE['WINDOW_HEIGHT']:
            GAME_STATE['enemies'].remove(enemy)
            GAME_STATE['lives'] -= 1
            continue
        enemy_center = (enemy['x'] + ENEMY_WIDTH//2, enemy['y'] + ENEMY_HEIGHT//2)
        for bullet in GAME_STATE['bullets'][:]:
            distance = math.hypot(bullet['x'] - enemy_center[0], bullet['y'] - enemy_center[1])
            if distance < ENEMY_WIDTH//2:
                if bullet in GAME_STATE['bullets']: GAME_STATE['bullets'].remove(bullet)
                if enemy in GAME_STATE['enemies']:
                    GAME_STATE['enemies'].remove(enemy)
                    GAME_STATE['explosions'].append({'x': enemy_center[0], 'y': enemy_center[1], 'timer': 10})
                GAME_STATE['score'] += 10
                break

    # Update explosions
    for explosion in GAME_STATE['explosions'][:]:
        if explosion['timer'] > 0:
            explosion['timer'] -= 1
        else:
            GAME_STATE['explosions'].remove(explosion)

    # Game over logic
    if GAME_STATE['lives'] <= 0:
        # Reset game state if needed, or keep as is for client to handle
        pass

    # Prepare formatted keypoints
    keypoints = results[0].keypoints.data[0].cpu().numpy().reshape(-1, 3)
    if len(keypoints) < 17:
        keypoints = np.zeros((17, 3))

    # Prepare state for websocket transmission
    state = {
        'score': GAME_STATE['score'],
        'lives': GAME_STATE['lives'],
        'ship_x': GAME_STATE['ship_x'],
        'bullets': list(GAME_STATE['bullets']),
        'enemies': list(GAME_STATE['enemies']),
        'explosions': list(GAME_STATE['explosions']),
        'game_over': GAME_STATE['lives'] <= 0,
        'window_width': GAME_STATE['WINDOW_WIDTH'],
        'window_height': GAME_STATE['WINDOW_HEIGHT'],
        'keypoints': keypoints
    }
    return state
