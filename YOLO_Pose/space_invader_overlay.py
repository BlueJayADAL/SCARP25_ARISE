import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import pygame
import numpy as np
import random
from collections import deque
import threading
import time
import math

# Load YOLO model
model = YOLO("models/yolo11n-pose_openvino_model")
imgsz = 320

# Initialize Pygame
pygame.init()
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 1000
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Nose Space Invaders")

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (179, 0, 255)
BLUE = (164, 224, 0)
YELLOW = (120, 163, 0)

# Game objects
SHIP_WIDTH = 40
SHIP_HEIGHT = 40
ship_x = WINDOW_WIDTH // 2

bullets = []
BULLET_SPEED = 20
BULLET_RADIUS = 3

enemies = []
ENEMY_WIDTH = 40
ENEMY_HEIGHT = 30
ENEMY_SPEED = 2

# Pose skeleton connections
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def get_keypoint_position(results, keypoint_num, axis='x'):
    if not 0 <= keypoint_num <= 16:
        raise ValueError("Keypoint number must be between 0 and 16")
    if axis.lower() not in ['x', 'y']:
        raise ValueError("Axis must be 'x' or 'y'")
    keypoint = results[0].keypoints.xyn[0][keypoint_num]
    return keypoint[0].item() if axis.lower() == 'x' else keypoint[1].item()

def draw_ship(x, y):
    points = [(x + SHIP_WIDTH//2, y), (x, y + SHIP_HEIGHT), (x + SHIP_WIDTH, y + SHIP_HEIGHT)]
    pygame.draw.polygon(screen, GREEN, points)
    flame_points = [(x + SHIP_WIDTH//2, y + SHIP_HEIGHT),
                    (x + SHIP_WIDTH//2 - 10, y + SHIP_HEIGHT + 10),
                    (x + SHIP_WIDTH//2 + 10, y + SHIP_HEIGHT + 10)]
    flame_color = (random.randint(200, 255), random.randint(100, 150), 0)
    pygame.draw.polygon(screen, flame_color, flame_points)

def draw_enemy(x, y):
    body = [(x + ENEMY_WIDTH//2, y), (x + ENEMY_WIDTH, y + ENEMY_HEIGHT//2),
            (x + ENEMY_WIDTH//2, y + ENEMY_HEIGHT), (x, y + ENEMY_HEIGHT//2)]
    wing_l = [(x, y + ENEMY_HEIGHT//2), (x - ENEMY_WIDTH//4, y + ENEMY_HEIGHT//2), (x, y + ENEMY_HEIGHT//3)]
    wing_r = [(x + ENEMY_WIDTH, y + ENEMY_HEIGHT//2), (x + ENEMY_WIDTH + ENEMY_WIDTH//4, y + ENEMY_HEIGHT//2),
              (x + ENEMY_WIDTH, y + ENEMY_HEIGHT//3)]
    eye = [(x + ENEMY_WIDTH//2 - 5, y + ENEMY_HEIGHT//2 - 5),
           (x + ENEMY_WIDTH//2 + 5, y + ENEMY_HEIGHT//2 - 5),
           (x + ENEMY_WIDTH//2, y + ENEMY_HEIGHT//2 + 5)]
    pygame.draw.polygon(screen, RED, body)
    pygame.draw.polygon(screen, BLUE, wing_l)
    pygame.draw.polygon(screen, BLUE, wing_r)
    eye_color = YELLOW if random.random() > 0.5 else RED
    pygame.draw.polygon(screen, eye_color, eye)

def draw_explosion(x, y):
    max_radius = 50
    for i in range(8):
        angle = (2 * math.pi * i) / 8
        points = []
        for j in range(3):
            point_angle = angle + (2 * math.pi * j) / 3
            radius = max_radius * (1 - random.random() * 0.3)
            point_x = x + math.cos(point_angle) * radius
            point_y = y + math.sin(point_angle) * radius
            points.append((point_x, point_y))
        color = (255, max(100, 255 - i * 20), 0)
        pygame.draw.polygon(screen, color, points)
    pygame.draw.circle(screen, WHITE, (int(x), int(y)), random.randint(5, 15))

SMOOTHING_WINDOW = 2
position_history = deque([WINDOW_WIDTH // 2] * SMOOTHING_WINDOW, maxlen=SMOOTHING_WINDOW)

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 640)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

latest_x_value = 0.5
latest_frame_pose = None
pose_thread_running = True
explosions = []

def create_enemy():
    return {'x': random.randint(0, WINDOW_WIDTH - ENEMY_WIDTH), 'y': 0}

def fire_bullet(x):
    bullets.append({'x': x + SHIP_WIDTH//2, 'y': WINDOW_HEIGHT - SHIP_HEIGHT - 10})

def pose_detection_thread():
    global latest_x_value, latest_frame_pose, pose_thread_running
    while pose_thread_running:
        frame = picam2.capture_array()
        results = model.predict(frame, imgsz=imgsz, verbose=False)
        try:
            nose_x = get_keypoint_position(results, 0, 'x')
            latest_x_value = nose_x
            #pose_frame = cv2.resize(frame, (160, 160))
            pose_frame = np.zeros((160, 160, 3), np.uint8)
            if results and results[0].keypoints is not None:
                keypoints = results[0].keypoints.xyn[0].cpu().numpy()
                for x, y in keypoints:
                    cv2.circle(pose_frame, (int(x * 160), int(y * 160)), 3, (0, 255, 0), -1)
                for i, j in POSE_CONNECTIONS:
                    try:
                        x1, y1 = keypoints[i]
                        x2, y2 = keypoints[j]
                        if not ((x1==0 and y1==0) or (x2==0 and y2==0)):
                            cv2.line(pose_frame, (int(x1 * 160), int(y1 * 160)), (int(x2 * 160), int(y2 * 160)), (255, 255, 255), 1)
                    except IndexError:
                        continue
            latest_frame_pose = cv2.cvtColor(pose_frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(e)
        time.sleep(0.01)

pose_thread = threading.Thread(target=pose_detection_thread)
pose_thread.start()

running = True
shoot_cooldown = 0
clock = pygame.time.Clock()
score = 0
lives = 5

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
            running = False
        if lives <= 0 and (event.type == pygame.KEYDOWN and event.key == pygame.K_c):
            lives = 5
            score = 0
            shoot_cooldown = 0  
            bullets = []
            enemies = []
            explosions = []
    
    # Display game over screen
    if lives <= 0:
        text = font.render(f'GAME OVER', True, WHITE)
        pygame.draw.rect(screen, (0, 50, 0), (WINDOW_WIDTH//2-text.get_width()//2-5, 300-5, text.get_width() + 10, 40))
        screen.blit(text, (WINDOW_WIDTH//2-text.get_width()//2, 300))
        text = font.render("Press 'c' to continue", True, WHITE)
        pygame.draw.rect(screen, (0, 50, 0), (WINDOW_WIDTH//2-text.get_width()//2-5, 400-5, text.get_width() + 10, 40))
        screen.blit(text, (WINDOW_WIDTH//2-text.get_width()//2, 400))
            
        pygame.display.flip()
        clock.tick(60)
        continue

    target_x = int(np.interp(latest_x_value, [0.1, 0.9], [WINDOW_WIDTH - SHIP_WIDTH, 0]))
    position_history.append(target_x)
    ship_x = int(sum(position_history) / len(position_history))
    ship_x = np.clip(ship_x, 0, WINDOW_WIDTH - SHIP_WIDTH)

    if random.random() < 0.02:
        enemies.append(create_enemy())

    shoot_cooldown -= 1
    if shoot_cooldown <= 0:
        fire_bullet(ship_x)
        shoot_cooldown = 12

    for bullet in bullets[:]:
        bullet['y'] -= BULLET_SPEED
        if bullet['y'] < 0:
            bullets.remove(bullet)

    for enemy in enemies[:]:
        enemy['y'] += ENEMY_SPEED
        if enemy['y'] > WINDOW_HEIGHT:
            enemies.remove(enemy)
            lives -= 1
            continue
        enemy_center = (enemy['x'] + ENEMY_WIDTH//2, enemy['y'] + ENEMY_HEIGHT//2)
        for bullet in bullets[:]:
            distance = math.hypot(bullet['x'] - enemy_center[0], bullet['y'] - enemy_center[1])
            if distance < ENEMY_WIDTH//2:
                if bullet in bullets: bullets.remove(bullet)
                if enemy in enemies:
                    enemies.remove(enemy)
                    explosions.append({'x': enemy_center[0], 'y': enemy_center[1], 'timer': 10})
                score += 10
                break

    screen.fill((0, 0, 0))
    draw_ship(ship_x, WINDOW_HEIGHT - SHIP_HEIGHT)

    for bullet in bullets:
        pygame.draw.circle(screen, BLUE, (int(bullet['x']), int(bullet['y'])), BULLET_RADIUS + 2)
        pygame.draw.circle(screen, WHITE, (int(bullet['x']), int(bullet['y'])), BULLET_RADIUS)

    for enemy in enemies:
        draw_enemy(enemy['x'], enemy['y'])

    for explosion in explosions[:]:
        if explosion['timer'] > 0:
            draw_explosion(explosion['x'], explosion['y'])
            explosion['timer'] -= 1
        else:
            explosions.remove(explosion)

    # Draw score
    font = pygame.font.Font(None, 36)
    score_text = font.render(f'SCORE: {score}', True, GREEN)
    pygame.draw.rect(screen, (0, 50, 0), (5, 55, score_text.get_width() + 10, 40))
    screen.blit(score_text, (10, 60))

    # Draw lives
    lives_text = font.render(f'LIVES: {lives}', True, RED)
    pygame.draw.rect(screen, (0, 50, 0), (5, 5, score_text.get_width() + 10, 40))
    screen.blit(lives_text, (10, 10))

    # Draw pose overlay
    if latest_frame_pose is not None:
        pose_surface = pygame.surfarray.make_surface(np.rot90(latest_frame_pose))
        screen.blit(pose_surface, (WINDOW_WIDTH - 170, 10))
        pygame.draw.rect(screen, WHITE, (WINDOW_WIDTH - 170, 10, 160, 160), 2)
        label = pygame.font.Font(None, 20).render("Pose View", True, WHITE)
        screen.blit(label, (WINDOW_WIDTH - 160, 175))

    pygame.display.flip()
    clock.tick(60)

pose_thread_running = False
pose_thread.join()
picam2.stop()
pygame.quit()

