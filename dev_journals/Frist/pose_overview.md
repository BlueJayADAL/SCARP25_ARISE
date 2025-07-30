# Directory-Level Code Structure and Functional Overview

This summary covers the main files and internal structure within the following directories:
- `YOLO_Pose`
- `YOLO_Pose/hailo`
- `UI/server/ARISE_vision_API`

---

## YOLO_Pose

### 1. `yolo_threaded.py`
- **Purpose:** Central threaded loop for pose estimation, exercise rep counting, and form checking using YOLO models. Supports both OpenCV and PiCamera, and optionally Hailo hardware.
- **Core Components:**
  - **Global Config:** Defines camera type, model paths, debug flags, etc.
  - **SharedState:** Thread-safe class for sharing runtime state and data across threads.
  - **Key Functions:**
    - `thread_main(shared_data, ...)`
      - Main pose tracking loop: Captures frames, runs YOLO pose model, calculates joint angles, checks exercise logic, updates rep count and form, and handles UI messages.
      - Handles user input to select exercise (`bicep curl`, `squat`, `arm raise`, `lunge`).
      - Logs rep and form data if enabled.
      - Supports pause, reset, and ROM adjustment via shared state.
      - Cleans up camera and windows at thread end.
    - Helper functions: `calculate_angle`, `A`, `display_text`, `adjust_ROM`.
  - **Data:** Uses `coords` (joint coordinates), `angles` (joint angles), `reps`, `bad_form_times`, and more, all stored/updated in `SharedState`.

### 2. `exercise_forms.py`
- **Purpose:** Provides utilities to check form for different exercises and individual joints.
- **Core Functions:**
  - `check_bad_form(current_exercise, coords, angles, dims, side)`
    - Returns a list of form issues for the current exercise (e.g., “KEEP_BACK_STRAIGHT”).
  - Specific form checks: `check_elbows_close_to_body`, `check_arms_straight`, `check_head_up`, etc.
  - Rep counters: `check_bicep_curl_rep`, `check_arm_raise_rep`, `check_squat_rep`, `check_lunge_rep`, and generic `check_rep`.

### 3. `shared_data.py`
- **Purpose:** Thread-safe storage for all shared runtime data (e.g., rep count, form flags, exercise state).
- **Class:**
  - `SharedState`
    - Methods: `set_value(key, value)`, `get_value(key)`, `delete_value(key)`, `get_all_data()`
    - Uses locks for thread safety.
  - Singleton instance: `shared_state`
  - Helper: `log_message(msg)`

### 4. `pose_demo.py`
- **Purpose:** Minimal example for running YOLO pose on frames from PiCamera and visualizing results in real-time.
- **Main Logic:** Captures frames, runs YOLO prediction, plots results, displays FPS overlay.

### 5. `space_invader_overlay.py`
- **Purpose:** Implements a “Nose Space Invaders” game using pose estimation for control.
- **Key Classes/Functions:**
  - Pygame game loop: ship/enemy/bullet/explosion logic.
  - Pose detection thread: Uses YOLO pose to update ship location based on nose position.
  - Drawing functions: `draw_ship`, `draw_enemy`, `draw_explosion`, and pose overlay.
  - Uses PiCamera for input and overlays pose on game window.

---

## YOLO_Pose/hailo

### 1. `hailo_pose_threaded.py`
- **Purpose:** Threaded pose estimation using Hailo hardware, similar to `yolo_threaded.py` but for Hailo.
- **Core Structure:**
  - Global config and imports for Hailo hardware.
  - `thread_main(shared_data, ...)`:
    - Spawns inference thread for Hailo.
    - Handles pose detection, form/rep logic, logging, and shared state updates.
    - Contains nearly identical logic to CPU version but with Hailo-specific inference paths.
  - **HailoSyncInference class:**
    - Provides threaded inference routines, post-processing, and single-inference interface.

### 2. `hailo_pose_util.py`
- **Purpose:** Hailo hardware abstraction and utility code.
- **Key Functions:**
  - `hailo_init`: Sets up Hailo model, post-processing, and threading.
  - `get_postprocess`, `postprocess`, `hailo_sync_infer`: Manage inference queue and post-processing.
  - **HailoSyncInference class:**
    - Methods for running and post-processing pose inference (threaded and single-shot).
    - Handles hardware setup, buffer management, and result visualization.

### 3. `pose_estimation_utils.py`
- **Purpose:** Post-processing and visualization for Hailo pose estimation.
- **Key Class:**
  - `PoseEstPostProcessing`
    - Methods for post-processing raw model outputs, decoding poses, non-max suppression, and visualization.

### 4. `github_test_pose.py`
- **Purpose:** Test harness for running Hailo pose estimation and visualizing results.
- **Key Functions:** Frame preprocessing, keypoint extraction, visualization, and handling inference buffers.

### 5. `pose_estimation_synchronous_camera.py`
- **Purpose:** Standalone script for synchronous pose estimation on Hailo hardware.
- **Key Functions:** Main loop to run inference, display pose results, and support command-line arguments for hardware paths.

---

## UI/server/ARISE_vision_API

### 1. `yolo_vision.py`
- **Purpose:** Core API backend logic for pose detection, exercise feedback, and state management.
- **Key Functions:**
  - `init_yolo(exercise=None)`: Initializes all global state, model, thresholds, and shared data.
  - `arise_vision(frame)`: Main logic to process incoming frames, run pose estimation, check exercise logic, update reps, form, and shared data for frontend.
  - Helper functions: `calculate_angle`, `A`, `adjust_ROM`, and inline `reset_bad_form_times`.

### 2. `exercise_forms.py`
- **Purpose:** Contains utilities for form checking and rep counting, similar to `YOLO_Pose/exercise_forms.py` but adapted for API/server context.
- **Main Functions:** Same as above; includes `check_bad_form`, rep checkers, and many single-joint checks.

### 3. `shared_data.py`
- **Purpose:** Thread-safe data container for API server state.
- **Class:** `SharedState` (identical to YOLO_Pose version), plus global singleton and logging helper.

### 4. `games.py`
- **Purpose:** Implements backend logic for pose-driven games (e.g., Space Invaders).
- **Core Functions:**
  - `init_model()`: Loads YOLO pose model.
  - `run_game_frame(frame)`: Main game logic—updates ship/enemy/bullet positions based on user pose, runs collision detection, handles lives/score, and prepares state for frontend.
  - Utility functions for keypoint extraction, enemy/bullet creation, etc.

### 5. `ws_pose.py`
- **Purpose:** WebSocket endpoints for real-time pose and game API.
- **Key Functions:**
  - `websocket_endpoint`: Handles incoming frames, runs `arise_vision`, and broadcasts results (including formatted keypoints) to clients.
  - `websocket_space_invaders`: Similar, but runs `run_game_frame` and sends game state.

---

## Shared Data, Core Classes, and Patterns

- **SharedState** (in both YOLO_Pose and API):
  - Used everywhere for thread-safe, dynamic sharing of runtime data (reps, angles, exercise status, etc).
- **Exercise Form Checking**:
  - Both the pose and API directories use a robust set of utilities to check exercise form and to update rep counts based on keypoint and angle data.
- **YOLO Model Use**:
  - All pose estimation functions rely on Ultralytics YOLO models for keypoint detection.
- **Game Integration**:
  - The “Space Invaders” game is implemented both as a Pygame application and as a backend+WebSocket API for the web UI, both driven by nose keypoint position.

---

## Summary Table

| File/Dir                                    | Purpose/Role                                 | Main Classes & Functions               |
|---------------------------------------------|----------------------------------------------|----------------------------------------|
| YOLO_Pose/yolo_threaded.py                  | Main threaded pose tracking/rep loop         | `thread_main`, helpers, `SharedState`  |
| YOLO_Pose/exercise_forms.py                 | Form/rep checking utilities                  | `check_bad_form`, `check_rep`, etc.    |
| YOLO_Pose/shared_data.py                    | Thread-safe shared state                     | `SharedState`                          |
| YOLO_Pose/pose_demo.py                      | Simple pose demo/visualization               | --                                     |
| YOLO_Pose/space_invader_overlay.py          | Pygame Space Invaders controller             | Game loop, pose overlay, helpers       |
| YOLO_Pose/hailo/hailo_pose_threaded.py      | Hailo-specific threaded pose loop            | `thread_main`, `HailoSyncInference`    |
| YOLO_Pose/hailo/hailo_pose_util.py          | Hailo hardware/model utility                 | `hailo_init`, `HailoSyncInference`     |
| YOLO_Pose/hailo/pose_estimation_utils.py    | Hailo pose post-processing                   | `PoseEstPostProcessing`                |
| YOLO_Pose/hailo/github_test_pose.py         | Hailo test script                            | Pose extraction/visualization          |
| YOLO_Pose/hailo/pose_estimation_synchronous_camera.py | Hailo script for sync camera use          | `HailoSyncInference`, `main`           |
| UI/server/ARISE_vision_API/yolo_vision.py   | Main API vision logic                        | `init_yolo`, `arise_vision`            |
| UI/server/ARISE_vision_API/exercise_forms.py| API-side form/rep checking                   | Same as above                          |
| UI/server/ARISE_vision_API/shared_data.py   | API shared state class                       | `SharedState`                          |
| UI/server/ARISE_vision_API/games.py         | Backend for pose-driven games                | `run_game_frame`, `init_model`         |
| UI/server/ARISE_vision_API/ws_pose.py       | WebSocket endpoints for vision/game          | `websocket_endpoint`, etc.             |

---

This README provides a high-level map of the major logic blocks, main classes, and how data flows between pose estimation, exercise logic, and interactive games within these core directories.