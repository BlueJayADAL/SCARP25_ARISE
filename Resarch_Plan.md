# Research Plan for ARISE Project

## Week 1: Orientation and Setup
**Goal:** Get familiar with project goals, relevant tools, and datasets.

| RA  | To-Do                                                                                                | Deliverables                                                                      |
| --- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| RA1 | - Set up Jetson Nano with TensorRT, PyTorch<br>- Survey and install lightweight pose models (YOLOv8/11-Pose, MoveNet)<br>- Identify benchmarks for FPS and accuracy | Installed toolchain, summary of model options and performance metrics             |
| RA2 | - Research ASR and TTS tools that run offline (e.g., Vosk, eSpeak)<br>- Collect example dialogue scripts for elderly exercise interaction<br>- Set up base Python voice I/O pipeline | Annotated sample utterances and initial voice input/output test scripts         |

## Week 2: Pose & Voice Prototype Initialization
**Goal:** Implement basic pose estimation and voice interaction pipeline on Jetson.

| RA  | To-Do                                                                                                | Deliverables                                                                      |
| --- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| RA1 | - Run YOLOv8-Pose or TRT-Pose on test videos<br>- Measure FPS and CPU/GPU usage at different input resolutions | Report with FPS/accuracy comparisons, pose overlay demo                             |
| RA2 | - Implement keyword-based voice command handler (start, stop, help)<br>- Build TTS playback for common instructions | Script for turn-based voice dialogue, demo video                                  |

## Week 3: Basic Exercise Loop
**Goal:** Connect pose and voice components to a basic coaching loop.

| RA  | To-Do                                                                                                   | Deliverables                                                                  |
| --- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| RA1 | - Detect start/end of an exercise movement (e.g., arm raise)<br>- Calculate and log joint angles or reps | Python script detecting reps with visualization                               |
| RA2 | - Develop dialogue flow for one full exercise (e.g., "Let's do 10 arm raises")<br>- Add fallback handling for silence or confusion | Scripted dialogue module with timing logic and retries                      |

## Week 4: Evaluation and Feedback Logic
**Goal:** Add feedback on form and spoken response based on user behavior.

| RA  | To-Do                                                                                                 | Deliverables                                                                   |
| --- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| RA1 | - Implement logic for form feedback (e.g., incomplete range)<br>- Smoothing and threshold logic for noisy detection | Script that gives real-time feedback on form accuracy                        |
| RA2 | - Implement empathetic responses ("You’re doing great!")<br>- Add error-tolerant ASR with fallback commands | Enhanced voice interaction module with 5+ coaching cases                     |

## Week 5: Intermediate Milestone + Sprint Review
**Goal:** Demonstrate full session loop: voice-prompted exercise with visual feedback.

| RA  | To-Do                                                                                                | Deliverables                                                                    |
| --- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| RA1 | - Integrate real-time video overlay of keypoints<br>- Log reps and errors for session summary            | Session tracker with rep count and simple posture chart                         |
| RA2 | - Enable voice summary at session end ("You did 8 out of 10!")<br>- Test interactions with mock elderly users (faculty/friends) | Session simulation script, preliminary user feedback notes                    |

## Week 6: Customization and User Profiles
**Goal:** Allow personalization for elderly users and error recovery.

| RA  | To-Do                                                                                                   | Deliverables                                                                    |
| --- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| RA1 | - Create config file per user (e.g., height, flexibility)<br>- Tune detection thresholds for different profiles | Configurable tracker that adjusts angles per user profile                     |
| RA2 | - Personalize voice speed, tone, vocabulary based on user profile<br>- Store conversation history and adjust prompts | Profile-based voice module with 3 user personas                               |

## Week 7: LLM/Dialogue Enhancement
**Goal:** Add natural language understanding (basic LLM or rule-based NLP).

| RA  | To-Do                                                                | Deliverables                                                              |
| --- | -------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| RA1 | - No major updates (support testing integration with RA2)            | -                                                                         |
| RA2 | - Integrate OpenRouter API (or local LLaMA) for question-answering<br>- Test dialogue prompts: “Why this exercise?”, “I’m tired” | LLM-enhanced dialogue with 3 example queries and responses                |

## Week 8: Integration & Stress Testing
**Goal:** Ensure voice and pose modules run concurrently on Jetson.

| RA  | To-Do                                                                                                   | Deliverables                                                                  |
| --- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| RA1 | - Optimize pose model with TensorRT FP16<br>- Run continuous 30-minute sessions and monitor resources     | Report on system performance, bottlenecks                                     |
| RA2 | - Test latency and accuracy during simultaneous input/output<br>- Simulate noisy environment and evaluate ASR robustness | Evaluation results under stress and voice-command timing chart              |

## Week 9: Final Demo & Evaluation
**Goal:** Showcase end-to-end system and collect usability feedback.

| RA  | To-Do                                                                                                | Deliverables                                                                   |
| --- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| RA1 | - Finalize visualization (overlay, summary graphs)<br>- Document known limitations                       | Final working code with annotated sample output                                |
| RA2 | - Conduct mock user session with voice-only interaction<br>- Collect mock feedback and adjust response logic | Demo video, final script, usability reflection memo                          |

## Week 10: Finalization and Dissemination
**Goal:** Finalize project documentation, code, and prepare for dissemination.

| RA  | To-Do                                                                                                   | Deliverables                                                                  |
| --- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| RA1 | - Perform final code cleanup and refactoring.<br>- Add comprehensive comments and documentation to the codebase.<br>- Create a README file for the project repository.<br>- Publish the project repository (e.g., on GitHub). | Cleaned and well-documented final codebase.<br>Publicly accessible project repository with README. |
| RA2 | - Draft the final project paper/report.<br>- Incorporate feedback from mock user sessions and evaluations.<br>- Finalize all figures, tables, and references for the paper.<br>- Submit the final paper/report. | Completed final project paper/report.<br>Submission confirmation (if applicable). |
