# SCARP25_ARISE
AI-Powered At-Home Exercise Systems for Older Adults

## Table of Contents
- [SCARP25\_ARISE](#scarp25_arise)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Research Plan](#research-plan)
  - [Business Plan](#business-plan)
  - [Technical Setup](#technical-setup)
  - [Models Used](#models-used)
  - [Program Startup](#program-startup)
  - [Running with Hailo-8](#running-with-hailo-8)

## Project Overview
The ARISE project aims to develop an AI-powered at-home exercise system for older adults. The system will utilize pose estimation and voice interaction to provide real-time feedback and coaching during exercise sessions. The goal is to enhance the safety and effectiveness of at-home exercise routines for elderly users.
The project will involve the integration of lightweight pose estimation models, offline automatic speech recognition, local large language models, and text-to-speech systems to create a seamless user experience. The system will be designed to run on low-power SBCs like the Raspberry Pi 5 and NVIDIA Jetson Nano, making it accessible for home use.
The project will also focus on user customization, allowing for personalized exercise routines and feedback based on individual user profiles. The final system will be tested with real users to gather feedback and improve the overall experience.
## Research Plan
The research plan outlines the tasks and goals for each week of the project. The plan is divided into 10 weeks, with specific objectives for each research assistant (RA) involved in the project. The tasks include setting up the technical environment, developing prototypes, integrating voice and pose components, and conducting user testing.
The plan also includes milestones for evaluating progress and making necessary adjustments to the project. Link to the [Research Plan](Resarch_Plan.md) for detailed tasks and deliverables for each week.
The project will be conducted in a collaborative environment, with regular meetings and updates to ensure alignment among team members. The final deliverable will be a fully functional AI-powered exercise system that can be used by older adults in their homes.
## Business Plan
1. Value proposition: 
    - Personalized, Adaptive Exercise Guidance: Tailored routines and real-time feedback for improved mobility and fall risk reduction.
    - Enhanced Privacy & Security: All processing occurs locally, eliminating cloud reliance and ensuring HIPAA compliance.
    - Accessibility & Affordability: Deployed on a cost-effective edge platform (Raspberry Pi 5 + Hailo-8 NPU).
    - Natural & Engaging Interaction: Intuitive voice-based conversational AI for ease of use.
    - Real-time Responsiveness: Minimized latency through dedicated computing accelerator and scheduling algorithm to ensure immediate feedback and a seamless user experience.
2. Customer Segments:
    - Primary: Elderly individuals (65+ years old) living independently or with family, concerned about mobility and fall prevention.
    - Secondary: Family members and caregivers of elderly individuals, seeking supportive home healthcare tools. Rehabilitation clinics and assisted living facilities.
3. A 90-second product video:
    - [![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/L8gj3mWvgJg/0.jpg)](https://www.youtube.com/watch?v=L8gj3mWvgJg)

## Technical Setup
The technical setup for the project involves the following components:
- **Hardware**: Raspberry Pi 5, camera for pose estimation, microphone for voice interaction.
- **Software**: Python, TensorRT, PyTorch, lightweight pose estimation models (YOLOv8/11-Pose), ASR (Vosk), local LLM (llama.cpp), TTS (Kokoro).
- **Libraries**: OpenCV for video processing, NumPy for numerical computations, PyTorch for deep learning.
- **Development Environment**: Jupyter Notebook for prototyping, Git for version control, Docker for containerization.
- **Testing Environment**: Simulated elderly users (faculty/friends) for initial testing, followed by real user testing.
- **Documentation**: Markdown files for project documentation, Jupyter Notebooks for code examples and tutorials.
- **Communication**: Slack or Teams for team communication and updates.
- **Backup**: Regular backups of code and data to prevent loss.
- **User Testing**: Plan for user testing sessions, including recruitment of elderly participants, consent forms, and feedback collection.
- **Feedback Loop**: Establish a feedback loop for continuous improvement based on user input and testing results.

## Models Used:
Add models to ```models/``` directory for seamless integration
- Llamma cpp LLM -> [SmolLm](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct-q4_k_m.gguf) from [Hugging Face](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF)
- VOSK STT -> [Vosk-small](https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip) from [Vosk](https://alphacephei.com/vosk/models)
- Kokoro Onnx TTS -> [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx) & [voices.bin](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin) from [GitHub](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0)
- Ultralytics Pose-tracking -> [yolo11n-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt) from [GitHub](https://github.com/ultralytics/assets/releases/tag/v8.3.0) *(Exported to OpenVino format)*
- *(Optional)* Hailo-8 Pose-tracking -> [yolov8m_pose](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov8m_pose.hef) from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)

## Program Startup
A Python virtual environment with packages listed in the `requirements.txt` is needed for running the ARISE system. Currently, the system has been tested to run on Python 3.11+

1. Create a new Python virtual environment:
    ```bash
    $ python -m venv ./ARISE_venv
    ``` 
    Then activate the environment:
    - Linux / Raspberry Pi 5:
    ```bash
    $ source ./ARISE_venv/bin/activate
    ```
    - Windows:
    ```
    ARISE_venv\Scripts\activate
    ```
2. Install requirements on your OS to the virtual environment
    ```bash
    $ pip install -r requirements.txt 
    ```    
3. Download and export the Ultralytics YOLOv11 Pose model to OpenVino format

    Export the model:
    ```bash
    $ yolo export model=yolo11n-pose.pt format=openvino imgsz=320
    ```
    Then move the output folder `yolo11n-pose_openvino_model/` into `models/` with the other downloaded models.
4. Run the ARISE system either as a conversational standalone backend or with the interactive user interface
    #### Standalone Backend
    ```bash
    $ python -m runnables.main
    ```
    #### Interactive User Interface *(must have Node.js and npm installed)*

    Navigate to the ```UI/client``` directory with:
    ```bash
    $ cd UI/client
    ```
    Then install modules and build the frontend UI:

    ```bash
    $ npm install
    $ npm run build
    ```
    or
    ```bash
    $ yarn install
    $ yarn build
    ```
    Once the build has finished from the client directory, start the server from `UI/server` directory:
    ```bash
    $ cd ../server
    ```
    Then run:
    ```bash
    $ uvicorn main:app --reload
    ```
    The web interface should be locally hosted and running on localhost. Follow the URL in the terminal to open the webpage in your browser.

## Running with Hailo-8
In order to run the ARISE system using the Hailo-8 NPU for Computer Vision inferencing, you must set up your system environment with HailoRT. Follow instructions from the [Hailo Developer Zone](https://hailo.ai/developer-zone/documentation/hailort-v4-20-0/?sp_referrer=install/install.html) to install HailoRT and pyHailoRT on your system. *Using this method to run the ARISE system does not support use of the Interactive User Interface.*

1. Ensure the [yolov8m_pose.hef](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/yolov8m_pose.hef) model is placed in your `models/` directory. 

2. Set the flag near the top of `YOLO_Pose/yolo_threaded.py` to `HAILO=1` to enable offloading of inferencing to your Hailo-8 device. 
    - Set `HAILO_METHOD=QUEUED` to improve processing speeds with queues
    - Set `HAILO_METHOD=SYNCHRONOUS` to block for inferencing on the thread

3. Run the standalone backend 
    ```bash
    $ python -m runnables.main
    ```