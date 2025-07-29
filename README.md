# SCARP25_ARISE
AI-Powered At-Home Exercise Systems for Older Adults

## Table of Contents
- [SCARP25\_ARISE](#scarp25_arise)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Research Plan](#research-plan)
  - [Business Plan](#business-plan)
  - [Technical Setup](#technical-setup)

## Project Overview
The ARISE project aims to develop an AI-powered at-home exercise system for older adults. The system will utilize pose estimation and voice interaction to provide real-time feedback and coaching during exercise sessions. The goal is to enhance the safety and effectiveness of at-home exercise routines for elderly users.
The project will involve the integration of lightweight pose estimation models, offline speech recognition, and text-to-speech systems to create a seamless user experience. The system will be designed to run on low-power devices like the NVIDIA Jetson Nano, making it accessible for home use.
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

## Program Startup
Startup needed for running the ARISE system 

1. install requirements on your OS to virtual enviornment
    - Raspberry Pi 5:
    ```
    pip install -r requirements.txt 
    ```
    - Windows:
    ```
    pip install -r win-requirements.txt
    ```
2. Run on conversational standalone backend or with interactive user interface
    - standalone backend
    ```
    python -m runnables.main
    ```
    - interactive user interface
    must have node.js installed
    ``` bash
    $ npm install
    ```
    or
    ``` bash
    $ yarn install
    ```
    then must run build in the client directory:
    ```
    $ cd client
    ```
    then
    ```bash
    $ npm run build
    ```
    or
    ```bash
    $ yarn build
    ```
    once build finished in client directory, startup from server directory
    ```
    $ cd ../server
    ```
    then run
    ```
    $ uvicorn main:app --reload
    ```
    the web interface should be locally hosted and running



