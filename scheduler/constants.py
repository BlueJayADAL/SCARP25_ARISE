# constants.py
"""
Constants and configuration parameters for AMRO scheduler experiments
"""

# Processor specifications
CPU_LATENCY_BASE = 100  # ms - base latency for CPU tasks
NPU_LATENCY_BASE = 10   # ms - base latency for NPU tasks

# Task type multipliers for different operations
TASK_TYPE_MULTIPLIERS = {
    'video_capture': 0.8,
    'image_preprocessing': 1.0,
    'pose_estimation': 10.5,
    'keypoint_processing': 0.6,
    'vad': 0.3,
    'asr': 2.0,
    'llm_inference': 8.0,
    'tts': 5.5,
    'audio_output': 0.4,
    'display_update': 0.5,
    'general_computation': 1.0
}

# Processor preferences for different task types
PROCESSOR_PREFERENCES = {
    'video_capture': 'CPU',
    'image_preprocessing': 'CPU',
    'pose_estimation': 'NPU',
    'keypoint_processing': 'CPU',
    'vad': 'CPU',
    'asr': 'NPU',
    'llm_inference': 'NPU',
    'tts': 'CPU',
    'audio_output': 'CPU',
    'display_update': 'CPU',
    'general_computation': 'CPU'
}

# Data transfer specifications
DATA_TRANSFER_RATE = 1000  # MB/s between CPU and NPU
DMA_OVERHEAD = 2  # ms - overhead for DMA operations

# Default MapScore weights
DEFAULT_WEIGHTS = {
    'urgency': 0.5,
    'latency_preference': 0.2,
    'starvation': 0.1,
    'criticality': 0.2
}

# Weight adjustment parameters
WEIGHT_LEARNING_RATE = 0.01
WEIGHT_MIN = 0.1
WEIGHT_MAX = 0.9

# Simulation parameters
SIMULATION_TIME = 10000  # ms
TASK_ARRIVAL_RATE = 50  # tasks per second
DEADLINE_RANGE = (2000, 4000)  # ms - range for task deadlines
DATA_SIZE_RANGE = (1, 100)  # MB - range for input/output data sizes

# Pipeline configurations
POSE_ESTIMATION_PIPELINE = [
    'video_capture',
    'image_preprocessing',
    'pose_estimation',
    'keypoint_processing'
]

CONVERSATIONAL_AI_PIPELINE = [
    'vad',
    'asr',
    'llm_inference',
    'tts',
    'audio_output'
]

# Experiment scenarios
EXPERIMENT_SCENARIOS = {
    'light_load': {
        'task_arrival_rate': 0.5,
        'pipeline_mix': {'pose': 0.6, 'conversational': 0.4}
    },
    'medium_load': {
        'task_arrival_rate': 5,
        'pipeline_mix': {'pose': 0.5, 'conversational': 0.5}
    },
    'heavy_load': {
        'task_arrival_rate': 20,
        'pipeline_mix': {'pose': 0.4, 'conversational': 0.6}
    },
    'burst_load': {
        'task_arrival_rate': 50,
        'pipeline_mix': {'pose': 0.3, 'conversational': 0.7}
    },
    'high_urgency_low_load': {
        'task_arrival_rate': 2,
        'deadline_range': (200, 400),
        'pipeline_mix': {'pose': 0.5, 'conversational': 0.5}
    },
    'processor_contention': {
        'task_arrival_rate': 30,
        'pipeline_mix': {'pose': 0.2, 'conversational': 0.8}
    },
    'dynamic_load': {
        'task_arrival_rates': [(5, 3000), (50, 4000), (10, 3000)],
        'pipeline_mix': {'pose': 0.5, 'conversational': 0.5}
    },
    'mixed_criticality': {
        'task_arrival_rate': 15,
        'pipeline_mix': {'pose': 0.5, 'conversational': 0.5},
        'criticality_mix': {'high': 0.3, 'medium': 0.4, 'low': 0.3}
    },
    'complex_multimodal': {
        'task_arrival_rate': 18,
        'pipeline_mix': {'pose': 0.5, 'conversational': 0.5},
        'criticality_mix': {'high': 0.4, 'medium': 0.4, 'low': 0.2},
        'deadline_range': (500, 1500)
    }
}

# Scheduling algorithms to compare
SCHEDULING_ALGORITHMS = [
    'AMRO',
    'FIFO',
    'Round Robin',
    'Earliest Deadline First',
    'Priority Scheduling'
]

# Performance metrics
PERFORMANCE_METRICS = [
    'average_response_time',
    'deadline_miss_rate',
    'cpu_utilization',
    'npu_utilization',
    'throughput',
    'average_waiting_time'
]