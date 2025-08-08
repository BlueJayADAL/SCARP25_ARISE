# task_model.py
"""
Task model and DAG structure for AMRO scheduler
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
import constants

class ProcessorType(Enum):
    CPU = "CPU"
    NPU = "NPU"

class TaskStatus(Enum):
    WAITING = "WAITING"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"

class Criticality(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

@dataclass
class Task:
    """
    Represents a computational task in the ARISE system
    """
    id: str
    task_type: str
    arrival_time: float
    deadline: float
    input_data_size: float  # MB
    output_data_size: float  # MB
    predecessors: Set[str]
    successors: Set[str]
    status: TaskStatus = TaskStatus.WAITING
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    assigned_processor: Optional[ProcessorType] = None
    last_scheduled_time: Optional[float] = None
    criticality: Criticality = Criticality.MEDIUM
    remaining_time: Optional[float] = None
    
    def __post_init__(self):
        if self.last_scheduled_time is None:
            self.last_scheduled_time = self.arrival_time
    
    def get_wcet(self, processor: ProcessorType) -> float:
        """Get Worst Case Execution Time for this task on given processor"""
        base_latency = constants.CPU_LATENCY_BASE if processor == ProcessorType.CPU else constants.NPU_LATENCY_BASE
        multiplier = constants.TASK_TYPE_MULTIPLIERS.get(self.task_type, 1.0)
        
        # Add some randomness to simulate real-world variation
        variation = random.uniform(0.8, 1.2)
        return base_latency * multiplier * variation
    
    def get_preferred_processor(self) -> ProcessorType:
        """Get the preferred processor for this task type"""
        pref = constants.PROCESSOR_PREFERENCES.get(self.task_type, 'CPU')
        return ProcessorType.CPU if pref == 'CPU' else ProcessorType.NPU
    
    def get_data_transfer_time(self, processor: ProcessorType) -> float:
        """Calculate data transfer time based on processor and data size"""
        if processor == self.get_preferred_processor():
            # No transfer needed if using preferred processor
            return 0.0
        
        # Transfer time includes input and output data
        total_data = self.input_data_size + self.output_data_size
        transfer_time = (total_data / constants.DATA_TRANSFER_RATE) * 1000  # Convert to ms
        return transfer_time + constants.DMA_OVERHEAD
    
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready to execute (all predecessors completed)"""
        return self.predecessors.issubset(completed_tasks)

class TaskDAG:
    """
    Directed Acyclic Graph representation of tasks and their dependencies
    """
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.completed_tasks: Set[str] = set()
        self.ready_tasks: Set[str] = set()
    
    def add_task(self, task: Task):
        """Add a task to the DAG"""
        self.tasks[task.id] = task
        if task.is_ready(self.completed_tasks):
            self.ready_tasks.add(task.id)
    
    def mark_completed(self, task_id: str):
        """Mark a task as completed and update ready tasks"""
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.COMPLETED
            self.completed_tasks.add(task_id)
            self.ready_tasks.discard(task_id)
            
            # Check if any new tasks become ready
            for tid, task in self.tasks.items():
                if (task.status == TaskStatus.WAITING and 
                    task.is_ready(self.completed_tasks)):
                    task.status = TaskStatus.READY
                    self.ready_tasks.add(tid)
    
    def get_ready_tasks(self) -> List[Task]:
        """Get all tasks that are ready to execute"""
        return [self.tasks[tid] for tid in self.ready_tasks if tid in self.tasks]
    
    def get_critical_path_work(self, task_id: str) -> float:
        """Calculate remaining work on critical path from this task"""
        if task_id not in self.tasks:
            return 0.0
        
        task = self.tasks[task_id]
        if task.status == TaskStatus.COMPLETED:
            return 0.0
        
        # Simple estimation: minimum WCET of this task plus max of successors
        min_wcet = min(task.get_wcet(ProcessorType.CPU), task.get_wcet(ProcessorType.NPU))
        
        if not task.successors:
            return min_wcet
        
        max_successor_work = 0.0
        for successor_id in task.successors:
            successor_work = self.get_critical_path_work(successor_id)
            max_successor_work = max(max_successor_work, successor_work)
        
        return min_wcet + max_successor_work

def create_pipeline_tasks(pipeline_type: str, start_time: float, pipeline_id: str, 
                          deadline_range: tuple, criticality: Criticality) -> List[Task]:
    """Create a sequence of tasks for a given pipeline"""
    tasks = []
    
    if pipeline_type == 'pose':
        task_sequence = constants.POSE_ESTIMATION_PIPELINE
    elif pipeline_type == 'conversational':
        task_sequence = constants.CONVERSATIONAL_AI_PIPELINE
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

    # Set a single deadline for the entire pipeline
    pipeline_deadline = start_time + random.uniform(*deadline_range)
    
    # Create tasks with dependencies
    for i, task_type in enumerate(task_sequence):
        task_id = f"{pipeline_id}_{task_type}_{i}"
        
        # Set data sizes
        input_size = random.uniform(*constants.DATA_SIZE_RANGE)
        output_size = random.uniform(*constants.DATA_SIZE_RANGE)
        
        # Set predecessors and successors
        predecessors = set()
        successors = set()
        
        if i > 0:
            predecessors.add(f"{pipeline_id}_{task_sequence[i-1]}_{i-1}")
        if i < len(task_sequence) - 1:
            successors.add(f"{pipeline_id}_{task_sequence[i+1]}_{i+1}")
        
        task = Task(
            id=task_id,
            task_type=task_type,
            arrival_time=start_time,
            deadline=pipeline_deadline,  # All tasks in pipeline share the same deadline
            input_data_size=input_size,
            output_data_size=output_size,
            predecessors=predecessors,
            successors=successors,
            criticality=criticality
        )
        
        tasks.append(task)
    
    return tasks

def generate_workload(scenario: str, duration: float) -> List[Task]:
    """Generate a workload based on the given scenario"""
    config = constants.EXPERIMENT_SCENARIOS[scenario]
    pipeline_mix = config['pipeline_mix']
    deadline_range = config.get('deadline_range', constants.DEADLINE_RANGE)
    
    tasks = []
    pipeline_counter = 0
    
    if 'task_arrival_rates' in config:  # Dynamic load scenario
        current_time = 0.0
        for arrival_rate, time_duration in config['task_arrival_rates']:
            segment_end_time = current_time + time_duration
            while current_time < segment_end_time:
                inter_arrival = random.expovariate(arrival_rate / 1000.0)
                current_time += inter_arrival
                
                if current_time >= segment_end_time:
                    break
                
                pipeline_type = 'pose' if random.random() < pipeline_mix['pose'] else 'conversational'
                criticality = Criticality.MEDIUM # Default for this scenario
                
                pipeline_tasks = create_pipeline_tasks(
                    pipeline_type, current_time, f"{pipeline_type}_{pipeline_counter}",
                    deadline_range, criticality
                )
                tasks.extend(pipeline_tasks)
                pipeline_counter += 1
    else:  # Static load scenarios
        arrival_rate = config['task_arrival_rate']
        criticality_mix = config.get('criticality_mix')

        current_time = 0.0
        while current_time < duration:
            inter_arrival = random.expovariate(arrival_rate / 1000.0)
            current_time += inter_arrival
            
            if current_time >= duration:
                break
            
            pipeline_type = 'pose' if random.random() < pipeline_mix['pose'] else 'conversational'
            
            if criticality_mix:
                rand_val = random.random()
                if rand_val < criticality_mix['high']:
                    criticality = Criticality.HIGH
                elif rand_val < criticality_mix['high'] + criticality_mix['medium']:
                    criticality = Criticality.MEDIUM
                else:
                    criticality = Criticality.LOW
            else:
                criticality = Criticality.MEDIUM

            pipeline_tasks = create_pipeline_tasks(
                pipeline_type, current_time, f"{pipeline_type}_{pipeline_counter}",
                deadline_range, criticality
            )
            tasks.extend(pipeline_tasks)
            pipeline_counter += 1
            
    return tasks

if __name__ == "__main__":
    # Example usage
    print("Generating example workload for 'heavy_load' scenario...")
    workload = generate_workload('heavy_load', 1000)
    
    print(f"Generated {len(workload)} tasks.")
    
    # Print details of first few tasks
    for task in workload[:5]:
        print(f"  Task ID: {task.id}")
        print(f"    Type: {task.task_type}")
        print(f"    Arrival: {task.arrival_time:.2f} ms")
        print(f"    Deadline: {task.deadline:.2f} ms")
        print(f"    Criticality: {task.criticality.value}")
        print(f"    Preferred Processor: {task.get_preferred_processor().value}")
        print(f"    Predecessors: {task.predecessors}")
        print(f"    Successors: {task.successors}")
        print("-" * 20)
