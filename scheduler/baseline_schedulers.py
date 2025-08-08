# baseline_schedulers.py
"""
Implementation of baseline scheduling algorithms for comparison with AMRO
"""

import heapq
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import random
from task_model import Task, TaskDAG, ProcessorType, TaskStatus, Criticality
import collections

@dataclass
class ProcessorState:
    is_busy: bool = False
    current_task: Optional[str] = None
    
class BaseScheduler:
    """Base class for all schedulers, now with preemption support"""
    
    def __init__(self):
        self.cpu_state = ProcessorState()
        self.npu_state = ProcessorState()
        self.preemptive = False

    def get_processor_state(self, processor: ProcessorType) -> ProcessorState:
        return self.cpu_state if processor == ProcessorType.CPU else self.npu_state

    def run_simulation(self, tasks: List[Task], duration: float) -> Dict:
        task_dag = TaskDAG()
        for task in tasks:
            task_dag.add_task(task)
        
        tasks.sort(key=lambda t: t.arrival_time)
        
        stats = {
            'completed_tasks': 0, 'deadline_misses': 0,
            'total_response_time': 0.0, 'total_waiting_time': 0.0,
            'cpu_busy_time': 0.0, 'npu_busy_time': 0.0,
            'task_details': []
        }
        
        current_time = 0.0
        task_arrival_index = 0
        
        while current_time < duration:
            # 1. Add newly arrived tasks to the ready queue
            while (task_arrival_index < len(tasks) and 
                   tasks[task_arrival_index].arrival_time <= current_time):
                task = tasks[task_arrival_index]
                if task.is_ready(task_dag.completed_tasks):
                    task.status = TaskStatus.READY
                    task_dag.ready_tasks.add(task.id)
                task_arrival_index += 1

            # 2. Process each processor
            for processor in [ProcessorType.CPU, ProcessorType.NPU]:
                state = self.get_processor_state(processor)
                
                # If processor is busy, update task progress
                if state.is_busy:
                    running_task = task_dag.tasks[state.current_task]
                    running_task.remaining_time -= 1
                    stats[f'{processor.value.lower()}_busy_time'] += 1
                    
                    if running_task.remaining_time <= 0:
                        # Task completion
                        running_task.finish_time = current_time
                        running_task.status = TaskStatus.COMPLETED
                        task_dag.mark_completed(running_task.id)
                        
                        stats['completed_tasks'] += 1
                        if running_task.finish_time > running_task.deadline:
                            stats['deadline_misses'] += 1
                        
                        response_time = max(0, running_task.finish_time - running_task.arrival_time)
                        waiting_time = max(0, (running_task.finish_time - running_task.arrival_time) - (running_task.get_wcet(processor) + running_task.get_data_transfer_time(processor)))

                        stats['total_response_time'] += response_time
                        stats['total_waiting_time'] += waiting_time
                        stats['task_details'].append({
                            'task_id': running_task.id, 'finish_time': running_task.finish_time,
                            'start_time': running_task.start_time,
                            'deadline': running_task.deadline, 'deadline_missed': running_task.finish_time > running_task.deadline,
                            'response_time': response_time,
                            'waiting_time': waiting_time,
                            'task_type': running_task.task_type,
                            'processor': processor.value
                        })
                        
                        state.is_busy = False
                        state.current_task = None

                # 3. Select next task if processor is idle or preemption is enabled
                if not state.is_busy or self.preemptive:
                    ready_tasks = task_dag.get_ready_tasks()
                    if ready_tasks:
                        selection = self.select_next_task(ready_tasks, current_time, task_dag, processor)
                        
                        if selection:
                            next_task, _ = selection
                            
                            if state.is_busy and state.current_task != next_task.id:
                                # Preemption
                                preempted_task = task_dag.tasks[state.current_task]
                                preempted_task.status = TaskStatus.READY
                                task_dag.ready_tasks.add(preempted_task.id)

                            if not state.is_busy or state.current_task != next_task.id:
                                # Schedule the new task
                                state.is_busy = True
                                state.current_task = next_task.id
                                next_task.status = TaskStatus.RUNNING
                                next_task.assigned_processor = processor
                                task_dag.ready_tasks.discard(next_task.id)
                                
                                if next_task.start_time is None:
                                    next_task.start_time = current_time
                                if next_task.remaining_time is None:
                                    next_task.remaining_time = next_task.get_wcet(processor) + next_task.get_data_transfer_time(processor)

            current_time += 1.0

        # Final statistics calculation
        if stats['completed_tasks'] > 0:
            stats['average_response_time'] = stats['total_response_time'] / stats['completed_tasks']
            stats['average_waiting_time'] = stats['total_waiting_time'] / stats['completed_tasks']
            stats['deadline_miss_rate'] = stats['deadline_misses'] / stats['completed_tasks']
        else:
            stats['average_response_time'] = 0.0
            stats['average_waiting_time'] = 0.0
            stats['deadline_miss_rate'] = 0.0
            
        stats['cpu_utilization'] = stats['cpu_busy_time'] / duration
        stats['npu_utilization'] = stats['npu_busy_time'] / duration
        stats['throughput'] = stats['completed_tasks'] / (duration / 1000.0)
        
        return stats

    def select_next_task(self, ready_tasks: List[Task], current_time: float, 
                        task_dag: TaskDAG, processor: ProcessorType) -> Optional[Tuple[Task, ProcessorType]]:
        raise NotImplementedError

class FIFOScheduler(BaseScheduler):
    def select_next_task(self, ready_tasks: List[Task], current_time: float, 
                        task_dag: TaskDAG, processor: ProcessorType) -> Optional[Tuple[Task, ProcessorType]]:
        if self.get_processor_state(processor).is_busy:
            return None
        ready_tasks.sort(key=lambda t: t.arrival_time)
        for task in ready_tasks:
            if task.get_preferred_processor() == processor:
                return (task, processor)
        return None

class RoundRobinScheduler(BaseScheduler):
    def __init__(self, time_quantum: float = 50.0):
        super().__init__()
        # This is a simplified RR for the new model, true RR is complex with preemption
        self.preemptive = False 

    def select_next_task(self, ready_tasks: List[Task], current_time: float, 
                        task_dag: TaskDAG, processor: ProcessorType) -> Optional[Tuple[Task, ProcessorType]]:
        if self.get_processor_state(processor).is_busy:
            return None
        ready_tasks.sort(key=lambda t: t.arrival_time)
        for task in ready_tasks:
            if task.get_preferred_processor() == processor:
                return (task, processor)
        return None

class EarliestDeadlineFirstScheduler(BaseScheduler):
    def select_next_task(self, ready_tasks: List[Task], current_time: float, 
                        task_dag: TaskDAG, processor: ProcessorType) -> Optional[Tuple[Task, ProcessorType]]:
        if self.get_processor_state(processor).is_busy:
            return None
        
        ready_tasks.sort(key=lambda t: t.deadline)
        for task in ready_tasks:
            if task.get_preferred_processor() == processor:
                return (task, processor)
        return None

class PriorityScheduler(BaseScheduler):
    def __init__(self):
        super().__init__()
        self.preemptive = True

    def calculate_priority(self, task: Task, current_time: float) -> float:
        time_to_deadline = task.deadline - current_time
        urgency = 1.0 / max(1.0, time_to_deadline)
        criticality_multiplier = {Criticality.HIGH: 1.5, Criticality.MEDIUM: 1.0, Criticality.LOW: 0.5}
        return urgency * criticality_multiplier.get(task.criticality, 1.0)

    def select_next_task(self, ready_tasks: List[Task], current_time: float, 
                        task_dag: TaskDAG, processor: ProcessorType) -> Optional[Tuple[Task, ProcessorType]]:
        best_task = None
        highest_priority = -1.0

        state = self.get_processor_state(processor)
        if state.is_busy:
            running_task = task_dag.tasks[state.current_task]
            highest_priority = self.calculate_priority(running_task, current_time)
            best_task = running_task

        for task in ready_tasks:
            if task.get_preferred_processor() == processor:
                priority = self.calculate_priority(task, current_time)
                if priority > highest_priority:
                    highest_priority = priority
                    best_task = task
        
        return (best_task, processor) if best_task else None

def create_scheduler(algorithm_name: str):
    if algorithm_name == 'FIFO':
        return FIFOScheduler()
    elif algorithm_name == 'Round Robin':
        return RoundRobinScheduler()
    elif algorithm_name == 'Earliest Deadline First':
        return EarliestDeadlineFirstScheduler()
    elif algorithm_name == 'Priority Scheduling':
        return PriorityScheduler()
    else:
        raise ValueError(f"Unknown scheduler: {algorithm_name}")
