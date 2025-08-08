# amro_scheduler.py
"""
Implementation of the Adaptive Multimodal Real-time Orchestrator (AMRO) scheduler
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import constants
from task_model import Task, TaskDAG, ProcessorType, TaskStatus, Criticality

@dataclass
class ProcessorState:
    """State of a processor"""
    is_busy: bool = False
    current_task: Optional[str] = None
    available_at: float = 0.0

class AMROScheduler:
    """
    Adaptive Multimodal Real-time Orchestrator scheduler implementation
    """
    
    def __init__(self):
        self.weights = constants.DEFAULT_WEIGHTS.copy()
        self.cpu_state = ProcessorState()
        self.npu_state = ProcessorState()
        self.completed_tasks_window = []  # For performance tracking
        self.window_size = 50  # Number of recent tasks to consider
        self.preemptive = True # AMRO is preemptive
        
    def calculate_urgency(self, task: Task, current_time: float, task_dag: TaskDAG) -> float:
        """Calculate urgency score for a task"""
        # Estimated remaining work on critical path
        remaining_work = task_dag.get_critical_path_work(task.id)
        
        # Calculate slack time
        elapsed_time = current_time - task.arrival_time
        slack = task.deadline - task.arrival_time - elapsed_time - remaining_work
        
        # Avoid division by zero
        slack = max(1.0, slack)
        
        return remaining_work / slack
    
    def calculate_latency_preference(self, task: Task, processor: ProcessorType) -> float:
        """Calculate latency preference score for a task on a processor"""
        est_latency = task.get_wcet(processor)
        data_transfer_time = task.get_data_transfer_time(processor)
        
        total_latency = est_latency + data_transfer_time
        
        # Return inverse of latency (higher score for lower latency)
        return 1.0 / max(1.0, total_latency)
    
    def calculate_starvation(self, task: Task, current_time: float) -> float:
        """Calculate starvation score for a task"""
        time_since_last_scheduled = current_time - task.last_scheduled_time
        preferred_processor = task.get_preferred_processor()
        est_latency = task.get_wcet(preferred_processor)
        
        return time_since_last_scheduled / max(1.0, est_latency)
    
    def calculate_criticality_score(self, task: Task) -> float:
        """Calculate criticality score for a task"""
        criticality_map = {
            Criticality.HIGH: 1.0,
            Criticality.MEDIUM: 0.5,
            Criticality.LOW: 0.1
        }
        return criticality_map.get(task.criticality, 0.5)

    def calculate_map_score(self, task: Task, processor: ProcessorType, 
                          current_time: float, task_dag: TaskDAG) -> float:
        """Calculate the Adaptive MapScore for a task on a processor"""
        urgency = self.calculate_urgency(task, current_time, task_dag)
        latency_pref = self.calculate_latency_preference(task, processor)
        starvation = self.calculate_starvation(task, current_time)
        criticality = self.calculate_criticality_score(task)
        
        map_score = (self.weights['urgency'] * urgency +
                    self.weights['latency_preference'] * latency_pref +
                    self.weights['starvation'] * starvation +
                    self.weights['criticality'] * criticality)
        
        return map_score
    
    def get_processor_state(self, processor: ProcessorType) -> ProcessorState:
        """Get the state of a processor"""
        if processor == ProcessorType.CPU:
            return self.cpu_state
        else:
            return self.npu_state
    
    def select_best_task(self, ready_tasks: List[Task], current_time: float, 
                        task_dag: TaskDAG, processor: ProcessorType) -> Optional[Tuple[Task, ProcessorType]]:
        """Select the best task to schedule next"""
        best_task = None
        best_score = -1.0

        state = self.get_processor_state(processor)
        if state.is_busy:
            running_task = task_dag.tasks[state.current_task]
            best_score = self.calculate_map_score(running_task, processor, current_time, task_dag)
            best_task = running_task
        
        for task in ready_tasks:
            if task.get_preferred_processor() == processor:
                score = self.calculate_map_score(task, processor, current_time, task_dag)
                if score > best_score:
                    best_score = score
                    best_task = task
        
        return (best_task, processor) if best_task else None
    
    def update_weights(self, current_time: float):
        """Update weights based on recent performance"""
        if len(self.completed_tasks_window) < 10:
            return
        
        # Calculate cost (deadline misses)
        total_cost = 0.0
        for task_info in self.completed_tasks_window:
            finish_time = task_info['finish_time']
            deadline = task_info['deadline']
            if finish_time > deadline:
                total_cost += finish_time - deadline
        
        # Simple heuristic adjustment
        if total_cost > 0:
            # Increase urgency weight if we're missing deadlines
            self.weights['urgency'] = min(constants.WEIGHT_MAX, 
                                        self.weights['urgency'] + constants.WEIGHT_LEARNING_RATE)
            self.weights['latency_preference'] = max(constants.WEIGHT_MIN,
                                                   self.weights['latency_preference'] - constants.WEIGHT_LEARNING_RATE/2)
        else:
            # Decrease urgency weight if we're meeting deadlines
            self.weights['urgency'] = max(constants.WEIGHT_MIN,
                                        self.weights['urgency'] - constants.WEIGHT_LEARNING_RATE/2)
            self.weights['latency_preference'] = min(constants.WEIGHT_MAX,
                                                   self.weights['latency_preference'] + constants.WEIGHT_LEARNING_RATE/2)
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total_weight
    
    def run_simulation(self, tasks: List[Task], duration: float) -> Dict:
        """Run the AMRO scheduler simulation"""
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
                        
                        # Add to completed tasks window for weight adjustment
                        self.completed_tasks_window.append(stats['task_details'][-1])
                        if len(self.completed_tasks_window) > self.window_size:
                            self.completed_tasks_window.pop(0)

                        state.is_busy = False
                        state.current_task = None

                # 3. Select next task if processor is idle or preemption is enabled
                if not state.is_busy or self.preemptive:
                    ready_tasks = task_dag.get_ready_tasks()
                    if ready_tasks:
                        selection = self.select_best_task(ready_tasks, current_time, task_dag, processor)
                        
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
            
            # Update weights periodically
            if int(current_time) % 1000 == 0:  # Every second
                self.update_weights(current_time)

            current_time += 1.0
        
        # Calculate final statistics
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
        stats['throughput'] = stats['completed_tasks'] / (duration / 1000.0)  # tasks per second
        
        return stats
