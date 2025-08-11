# experiment_runner.py
"""
Experiment runner for comparing AMRO scheduler with baseline algorithms
"""

import random
import time
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

import constants
from task_model import generate_workload
from amro_scheduler import AMROScheduler
from baseline_schedulers import create_scheduler

class ExperimentRunner:
    """Run experiments comparing different scheduling algorithms"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.results = {}
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def run_single_experiment(self, algorithm: str, scenario: str, 
                            duration: float = None) -> Dict:
        """Run a single experiment with given algorithm and scenario"""
        if duration is None:
            duration = constants.SIMULATION_TIME
        
        # Generate workload
        tasks = generate_workload(scenario, duration)
        
        # Create scheduler
        if algorithm == 'AMRO':
            scheduler = AMROScheduler()
        else:
            scheduler = create_scheduler(algorithm)
        
        # Run simulation
        start_time = time.time()
        results = scheduler.run_simulation(tasks, duration)
        end_time = time.time()
        
        # Add metadata
        results['algorithm'] = algorithm
        results['scenario'] = scenario
        results['duration'] = duration
        results['total_tasks'] = len(tasks)
        results['simulation_time'] = end_time - start_time
        
        return results
    
    def run_comprehensive_experiments(self) -> Dict:
        """Run comprehensive experiments across all algorithms and scenarios"""
        print("Running comprehensive scheduler comparison experiments...")
        
        results = defaultdict(lambda: defaultdict(list))
        
        # Run multiple iterations for statistical significance
        num_iterations = 5
        
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}/{num_iterations}")
            
            for scenario in constants.EXPERIMENT_SCENARIOS.keys():
                print(f"  Running scenario: {scenario}")
                
                for algorithm in constants.SCHEDULING_ALGORITHMS:
                    print(f"    Algorithm: {algorithm}")
                    
                    # Set different seed for each iteration
                    random.seed(self.random_seed + iteration * 100)
                    np.random.seed(self.random_seed + iteration * 100)
                    
                    try:
                        result = self.run_single_experiment(algorithm, scenario)
                        results[scenario][algorithm].append(result)
                    except Exception as e:
                        print(f"    Error running {algorithm}: {e}")
                        continue
        
        self.results = results
        return results
    
    def calculate_statistics(self) -> Dict:
        """Calculate summary statistics from experimental results"""
        stats = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        
        for scenario, algorithms in self.results.items():
            for algorithm, runs in algorithms.items():
                if not runs:
                    continue
                
                for metric in constants.PERFORMANCE_METRICS:
                    values = [run.get(metric, 0) for run in runs]
                    stats[scenario][algorithm][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values)
                    }
        
        return stats
    
    def create_performance_plots(self, save_plots: bool = True):
        """Create comprehensive performance comparison plots"""
        if not self.results:
            print("No results to plot. Run experiments first.")
            return
        
        stats = self.calculate_statistics()
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('AMRO Scheduler Performance Comparison', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Metrics to plot
        metrics = [
            ('average_response_time', 'Average Response Time (ms)'),
            ('deadline_miss_rate', 'Deadline Miss Rate'),
            ('cpu_utilization', 'CPU Utilization'),
            ('npu_utilization', 'NPU Utilization'),
            ('throughput', 'Throughput (tasks/sec)'),
            ('average_waiting_time', 'Average Waiting Time (ms)')
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx]
            
            # Prepare data for plotting
            scenarios = list(stats.keys())
            algorithms = constants.SCHEDULING_ALGORITHMS
            
            # Create bar chart data
            bar_data = []
            bar_labels = []
            bar_colors = []
            
            color_map = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
            
            for i, scenario in enumerate(scenarios):
                for j, algorithm in enumerate(algorithms):
                    if algorithm in stats[scenario] and metric in stats[scenario][algorithm]:
                        value = stats[scenario][algorithm][metric]['mean']
                        error = stats[scenario][algorithm][metric]['std']
                        
                        bar_data.append(value)
                        bar_labels.append(f"{scenario}\n{algorithm}")
                        bar_colors.append(color_map[j])
            
            # Create grouped bar chart
            x_pos = np.arange(len(scenarios))
            width = 0.12
            
            for j, algorithm in enumerate(algorithms):
                values = []
                errors = []
                
                for scenario in scenarios:
                    if algorithm in stats[scenario] and metric in stats[scenario][algorithm]:
                        values.append(stats[scenario][algorithm][metric]['mean'])
                        errors.append(stats[scenario][algorithm][metric]['std'])
                    else:
                        values.append(0)
                        errors.append(0)
                
                ax.bar(x_pos + j * width, values, width, 
                      label=algorithm, color=color_map[j], alpha=0.8,
                      yerr=errors, capsize=3)
            
            ax.set_xlabel('Scenarios')
            ax.set_ylabel(title)
            ax.set_title(f'{title} by Algorithm and Scenario')
            ax.set_xticks(x_pos + width * (len(algorithms) - 1) / 2)
            ax.set_xticklabels(scenarios, rotation=45, ha='right')
            
            if idx == 0:  # Add legend to first subplot
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Format y-axis based on metric type
            if 'rate' in metric or 'utilization' in metric:
                ax.set_ylim(0, 1.1)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('scheduler_performance_comparison.png', dpi=300, bbox_inches='tight')
            print("Performance comparison plot saved as 'scheduler_performance_comparison.png'")
        
        plt.show()
    
    def create_detailed_analysis_plots(self, save_plots: bool = True):
        """Create detailed analysis plots for AMRO scheduler"""
        if not self.results:
            print("No results to plot. Run experiments first.")
            return
        
        amro_results = []
        for scenario, algorithms in self.results.items():
            if 'AMRO' in algorithms:
                for run in algorithms['AMRO']:
                    for task_detail in run.get('task_details', []):
                        task_detail['scenario'] = scenario
                        amro_results.append(task_detail)
        
        if not amro_results:
            print("No AMRO results found for detailed analysis.")
            return
        
        df = pd.DataFrame(amro_results)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('AMRO Scheduler Detailed Analysis', fontsize=16, fontweight='bold')
        
        # 1. Response time distribution by scenario
        ax1 = axes[0]
        for scenario in df['scenario'].unique():
            scenario_data = df[df['scenario'] == scenario]['response_time']
            ax1.hist(scenario_data, alpha=0.7, label=scenario, bins=30)
        ax1.set_xlabel('Response Time (ms)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Response Time Distribution by Scenario')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Deadline miss analysis
        ax2 = axes[1]
        deadline_miss_by_scenario = df.groupby('scenario')['deadline_missed'].mean()
        deadline_miss_by_scenario.plot(kind='bar', ax=ax2, color='red', alpha=0.7)
        ax2.set_xlabel('Scenario')
        ax2.set_ylabel('Deadline Miss Rate')
        ax2.set_title('Deadline Miss Rate by Scenario')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('amro_detailed_analysis.png', dpi=300, bbox_inches='tight')
            print("Detailed analysis plot saved as 'amro_detailed_analysis.png'")
        
        plt.show()

    def create_timeline_plots(self, save_plots: bool = True):
        """Create timeline plots for each scenario"""
        print("DEBUG: Entering create_timeline_plots") # Debug print
        if not self.results:
            print("No results to plot. Run experiments first.")
            return

        for scenario in self.results:
            print(f"DEBUG: Processing scenario: {scenario}") # Debug print
            fig, ax = plt.subplots(figsize=(20, 10))
            
            amro_results = self.results[scenario].get('AMRO')
            if not amro_results:
                print(f"DEBUG: No AMRO results for scenario {scenario}") # Debug print
                continue

            # Use the first run for the timeline plot
            task_details = amro_results[0].get('task_details', [])
            print(f"DEBUG: Task details length for {scenario}: {len(task_details)}") # Debug print
            if not task_details:
                print(f"DEBUG: Task details empty for scenario {scenario}") # Debug print
                continue

            df = pd.DataFrame(task_details)
            df['duration'] = df['finish_time'] - df['start_time']

            # Sort by start time for better visualization
            df = df.sort_values(by='start_time').reset_index(drop=True)

            # Create a unique y-position for each task
            task_y_pos = {task_id: i for i, task_id in enumerate(df['task_id'].unique())}

            for i, task in df.iterrows():
                color = 'blue' if task['processor'] == 'CPU' else 'orange'
                ax.barh(task_y_pos[task['task_id']], task['duration'], left=task['start_time'], 
                        height=0.8, color=color, edgecolor='black')
                ax.text(task['start_time'] + task['duration']/2, task_y_pos[task['task_id']], 
                        task['task_type'], ha='center', va='center', color='white', weight='bold')

            ax.set_yticks(list(task_y_pos.values()))
            ax.set_yticklabels(list(task_y_pos.keys()))
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Task ID')
            ax.set_title(f'Task Timeline for {scenario} Scenario')
            ax.legend(['CPU', 'NPU'])
            
            if save_plots:
                plt.savefig(f'{scenario}_timeline.png', dpi=300, bbox_inches='tight')
                print(f"Timeline plot saved as '{scenario}_timeline.png'")
            
            plt.show()

    def generate_report(self):
        """Generate a comprehensive performance report"""
        if not self.results:
            return "No experimental results available. Run experiments first."
        
        stats = self.calculate_statistics()
        
        report = []
        report.append("="*80)
        report.append("AMRO SCHEDULER PERFORMANCE EVALUATION REPORT")
        report.append("="*80)
        report.append("")

        # Summary statistics
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 50)
        
        # Find best performing algorithm for each metric
        best_algorithms = {}
        for scenario in stats:
            for algorithm in stats[scenario]:
                for metric in constants.PERFORMANCE_METRICS:
                    if metric in stats[scenario][algorithm]:
                        key = f"{scenario}_{metric}"
                        value = stats[scenario][algorithm][metric]['mean']
                        
                        if key not in best_algorithms:
                            best_algorithms[key] = (algorithm, value)
                        else:
                            current_best = best_algorithms[key]
                            # Lower is better for response time, waiting time, miss rate
                            if metric in ['average_response_time', 'average_waiting_time', 'deadline_miss_rate']:
                                if value < current_best[1]:
                                    best_algorithms[key] = (algorithm, value)
                            else:  # Higher is better for utilization and throughput
                                if value > current_best[1]:
                                    best_algorithms[key] = (algorithm, value)
        
        # Count wins for each algorithm
        algorithm_wins = defaultdict(int)
        for key, (algorithm, _) in best_algorithms.items():
            algorithm_wins[algorithm] += 1
        
        report.append("Best performing algorithms by metric:")
        for algo, wins in sorted(algorithm_wins.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {algo}: {wins} wins")
        
        report.append("")
        report.append("DETAILED RESULTS BY SCENARIO")
        report.append("-" * 50)
        
        for scenario in sorted(stats.keys()):
            report.append(f"\n{scenario.upper()} SCENARIO:")
            report.append("  " + "-" * 40)
            
            for metric in constants.PERFORMANCE_METRICS:
                report.append(f"\n  {metric.replace('_', ' ').title()}:")
                
                # Sort algorithms by performance for this metric
                algo_performance = []
                for algorithm in constants.SCHEDULING_ALGORITHMS:
                    if algorithm in stats[scenario] and metric in stats[scenario][algorithm]:
                        mean_val = stats[scenario][algorithm][metric]['mean']
                        std_val = stats[scenario][algorithm][metric]['std']
                        algo_performance.append((algorithm, mean_val, std_val))
                
                # Sort by performance (lower is better for some metrics)
                if metric in ['average_response_time', 'average_waiting_time', 'deadline_miss_rate']:
                    algo_performance.sort(key=lambda x: x[1])
                else:
                    algo_performance.sort(key=lambda x: x[1], reverse=True)
                
                for i, (algo, mean_val, std_val) in enumerate(algo_performance):
                    rank = i + 1
                    if metric in ['deadline_miss_rate']:
                        report.append(f"    {rank}. {algo}: {mean_val:.1%} ± {std_val:.1%}")
                    elif metric in ['cpu_utilization', 'npu_utilization']:
                        report.append(f"    {rank}. {algo}: {mean_val:.1%} ± {std_val:.1%}")
                    else:
                        report.append(f"    {rank}. {algo}: {mean_val:.2f} ± {std_val:.2f}")
        
        return "\n".join(report)
    
    def run_full_evaluation(self):
        """Run complete evaluation including experiments, plots, and report"""
        print("Starting full AMRO scheduler evaluation...")
        
        # Run experiments
        self.run_comprehensive_experiments()
        
        # Generate plots
        self.create_performance_plots()
        self.create_detailed_analysis_plots()
        self.create_timeline_plots()
        
        # Generate and save report
        report = self.generate_report()
        
        with open('scheduler_evaluation_report.txt', 'w') as f:
            f.write(report)
        
        print("\nEvaluation complete!")
        print("Files generated:")
        print("- scheduler_performance_comparison.png")
        print("- amro_detailed_analysis.png")
        print("- scheduler_evaluation_report.txt")
        
        return report


if __name__ == "__main__":
    # Run the complete evaluation
    runner = ExperimentRunner()
    report = runner.run_full_evaluation()
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    print(report)