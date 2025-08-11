#!/usr/bin/env python3
"""
Main runner script for AMRO scheduler evaluation
"""

import sys
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiment_runner import ExperimentRunner
import constants

def main():
    parser = argparse.ArgumentParser(description='AMRO Scheduler Evaluation')
    parser.add_argument('--scenario', type=str, 
                       choices=list(constants.EXPERIMENT_SCENARIOS.keys()),
                       help='Run specific scenario only')
    parser.add_argument('--algorithm', type=str,
                       choices=constants.SCHEDULING_ALGORITHMS,
                       help='Run specific algorithm only')
    parser.add_argument('--duration', type=int, default=constants.SIMULATION_TIME,
                       help='Simulation duration in milliseconds')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with reduced scenarios')
    
    args = parser.parse_args()
    
    print("="*60)
    print("AMRO SCHEDULER EVALUATION")
    print("="*60)
    print(f"Random seed: {args.seed}")
    print(f"Simulation duration: {args.duration} ms")
    print()
    
    # Create experiment runner
    runner = ExperimentRunner(random_seed=args.seed)
    
    if args.scenario and args.algorithm:
        # Run single experiment
        print(f"Running single experiment: {args.algorithm} on {args.scenario}")
        result = runner.run_single_experiment(args.algorithm, args.scenario, args.duration)
        
        print("\nResults:")
        print(f"  Completed tasks: {result['completed_tasks']}")
        print(f"  Deadline miss rate: {result['deadline_miss_rate']:.1%}")
        print(f"  Average response time: {result['average_response_time']:.2f} ms")
        print(f"  CPU utilization: {result['cpu_utilization']:.1%}")
        print(f"  NPU utilization: {result['npu_utilization']:.1%}")
        print(f"  Throughput: {result['throughput']:.2f} tasks/sec")
        
    elif args.quick:
        # Run quick evaluation with subset of scenarios
        print("Running quick evaluation...")
        quick_scenarios = ['light_load', 'medium_load']
        quick_algorithms = ['AMRO', 'FIFO', 'Earliest Deadline First']
        
        # Temporarily modify constants for quick run
        original_scenarios = constants.EXPERIMENT_SCENARIOS.copy()
        original_algorithms = constants.SCHEDULING_ALGORITHMS.copy()
        
        constants.EXPERIMENT_SCENARIOS = {k: v for k, v in original_scenarios.items() if k in quick_scenarios}
        constants.SCHEDULING_ALGORITHMS = quick_algorithms
        
        try:
            runner.run_comprehensive_experiments()
            
            if not args.no_plots:
                runner.create_performance_plots()
                runner.create_detailed_analysis_plots()
            
            report = runner.generate_report()
            print("\n" + report)
            
        finally:
            # Restore original constants
            constants.EXPERIMENT_SCENARIOS = original_scenarios
            constants.SCHEDULING_ALGORITHMS = original_algorithms
    
    else:
        # Run full evaluation
        print("Running full comprehensive evaluation...")
        runner.run_comprehensive_experiments()
        
        if not args.no_plots:
            runner.create_performance_plots()
            runner.create_detailed_analysis_plots()
        
        report = runner.generate_report()
        
        # Save report
        with open('scheduler_evaluation_report.txt', 'w') as f:
            f.write(report)
        
        print("\n" + report)
        print(f"\nReport saved to: scheduler_evaluation_report.txt")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()