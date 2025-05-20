import json
import click
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict, List
import os

from jssp_sim.core.instance import Instance
from jssp_sim.core.params import GAParams
from jssp_sim.algorithms.fifo import run_fifo
from jssp_sim.algorithms.spt import run_spt
from jssp_sim.algorithms.ga import run_ga


@click.group()
def cli():
    """Job Shop Scheduling Problem Simulator"""
    pass


@cli.command()
@click.option('--algo', type=click.Choice(['fifo', 'spt', 'ga']), required=True, help='Algorithm to use')
@click.option('--jobs', type=int, default=20, help='Number of jobs')
@click.option('--machines', type=int, default=5, help='Number of machines')
@click.option('--min-time', type=int, default=1, help='Minimum processing time')
@click.option('--max-time', type=int, default=10, help='Maximum processing time')
@click.option('--pop-size', type=int, default=50, help='GA population size')
@click.option('--generations', type=int, default=100, help='GA generations')
@click.option('--crossover', type=float, default=0.8, help='GA crossover rate')
@click.option('--mutation', type=float, default=0.1, help='GA mutation rate')
@click.option('--seed', type=int, default=None, help='Random seed for reproducibility')
@click.option('--output', type=str, default=None, help='Output file for results (JSON)')
def run(algo, jobs, machines, min_time, max_time, pop_size, generations, crossover, mutation, seed, output):
    """Run a job shop scheduling algorithm"""
    # Create problem instance
    instance = Instance(jobs=jobs, machines=machines, min_time=min_time, max_time=max_time, seed=seed)
    
    # Run the selected algorithm
    if algo == 'fifo':
        metrics, schedule = run_fifo(instance)
    elif algo == 'spt':
        metrics, schedule = run_spt(instance)
    elif algo == 'ga':
        ga_params = GAParams(
            population_size=pop_size,
            generations=generations,
            crossover_rate=crossover,
            mutation_rate=mutation
        )
        metrics, schedule = run_ga(instance, ga_params)
    
    # Prepare results for output
    results = {
        "instance": instance.to_dict(),
        "algorithm": algo,
        "metrics": metrics,
        "schedule": [
            {
                "job_id": op.job_id,
                "machine_id": op.machine_id,
                "processing_time": op.processing_time,
                "start_time": op.start_time,
                "end_time": op.end_time
            }
            for op in schedule
        ]
    }
    
    if algo == 'ga':
        results["ga_params"] = {
            "population_size": pop_size,
            "generations": generations,
            "crossover_rate": crossover,
            "mutation_rate": mutation
        }
    
    # Output results
    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        click.echo(f"Results saved to {output}")
    else:
        # Print a summary to the console
        click.echo(f"Algorithm: {algo.upper()}")
        click.echo(f"Makespan: {metrics['makespan']}")
        click.echo(f"Total flow time: {metrics['total_flow_time']}")
        click.echo(f"Average machine utilization: {metrics['avg_machine_utilization']:.2f}")
        click.echo(f"CPU time: {metrics['cpu_time']:.4f} seconds")


@cli.command()
@click.option('--input', type=str, required=True, help='Input JSON file with schedule')
@click.option('--format', type=click.Choice(['json', 'csv', 'png']), default='png', help='Export format')
@click.option('--output', type=str, default=None, help='Output file name')
def export(input, format, output):
    """Export a schedule to various formats"""
    # Load the input file
    with open(input, 'r') as f:
        data = json.load(f)
    
    # Extract the schedule and other data
    schedule = data["schedule"]
    algorithm = data["algorithm"]
    
    # Default output filename if not provided
    if not output:
        base_name = os.path.splitext(os.path.basename(input))[0]
        output = f"{base_name}_{algorithm}.{format}"
    
    if format == 'json':
        # Just save as is
        with open(output, 'w') as f:
            json.dump(data, f, indent=2)
        click.echo(f"JSON exported to {output}")
        
    elif format == 'csv':
        # Export as CSV
        with open(output, 'w') as f:
            f.write("job_id,machine_id,processing_time,start_time,end_time\n")
            for op in schedule:
                f.write(f"{op['job_id']},{op['machine_id']},{op['processing_time']},{op['start_time']},{op['end_time']}\n")
        click.echo(f"CSV exported to {output}")
        
    elif format == 'png':
        # Export as Gantt chart
        _export_gantt(data, output)
        click.echo(f"Gantt chart exported to {output}")


def _export_gantt(data: Dict, output: str) -> None:
    """
    Export a schedule as a Gantt chart.
    
    Args:
        data: Dictionary with schedule and other data
        output: Output file name
    """
    schedule = data["schedule"]
    algorithm = data["algorithm"].upper()
    makespan = data["metrics"]["makespan"]
    
    # Extract job and machine count
    num_jobs = len(set(op["job_id"] for op in schedule))
    num_machines = len(set(op["machine_id"] for op in schedule))
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define colors for jobs (using a colormap)
    colors = plt.cm.get_cmap('tab20', num_jobs)
    
    # Plot each operation as a rectangle
    for op in schedule:
        job_id = op["job_id"]
        machine_id = op["machine_id"]
        start = op["start_time"]
        duration = op["processing_time"]
        
        # Create a rectangle for the operation
        rect = patches.Rectangle(
            (start, machine_id - 0.4),
            duration,
            0.8,
            linewidth=1,
            edgecolor='black',
            facecolor=colors(job_id),
            alpha=0.7
        )
        
        # Add the rectangle to the plot
        ax.add_patch(rect)
        
        # Add job ID label in the middle of the rectangle
        ax.text(
            start + duration / 2,
            machine_id,
            f"J{job_id}",
            ha='center',
            va='center',
            fontsize=8,
            color='black'
        )
    
    # Set the axis limits and labels
    ax.set_xlim(0, makespan * 1.05)
    ax.set_ylim(-0.5, num_machines - 0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f'M{i}' for i in range(num_machines)])
    
    # Set the title
    ax.set_title(f'{algorithm} Schedule - Makespan: {makespan}')
    
    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


if __name__ == '__main__':
    cli() 