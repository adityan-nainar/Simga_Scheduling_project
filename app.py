import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import json
import io
from base64 import b64encode
import os
from typing import Dict, List, Tuple

from jssp_sim.core.instance import Instance, Operation
from jssp_sim.core.params import GAParams
from jssp_sim.algorithms.fifo import run_fifo
from jssp_sim.algorithms.spt import run_spt
from jssp_sim.algorithms.ga import run_ga


def main():
    st.set_page_config(
        page_title="JSSP-Sim: Job Shop Scheduling Simulator",
        page_icon="📊",
        layout="wide",
    )
    
    st.title("JSSP-Sim: Job Shop Scheduling Simulator")
    st.write("Compare FIFO, SPT, and GA scheduling algorithms for the Job Shop Scheduling Problem.")
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("Problem Parameters")
        
        jobs = st.number_input("Number of Jobs", min_value=2, max_value=100, value=20)
        machines = st.number_input("Number of Machines", min_value=2, max_value=50, value=5)
        min_time = st.number_input("Min Processing Time", min_value=1, value=1)
        max_time = st.number_input("Max Processing Time", min_value=2, value=10)
        
        st.header("GA Parameters")
        
        pop_size = st.number_input("Population Size", min_value=10, value=50)
        generations = st.number_input("Generations", min_value=10, value=100)
        crossover_rate = st.slider("Crossover Rate", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
        mutation_rate = st.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        
        seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42)
        
        run_button = st.button("Run Simulation", type="primary")
    
    # Main content area
    if run_button:
        with st.spinner("Running simulation..."):
            # Create the problem instance
            instance = Instance(jobs=jobs, machines=machines, min_time=min_time, max_time=max_time, seed=seed)
            
            # Run FIFO algorithm
            metrics_fifo, schedule_fifo = run_fifo(instance)
            
            # Run SPT algorithm
            metrics_spt, schedule_spt = run_spt(instance)
            
            # Run GA algorithm
            ga_params = GAParams(
                population_size=pop_size,
                generations=generations,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate
            )
            metrics_ga, schedule_ga = run_ga(instance, ga_params)
            
            # Convert all NumPy values to Python primitives
            def convert_schedule(schedule):
                return [
                    {
                        "job_id": int(op.job_id),
                        "machine_id": int(op.machine_id),
                        "processing_time": int(op.processing_time),
                        "start_time": int(op.start_time),
                        "end_time": int(op.end_time)
                    }
                    for op in schedule
                ]
                
            def convert_metrics(metrics):
                result = {}
                for k, v in metrics.items():
                    if isinstance(v, dict):
                        result[k] = convert_metrics(v)
                    elif isinstance(v, list):
                        result[k] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in v]
                    elif isinstance(v, (np.integer, np.floating)):
                        result[k] = float(v) if isinstance(v, np.floating) else int(v)
                    else:
                        result[k] = v
                return result
            
            # Prepare results dictionary with only Python primitive types
            results = {
                "instance": {
                    "jobs": int(instance.num_jobs),
                    "machines": int(instance.num_machines),
                    "min_time": int(instance.min_time),
                    "max_time": int(instance.max_time)
                },
                "fifo": {
                    "metrics": convert_metrics(metrics_fifo),
                    "schedule": convert_schedule(schedule_fifo)
                },
                "spt": {
                    "metrics": convert_metrics(metrics_spt),
                    "schedule": convert_schedule(schedule_spt)
                },
                "ga": {
                    "metrics": convert_metrics(metrics_ga),
                    "schedule": convert_schedule(schedule_ga),
                    "params": {
                        "population_size": int(ga_params.population_size),
                        "generations": int(ga_params.generations),
                        "crossover_rate": float(ga_params.crossover_rate),
                        "mutation_rate": float(ga_params.mutation_rate),
                        "tournament_size": int(ga_params.tournament_size),
                        "elitism_rate": float(ga_params.elitism_rate)
                    }
                }
            }
            
            # Store results in session state
            st.session_state.results = results
        
        st.success("Simulation completed!")
        
        # Display results
        display_results(st.session_state.results)
    elif "results" in st.session_state:
        # If we've already run a simulation, display the results
        display_results(st.session_state.results)
    else:
        # First time loading the app
        st.info("Set parameters and click 'Run Simulation' to start.")


def display_results(results: Dict):
    """Display the results of the simulation."""
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Metrics Comparison", "Gantt Charts", "Raw Data"])
    
    with tab1:
        display_metrics_comparison(results)
    
    with tab2:
        display_gantt_charts(results)
    
    with tab3:
        display_raw_data(results)


def display_metrics_comparison(results: Dict):
    """Display a comparison of metrics between algorithms."""
    st.header("Metrics Comparison")
    
    # Extract metrics
    fifo_metrics = results["fifo"]["metrics"]
    spt_metrics = results["spt"]["metrics"]
    ga_metrics = results["ga"]["metrics"]
    
    # Create a comparison table - use numeric values directly, not formatted strings
    metrics_data = {
        "Metric": ["Makespan", "Total Flow Time", "Avg Flow Time", "Avg Machine Utilization", "CPU Time (s)"],
        "FIFO": [
            fifo_metrics["makespan"],
            fifo_metrics["total_flow_time"],
            fifo_metrics["avg_flow_time"],
            fifo_metrics["avg_machine_utilization"],
            fifo_metrics["cpu_time"]
        ],
        "SPT": [
            spt_metrics["makespan"],
            spt_metrics["total_flow_time"],
            spt_metrics["avg_flow_time"],
            spt_metrics["avg_machine_utilization"],
            spt_metrics["cpu_time"]
        ],
        "GA": [
            ga_metrics["makespan"],
            ga_metrics["total_flow_time"],
            ga_metrics["avg_flow_time"],
            ga_metrics["avg_machine_utilization"],
            ga_metrics["cpu_time"]
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Format the display dataframe separately for display purposes
    display_df = pd.DataFrame({
        "Metric": metrics_data["Metric"],
        "FIFO": [
            metrics_data["FIFO"][0],
            metrics_data["FIFO"][1],
            metrics_data["FIFO"][2],
            f"{metrics_data['FIFO'][3]:.2f}",
            f"{metrics_data['FIFO'][4]:.4f}"
        ],
        "SPT": [
            metrics_data["SPT"][0],
            metrics_data["SPT"][1],
            metrics_data["SPT"][2],
            f"{metrics_data['SPT'][3]:.2f}",
            f"{metrics_data['SPT'][4]:.4f}"
        ],
        "GA": [
            metrics_data["GA"][0],
            metrics_data["GA"][1],
            metrics_data["GA"][2],
            f"{metrics_data['GA'][3]:.2f}",
            f"{metrics_data['GA'][4]:.4f}"
        ]
    })
    
    st.dataframe(display_df, use_container_width=True)
    
    # Create some bar charts for visualization
    col1, col2 = st.columns(2)
    
    # Makespan comparison
    with col1:
        st.subheader("Makespan Comparison")
        makespan_data = {
            "Algorithm": ["FIFO", "SPT", "GA"],
            "Makespan": [fifo_metrics["makespan"], spt_metrics["makespan"], ga_metrics["makespan"]]
        }
        makespan_df = pd.DataFrame(makespan_data)
        st.bar_chart(makespan_df.set_index("Algorithm"), use_container_width=True)
    
    # Machine utilization comparison
    with col2:
        st.subheader("Machine Utilization Comparison")
        util_data = {
            "Algorithm": ["FIFO", "SPT", "GA"],
            "Utilization": [
                fifo_metrics["avg_machine_utilization"],
                spt_metrics["avg_machine_utilization"],
                ga_metrics["avg_machine_utilization"]
            ]
        }
        util_df = pd.DataFrame(util_data)
        st.bar_chart(util_df.set_index("Algorithm"), use_container_width=True)
    
    # If GA was run, show the convergence plot
    if "best_makespan_history" in ga_metrics:
        st.subheader("GA Convergence")
        history = ga_metrics["best_makespan_history"]
        history_data = {
            "Generation": list(range(len(history))),
            "Best Makespan": history
        }
        history_df = pd.DataFrame(history_data)
        st.line_chart(history_df.set_index("Generation"), use_container_width=True)


def display_gantt_charts(results: Dict):
    """Display Gantt charts for all algorithms."""
    st.header("Gantt Charts")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("FIFO Schedule")
        fifo_gantt = create_gantt_chart(results["fifo"]["schedule"], "FIFO", 
                                         results["fifo"]["metrics"]["makespan"],
                                         results["instance"]["machines"])
        st.pyplot(fifo_gantt)
        st.download_button(
            label="Download FIFO Gantt",
            data=get_image_download_link(fifo_gantt),
            file_name="fifo_gantt.png",
            mime="image/png"
        )
    
    with col2:
        st.subheader("SPT Schedule")
        spt_gantt = create_gantt_chart(results["spt"]["schedule"], "SPT", 
                                        results["spt"]["metrics"]["makespan"],
                                        results["instance"]["machines"])
        st.pyplot(spt_gantt)
        st.download_button(
            label="Download SPT Gantt",
            data=get_image_download_link(spt_gantt),
            file_name="spt_gantt.png",
            mime="image/png"
        )
    
    with col3:
        st.subheader("GA Schedule")
        ga_gantt = create_gantt_chart(results["ga"]["schedule"], "GA", 
                                       results["ga"]["metrics"]["makespan"],
                                       results["instance"]["machines"])
        st.pyplot(ga_gantt)
        st.download_button(
            label="Download GA Gantt",
            data=get_image_download_link(ga_gantt),
            file_name="ga_gantt.png",
            mime="image/png"
        )


def create_gantt_chart(schedule: List[Dict], algorithm: str, makespan: int, num_machines: int):
    """Create a Gantt chart for a schedule."""
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Extract job and machine count
    num_jobs = len(set(op["job_id"] for op in schedule))
    
    # Define colors for jobs (using a colormap)
    # Fix the deprecation warning by using matplotlib.colormaps
    colors = plt.colormaps.get_cmap('tab20', num_jobs)
    
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
    
    plt.tight_layout()
    
    return fig


def get_image_download_link(fig):
    """Generate a download link for a matplotlib figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    return buf.getvalue()


def display_raw_data(results: Dict):
    """Display the raw data for all algorithms."""
    st.header("Raw Data")
    
    # Create tabs for each algorithm
    fifo_tab, spt_tab, ga_tab, download_tab = st.tabs(["FIFO", "SPT", "GA", "Download All"])
    
    with fifo_tab:
        st.subheader("FIFO Schedule")
        fifo_df = pd.DataFrame(results["fifo"]["schedule"])
        st.dataframe(fifo_df, use_container_width=True)
    
    with spt_tab:
        st.subheader("SPT Schedule")
        spt_df = pd.DataFrame(results["spt"]["schedule"])
        st.dataframe(spt_df, use_container_width=True)
    
    with ga_tab:
        st.subheader("GA Schedule")
        ga_df = pd.DataFrame(results["ga"]["schedule"])
        st.dataframe(ga_df, use_container_width=True)
    
    with download_tab:
        st.subheader("Download Results")
        
        try:
            # Create JSON string
            json_data = json.dumps(results, indent=2)
            
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name="jssp_results.json",
                mime="application/json"
            )
            
            # Create CSV downloads
            fifo_csv = fifo_df.to_csv(index=False)
            st.download_button(
                label="Download FIFO CSV",
                data=fifo_csv,
                file_name="fifo_schedule.csv",
                mime="text/csv"
            )
            
            spt_csv = spt_df.to_csv(index=False)
            st.download_button(
                label="Download SPT CSV",
                data=spt_csv,
                file_name="spt_schedule.csv",
                mime="text/csv"
            )
            
            ga_csv = ga_df.to_csv(index=False)
            st.download_button(
                label="Download GA CSV",
                data=ga_csv,
                file_name="ga_schedule.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error generating downloads: {e}")


if __name__ == "__main__":
    main() 