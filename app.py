import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import json
import io
from typing import Dict, List, Any

from jssp_sim.core.instance import Instance
from jssp_sim.core.params import GAParams
from jssp_sim.algorithms.fifo import run_fifo
from jssp_sim.algorithms.spt import run_spt
from jssp_sim.algorithms.ga import run_ga

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy and other special types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return None if np.isnan(obj) else float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super().default(obj)

def convert_numpy_to_python(data: Any) -> Any:
    """Recursively converts NumPy types and NaN to native Python types."""
    if isinstance(data, (int, float, str, bool, type(None))):
        return data 
    if isinstance(data, list) or isinstance(data, tuple):
        return [convert_numpy_to_python(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_numpy_to_python(value) for key, value in data.items()}
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return None if np.isnan(data) else float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.bool_):
        return bool(data)
    elif pd.isna(data): # Should catch unhandled NaNs, NaTs
        return None
    return data

def main():
    st.set_page_config(
        page_title="JSSP-Sim: Job Shop Scheduling Simulator",
        page_icon="📊",
        layout="wide",
    )
    
    st.title("JSSP-Sim: Job Shop Scheduling Simulator")
    st.write("Compare FIFO, SPT, and GA scheduling algorithms for the Job Shop Scheduling Problem.")
    
    with st.sidebar:
        st.header("Problem Parameters")
        jobs = int(st.number_input("Jobs", 2, 100, 20))
        machines = int(st.number_input("Machines", 2, 50, 5))
        min_time = int(st.number_input("Min Time", 1, value=1))
        max_time = int(st.number_input("Max Time", 2, value=10))
        
        st.header("GA Parameters")
        pop_size = int(st.number_input("Population", 10, value=50))
        generations = int(st.number_input("Generations", 10, value=100))
        crossover_rate = float(st.slider("Crossover Rate", 0.0, 1.0, 0.8, 0.05))
        mutation_rate = float(st.slider("Mutation Rate", 0.0, 1.0, 0.1, 0.05))
        seed = int(st.number_input("Seed", 0, 9999, 42))
        run_button = st.button("Run Simulation", type="primary")
    
    if run_button:
        with st.spinner("Running simulation..."):
            instance = Instance(jobs, machines, min_time, max_time, seed=seed)
            
            metrics_fifo_raw, schedule_fifo_raw = run_fifo(instance)
            metrics_spt_raw, schedule_spt_raw = run_spt(instance)
            
            ga_params_obj = GAParams(
                population_size=pop_size, generations=generations,
                crossover_rate=crossover_rate, mutation_rate=mutation_rate
            )
            metrics_ga_raw, schedule_ga_raw = run_ga(instance, ga_params_obj)

            # Convert all raw metrics and params immediately
            metrics_fifo = convert_numpy_to_python(metrics_fifo_raw)
            metrics_spt = convert_numpy_to_python(metrics_spt_raw)
            metrics_ga = convert_numpy_to_python(metrics_ga_raw)
            ga_params_dict = convert_numpy_to_python(ga_params_obj.to_dict())

            def convert_schedule_list(schedule_data: List[Any]) -> List[Dict[str, int]]:
                return [
                    {
                        "job_id": int(op.job_id), "machine_id": int(op.machine_id),
                        "processing_time": int(op.processing_time),
                        "start_time": int(op.start_time), "end_time": int(op.end_time)
                    }
                    for op in schedule_data
                ]

            schedule_fifo = convert_schedule_list(schedule_fifo_raw)
            schedule_spt = convert_schedule_list(schedule_spt_raw)
            schedule_ga = convert_schedule_list(schedule_ga_raw)

            results = {
                "instance": {
                    "jobs": jobs, "machines": machines,
                    "min_time": min_time, "max_time": max_time
                },
                "fifo": {"metrics": metrics_fifo, "schedule": schedule_fifo},
                "spt": {"metrics": metrics_spt, "schedule": schedule_spt},
                "ga": {"metrics": metrics_ga, "schedule": schedule_ga, "params": ga_params_dict}
            }
            
            # Final global conversion before storing in session state
            st.session_state.results = convert_numpy_to_python(results)
        
        st.success("Simulation completed!")
        display_results(st.session_state.results)

    elif "results" in st.session_state:
        # Ensure results from session state are also clean upon retrieval
        cleaned_results = convert_numpy_to_python(st.session_state.results)
        display_results(cleaned_results)
    else:
        st.info("Set parameters and click 'Run Simulation' to start.")

def display_results(results: Dict):
    tab1, tab2, tab3 = st.tabs(["Metrics Comparison", "Gantt Charts", "Raw Data"])
    with tab1: display_metrics_comparison(results)
    with tab2: display_gantt_charts(results)
    with tab3: display_raw_data(results)

def display_metrics_comparison(results: Dict):
    st.header("Metrics Comparison")
    fifo_metrics = results["fifo"]["metrics"]
    spt_metrics = results["spt"]["metrics"]
    ga_metrics = results["ga"]["metrics"]

    metrics_order = ["makespan", "total_flow_time", "avg_flow_time", "avg_machine_utilization", "cpu_time"]
    metric_display_names = {
        "makespan": "Makespan", "total_flow_time": "Total Flow Time", "avg_flow_time": "Avg Flow Time (2dp)",
        "avg_machine_utilization": "Avg Machine Util. (2dp)", "cpu_time": "CPU Time (s, 4dp)"
    }
    format_strings = {
        "avg_flow_time": "{:.2f}", "avg_machine_utilization": "{:.2f}", "cpu_time": "{:.4f}"
    }

    # Prepare data for display_df (all string, formatted)
    display_data_for_table = {"Metric": [metric_display_names[m] for m in metrics_order]}
    numeric_data_for_charts = {"Algorithm": ["FIFO", "SPT", "GA"]}
    
    all_metrics = {"FIFO": fifo_metrics, "SPT": spt_metrics, "GA": ga_metrics}

    for algo_name, metrics_dict in all_metrics.items():
        display_data_for_table[algo_name] = []
        for metric_key in metrics_order:
            val = metrics_dict.get(metric_key)
            # For display table, always format, ensure it's a string
            if val is None: display_data_for_table[algo_name].append("N/A")
            elif metric_key in format_strings: display_data_for_table[algo_name].append(format_strings[metric_key].format(val))
            else: display_data_for_table[algo_name].append(str(val))
    
    display_df = pd.DataFrame(display_data_for_table).set_index("Metric")
    st.dataframe(display_df, use_container_width=True)

    # Prepare numeric data for charts
    numeric_data_for_charts["Makespan"] = [all_metrics[algo]["makespan"] for algo in ["FIFO", "SPT", "GA"]]
    numeric_data_for_charts["Utilization"] = [all_metrics[algo]["avg_machine_utilization"] for algo in ["FIFO", "SPT", "GA"]]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Makespan Comparison")
        # Filter out None values for makespan chart
        valid_makespan_data = {
            "Algorithm": [a for a, m in zip(numeric_data_for_charts["Algorithm"], numeric_data_for_charts["Makespan"]) if m is not None],
            "Makespan": [m for m in numeric_data_for_charts["Makespan"] if m is not None]
        }
        if valid_makespan_data["Makespan"]:
            makespan_df = pd.DataFrame(valid_makespan_data).set_index("Algorithm")
            st.bar_chart(makespan_df)
        else: st.info("No makespan data to plot.")
    
    with col2:
        st.subheader("Machine Utilization Comparison")
        # Filter out None values for utilization chart
        valid_util_data = {
            "Algorithm": [a for a, u in zip(numeric_data_for_charts["Algorithm"], numeric_data_for_charts["Utilization"]) if u is not None],
            "Utilization": [u for u in numeric_data_for_charts["Utilization"] if u is not None]
        }
        if valid_util_data["Utilization"]:
            util_df = pd.DataFrame(valid_util_data).set_index("Algorithm")
            st.bar_chart(util_df)
        else: st.info("No utilization data to plot.")

    ga_metrics_clean = ga_metrics
    if "best_makespan_history" in ga_metrics_clean:
        st.subheader("GA Convergence")
        history = ga_metrics_clean["best_makespan_history"]
        if history and isinstance(history, list) and all(isinstance(x, (int, float)) for x in history):
            history_df = pd.DataFrame({"Generation": list(range(len(history))), "Best Makespan": history})
            st.line_chart(history_df.set_index("Generation"))
        else:
            st.warning("GA history data not plottable or unavailable.")

def display_gantt_charts(results: Dict):
    st.header("Gantt Charts")
    instance_machines = results["instance"]["machines"]
    col1, col2, col3 = st.columns(3)
    
    algorithms = ["fifo", "spt", "ga"]
    titles = ["FIFO", "SPT", "GA"]
    
    for i, (col, algo_key, title) in enumerate(zip([col1, col2, col3], algorithms, titles)):
        with col:
            st.subheader(f"{title} Schedule")
            schedule_data = results[algo_key]["schedule"]
            makespan = results[algo_key]["metrics"]["makespan"]
            gantt_fig = create_gantt_chart(schedule_data, title, makespan, instance_machines)
            st.pyplot(gantt_fig)
            st.download_button(
                label=f"Download {title} Gantt",
                data=get_image_download_link(gantt_fig),
                file_name=f"{algo_key}_gantt.png",
                mime="image/png"
            )

def create_gantt_chart(schedule: List[Dict[str, Any]], algorithm: str, makespan: int, num_machines: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6)) 
    job_ids = sorted(list(set(op.get("job_id") for op in schedule if op.get("job_id") is not None)))
    num_jobs = len(job_ids)
    
    # Corrected: Use plt.colormaps['colormap_name'] and then sample from it.
    # 'tab20' provides 20 distinct colors. If num_jobs > 20, colors will repeat.
    try:
        cmap = plt.colormaps['tab20'] 
    except KeyError:
        # Fallback if 'tab20' is somehow not available, though it should be standard.
        cmap = plt.colormaps['viridis'] 
        
    job_colors = {job_id: cmap(i % cmap.N if cmap.N > 0 else 0) for i, job_id in enumerate(job_ids)}

    for op in schedule:
        job_id, machine_id = op.get("job_id"), op.get("machine_id")
        start, duration = op.get("start_time"), op.get("processing_time")
        if None not in [job_id, machine_id, start, duration]:
            rect_facecolor = job_colors.get(job_id, cmap(0)) # Default to first color of cmap if job_id somehow not in job_colors
            rect = patches.Rectangle(
                (start, machine_id - 0.4), duration, 0.8,
                linewidth=1, edgecolor='black', facecolor=rect_facecolor, alpha=0.7
            )
            ax.add_patch(rect)
            ax.text(start + duration / 2, machine_id, f"J{job_id}", ha='center', va='center', fontsize=8, color='black')
    
    ax.set_xlim(0, makespan * 1.05 if makespan > 0 else 10)
    ax.set_ylim(-0.5, num_machines - 0.5 if num_machines > 0 else 0.5)
    ax.set_xlabel("Time"); ax.set_ylabel("Machine")
    if num_machines > 0:
        ax.set_yticks(range(num_machines))
        ax.set_yticklabels([f"M{i}" for i in range(num_machines)])
    ax.set_title(f"{algorithm} Schedule - Makespan: {makespan}")
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(); return fig

def get_image_download_link(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close(fig) # Close the figure to free memory
    return buf.getvalue()

def display_raw_data(results: Dict):
    st.header("Raw Data")
    # results should be clean by now from session state
    
    tabs_data = {
        "FIFO": results["fifo"]["schedule"],
        "SPT": results["spt"]["schedule"],
        "GA": results["ga"]["schedule"]
    }
    
    tab_keys = list(tabs_data.keys())
    tabs = st.tabs(tab_keys + ["Download All"])

    for i, key in enumerate(tab_keys):
        with tabs[i]:
            st.subheader(f"{key} Schedule")
            df = pd.DataFrame(tabs_data[key])
            st.dataframe(df, use_container_width=True)

    with tabs[-1]: # Download All tab
        st.subheader("Download Results")
        try:
            # Use NumpyJSONEncoder as a safeguard, though results should be Python-native.
            json_data = json.dumps(results, indent=2, cls=NumpyJSONEncoder)
            st.download_button(
                label="Download JSON", data=json_data,
                file_name="jssp_results.json", mime="application/json"
            )
            
            for key in tab_keys:
                df_to_download = pd.DataFrame(tabs_data[key])
                csv_data = df_to_download.to_csv(index=False)
                st.download_button(
                    label=f"Download {key} CSV", data=csv_data,
                    file_name=f"{key.lower()}_schedule.csv", mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error generating downloads: {str(e)}")
            st.exception(e) # Provides more details for debugging

if __name__ == "__main__":
    main() 