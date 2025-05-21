import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import json
import io
from typing import Dict, List, Tuple, Any

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
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj): # Handles np.nan, pd.NaT
            return None
        # Add handling for Path objects if they ever appear, though unlikely here
        # from pathlib import Path
        # if isinstance(obj, Path):
        #     return str(obj)
        return super().default(obj)

def convert_numpy_to_python(data: Any) -> Any:
    """Recursively converts NumPy types in a data structure to native Python types."""
    if isinstance(data, list) or isinstance(data, tuple):
        return [convert_numpy_to_python(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_numpy_to_python(value) for key, value in data.items()}
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.bool_): # NumPy bool
        return bool(data)
    elif pd.isna(data): # This should catch np.nan, pd.NaT
        return None
    # elif isinstance(data, np.datetime64): # Handle if necessary
    #     return str(data) # Or some other appropriate conversion
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
    
    if run_button:
        with st.spinner("Running simulation..."):
            instance = Instance(jobs=jobs, machines=machines, min_time=min_time, max_time=max_time, seed=seed)
            
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
                    "jobs": int(instance.num_jobs), "machines": int(instance.num_machines),
                    "min_time": int(instance.min_time), "max_time": int(instance.max_time)
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

    # Data for internal use and charts (raw numbers)
    metrics_data_numeric = {
        "Metric": ["Makespan", "Total Flow Time", "Avg Flow Time", "Avg Machine Utilization", "CPU Time (s)"],
        "FIFO": [
            fifo_metrics["makespan"], fifo_metrics["total_flow_time"], fifo_metrics["avg_flow_time"],
            fifo_metrics["avg_machine_utilization"], fifo_metrics["cpu_time"]
        ],
        "SPT": [
            spt_metrics["makespan"], spt_metrics["total_flow_time"], spt_metrics["avg_flow_time"],
            spt_metrics["avg_machine_utilization"], spt_metrics["cpu_time"]
        ],
        "GA": [
            ga_metrics["makespan"], ga_metrics["total_flow_time"], ga_metrics["avg_flow_time"],
            ga_metrics["avg_machine_utilization"], ga_metrics["cpu_time"]
        ]
    }
    # metrics_df_numeric = pd.DataFrame(metrics_data_numeric) # If needed for checks or direct chart use

    # Data for st.dataframe (formatted strings)
    display_df_data = {
        "Metric": metrics_data_numeric["Metric"],
        "FIFO": [
            str(metrics_data_numeric["FIFO"][0]), str(metrics_data_numeric["FIFO"][1]),
            f"{metrics_data_numeric['FIFO'][2]:.2f}", f"{metrics_data_numeric['FIFO'][3]:.2f}", f"{metrics_data_numeric['FIFO'][4]:.4f}"
        ],
        "SPT": [
            str(metrics_data_numeric["SPT"][0]), str(metrics_data_numeric["SPT"][1]),
            f"{metrics_data_numeric['SPT'][2]:.2f}", f"{metrics_data_numeric['SPT'][3]:.2f}", f"{metrics_data_numeric['SPT'][4]:.4f}"
        ],
        "GA": [
            str(metrics_data_numeric["GA"][0]), str(metrics_data_numeric["GA"][1]),
            f"{metrics_data_numeric['GA'][2]:.2f}", f"{metrics_data_numeric['GA'][3]:.2f}", f"{metrics_data_numeric['GA'][4]:.4f}"
        ]
    }
    display_df = pd.DataFrame(display_df_data)
    st.dataframe(display_df.set_index("Metric"), use_container_width=True) # Set index for better display

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Makespan Comparison")
        makespan_data = {
            "Algorithm": ["FIFO", "SPT", "GA"],
            "Makespan": [metrics_data_numeric["FIFO"][0], metrics_data_numeric["SPT"][0], metrics_data_numeric["GA"][0]]
        }
        st.bar_chart(pd.DataFrame(makespan_data).set_index("Algorithm"))
    
    with col2:
        st.subheader("Machine Utilization Comparison")
        util_data = {
            "Algorithm": ["FIFO", "SPT", "GA"],
            "Utilization": [metrics_data_numeric["FIFO"][3], metrics_data_numeric["SPT"][3], metrics_data_numeric["GA"][3]]
        }
        st.bar_chart(pd.DataFrame(util_data).set_index("Algorithm"))

    if "best_makespan_history" in ga_metrics: # ga_metrics is clean
        st.subheader("GA Convergence")
        history = ga_metrics["best_makespan_history"] # Should be a list of numbers
        history_df = pd.DataFrame({"Generation": list(range(len(history))), "Best Makespan": history})
        st.line_chart(history_df.set_index("Generation"))

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
    fig, ax = plt.subplots(figsize=(12, 6)) # Adjusted size
    
    job_ids = sorted(list(set(op["job_id"] for op in schedule)))
    num_jobs = len(job_ids)
    
    # Use matplotlib.colormaps for modern API
    cmap = plt.colormaps.get_cmap('tab20', num_jobs if num_jobs > 0 else 1)
    job_colors = {job_id: cmap(i % cmap.N) for i, job_id in enumerate(job_ids)}

    for op in schedule:
        job_id, machine_id, start, duration = op["job_id"], op["machine_id"], op["start_time"], op["processing_time"]
        rect = patches.Rectangle(
            (start, machine_id - 0.4), duration, 0.8,
            linewidth=1, edgecolor='black', facecolor=job_colors.get(job_id, 'gray'), alpha=0.7
        )
        ax.add_patch(rect)
        ax.text(start + duration / 2, machine_id, f"J{job_id}", ha='center', va='center', fontsize=8, color='black')
    
    ax.set_xlim(0, makespan * 1.05 if makespan > 0 else 10)
    ax.set_ylim(-0.5, num_machines - 0.5 if num_machines > 0 else 0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    if num_machines > 0:
        ax.set_yticks(range(num_machines))
        ax.set_yticklabels([f"M{i}" for i in range(num_machines)])
    ax.set_title(f"{algorithm} Schedule - Makespan: {makespan}")
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

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