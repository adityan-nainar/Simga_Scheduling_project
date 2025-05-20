# JSSP-Sim: FIFO | SPT | GA

A Job Shop Scheduling Problem (JSSP) Simulator with FIFO, SPT, and Genetic Algorithm implementations.

## Overview

JSSP-Sim is a pure-Python library for generating and evaluating job-shop schedules under different algorithms:

- **FIFO**: First-In-First-Out scheduling
- **SPT**: Shortest Processing Time
- **GA**: Genetic Algorithm with tunable parameters

## Installation

Install from source:

```bash
git clone https://github.com/username/jssp_sim.git
cd jssp_sim
pip install -e .
```

## Usage

### Command-Line Interface

```bash
# Run a simulation with FIFO algorithm
jssp_sim run --algo fifo --jobs 20 --machines 5 --output fifo_results.json

# Run a simulation with SPT algorithm  
jssp_sim run --algo spt --jobs 20 --machines 5 --output spt_results.json

# Run a simulation with GA algorithm
jssp_sim run --algo ga --jobs 20 --machines 5 \
    --pop-size 50 --generations 100 --crossover 0.8 --mutation 0.1 \
    --output ga_results.json

# Export a Gantt chart from results
jssp_sim export --input ga_results.json --format png --output ga_gantt.png
```

### Python API

```python
from jssp_sim import Instance, GAParams, run_fifo, run_spt, run_ga

# Create a problem instance
inst = Instance(jobs=20, machines=5, min_time=1, max_time=10)

# Run the FIFO algorithm
metrics_fifo, schedule_fifo = run_fifo(inst)

# Run the SPT algorithm
metrics_spt, schedule_spt = run_spt(inst)

# Run the GA algorithm
ga_params = GAParams(
    population_size=50,
    generations=100,
    crossover_rate=0.8,
    mutation_rate=0.1
)
metrics_ga, schedule_ga = run_ga(inst, ga_params)

# Print makespan for each algorithm
print(f"FIFO makespan: {metrics_fifo['makespan']}")
print(f"SPT makespan: {metrics_spt['makespan']}")
print(f"GA makespan: {metrics_ga['makespan']}")
```

## Streamlit App

To run the interactive Streamlit app:

```bash
streamlit run app.py
```

## Features

- Pure Python implementation with minimal dependencies
- Modular design separating algorithms from UI
- Comprehensive metrics (makespan, flow time, machine utilization)
- Visualization via Gantt charts
- Customizable GA parameters
- Export results in JSON, CSV, or PNG formats

## License

MIT License 