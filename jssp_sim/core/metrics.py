from typing import List, Dict, Tuple
import time
from copy import deepcopy

from jssp_sim.core.instance import Operation, Instance


def calculate_metrics(instance: Instance, operations: List[Operation]) -> Dict:
    """
    Calculate metrics for a given schedule.
    
    Args:
        instance: The problem instance
        operations: List of scheduled operations (with start_time and end_time set)
    
    Returns:
        Dictionary of metrics
    """
    # Make a deep copy to avoid modifying the original operations
    operations = deepcopy(operations)
    
    # Check that all operations have start and end times
    for op in operations:
        if op.start_time is None or op.end_time is None:
            raise ValueError(f"Operation {op} is not scheduled")
    
    # Calculate makespan (maximum end time across all operations)
    makespan = max(op.end_time for op in operations)
    
    # Calculate total flow time (sum of completion times for each job)
    job_completion_times = {}
    for op in operations:
        if op.job_id not in job_completion_times or op.end_time > job_completion_times[op.job_id]:
            job_completion_times[op.job_id] = op.end_time
    
    total_flow_time = sum(job_completion_times.values())
    
    # Calculate machine utilization
    machine_busy_time = {}
    machine_idle_time = {}
    
    for m in range(instance.num_machines):
        machine_ops = [op for op in operations if op.machine_id == m]
        machine_ops.sort(key=lambda op: op.start_time)
        
        busy_time = sum(op.processing_time for op in machine_ops)
        machine_busy_time[m] = busy_time
        
        # Calculate idle time (gaps between operations)
        if not machine_ops:
            machine_idle_time[m] = 0
            continue
            
        idle_time = 0
        last_end = 0
        
        for op in machine_ops:
            if op.start_time > last_end:
                idle_time += op.start_time - last_end
            last_end = op.end_time
        
        machine_idle_time[m] = idle_time
    
    # Average machine utilization
    total_busy_time = sum(machine_busy_time.values())
    avg_machine_util = total_busy_time / (instance.num_machines * makespan) if makespan > 0 else 0
    
    return {
        "makespan": makespan,
        "total_flow_time": total_flow_time,
        "avg_flow_time": total_flow_time / instance.num_jobs,
        "total_processing_time": sum(op.processing_time for op in operations),
        "total_machine_busy_time": total_busy_time,
        "total_machine_idle_time": sum(machine_idle_time.values()),
        "avg_machine_utilization": avg_machine_util,
        "machine_busy_times": machine_busy_time,
        "machine_idle_times": machine_idle_time,
        "job_completion_times": job_completion_times
    }


def validate_schedule(instance: Instance, operations: List[Operation]) -> bool:
    """
    Validate that a schedule is feasible.
    
    Args:
        instance: The problem instance
        operations: List of scheduled operations
    
    Returns:
        True if the schedule is valid, raises ValueError otherwise
    """
    # Check that all operations have start and end times
    for op in operations:
        if op.start_time is None or op.end_time is None:
            raise ValueError(f"Operation {op} is not scheduled")
        
        # Check that end time is consistent with processing time
        if op.end_time != op.start_time + op.processing_time:
            raise ValueError(
                f"Operation {op} has inconsistent timing: "
                f"end_time ({op.end_time}) != start_time ({op.start_time}) + processing_time ({op.processing_time})"
            )
    
    # Check job precedence constraints
    for job in instance.jobs:
        job_ops = [op for op in operations if op.job_id == job.job_id]
        job_ops.sort(key=lambda op: [op.machine_id])
        
        for i in range(1, len(job_ops)):
            prev_op = job_ops[i-1]
            curr_op = job_ops[i]
            
            # Each operation in a job must start after the previous operation is completed
            if curr_op.start_time < prev_op.end_time:
                raise ValueError(
                    f"Job precedence constraint violated for job {job.job_id}: "
                    f"Operation on machine {curr_op.machine_id} starts at {curr_op.start_time} "
                    f"but previous operation on machine {prev_op.machine_id} ends at {prev_op.end_time}"
                )
    
    # Check machine capacity constraints
    for m in range(instance.num_machines):
        machine_ops = [op for op in operations if op.machine_id == m]
        
        # Sort operations by start time
        machine_ops.sort(key=lambda op: op.start_time)
        
        # Check for overlaps
        for i in range(1, len(machine_ops)):
            prev_op = machine_ops[i-1]
            curr_op = machine_ops[i]
            
            if curr_op.start_time < prev_op.end_time:
                raise ValueError(
                    f"Machine capacity constraint violated for machine {m}: "
                    f"Operation for job {curr_op.job_id} starts at {curr_op.start_time} "
                    f"but previous operation for job {prev_op.job_id} ends at {prev_op.end_time}"
                )
    
    return True 