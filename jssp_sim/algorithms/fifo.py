import time
from typing import List, Dict, Tuple
from copy import deepcopy

from jssp_sim.core.instance import Instance, Operation
from jssp_sim.core.metrics import calculate_metrics, validate_schedule


def run_fifo(instance: Instance) -> Tuple[Dict, List[Operation]]:
    """
    Run the First-In-First-Out (FIFO) scheduling algorithm.
    
    Operations are scheduled in the order they are defined in each job,
    with priority given to jobs in order of their ID.
    
    Args:
        instance: The job shop problem instance
        
    Returns:
        Tuple of (metrics, scheduled_operations)
    """
    start_time = time.time()
    
    # Create a deep copy of the operations to avoid modifying the original instance
    operations = deepcopy(instance.operations)
    
    # Create a mapping to track which copied operation corresponds to which original operation
    # and to determine the correct sequence
    operation_mapping = {}
    for i, original_op in enumerate(instance.operations):
        for copied_op in operations:
            if (original_op.job_id == copied_op.job_id and 
                original_op.machine_id == copied_op.machine_id and
                original_op.processing_time == copied_op.processing_time):
                operation_mapping[copied_op] = i
                break
    
    # Keep track of when each machine and job was last used
    machine_availability = {m: 0 for m in range(instance.num_machines)}
    job_availability = {j: 0 for j in range(instance.num_jobs)}
    
    # Scheduled operations list
    scheduled_ops = []
    
    # Process operations in order of job ID
    for job_id in range(instance.num_jobs):
        # Get operations for this job
        job_ops = [op for op in operations if op.job_id == job_id]
        
        # Sort operations based on their order in the original job
        # For each job, the operations must be processed in a specific sequence
        job_ops.sort(key=lambda op: operation_mapping[op])
        
        for op in job_ops:
            # Calculate the earliest start time for this operation
            # It must wait for both the machine to be available and the previous operation in the job to finish
            earliest_start = max(machine_availability[op.machine_id], job_availability[op.job_id])
            
            # Schedule the operation
            op.start_time = earliest_start
            op.end_time = earliest_start + op.processing_time
            
            # Update machine and job availability
            machine_availability[op.machine_id] = op.end_time
            job_availability[op.job_id] = op.end_time
            
            scheduled_ops.append(op)
    
    # Calculate CPU time
    cpu_time = time.time() - start_time
    
    # Validate the schedule
    validate_schedule(instance, scheduled_ops)
    
    # Calculate metrics
    metrics = calculate_metrics(instance, scheduled_ops)
    metrics["algorithm"] = "FIFO"
    metrics["cpu_time"] = cpu_time
    
    return metrics, scheduled_ops 