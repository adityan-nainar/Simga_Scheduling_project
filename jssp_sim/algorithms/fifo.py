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
    
    # Make a deep copy of the instance to avoid modifying the original
    copied_instance = deepcopy(instance)
    
    # Keep track of when each machine and job was last used
    machine_availability = {m: 0 for m in range(copied_instance.num_machines)}
    job_availability = {j: 0 for j in range(copied_instance.num_jobs)}
    
    # Scheduled operations list
    scheduled_ops = []
    
    # Process jobs in order of their ID (FIFO)
    for job in copied_instance.jobs:
        # Get all operations for this job
        # The operations in job.operations are already in the correct sequence
        for op in job.operations:
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
    validate_schedule(copied_instance, scheduled_ops)
    
    # Calculate metrics
    metrics = calculate_metrics(copied_instance, scheduled_ops)
    metrics["algorithm"] = "FIFO"
    metrics["cpu_time"] = cpu_time
    
    return metrics, scheduled_ops 