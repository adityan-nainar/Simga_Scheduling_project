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
    
    # Create a deep copy of the instance to avoid modifying the original
    copied_instance = deepcopy(instance)
    
    # Keep track of when each machine is available
    machine_availability = {m: 0 for m in range(copied_instance.num_machines)}
    
    # Create lists to hold the operations for each job
    job_operations = []
    for job in copied_instance.jobs:
        # Store the operations in their exact sequence
        job_operations.append(list(job.operations))
    
    # List to hold all scheduled operations
    scheduled_ops = []
    
    # Keep scheduling operations until all jobs are done
    while any(job_ops for job_ops in job_operations):
        for job_idx, job_ops in enumerate(job_operations):
            if not job_ops:  # Skip empty job operation lists
                continue
                
            # Get the next operation for this job
            op = job_ops[0]
            
            # Calculate when this operation can start
            # If it's the first operation in the job, can start as soon as machine is available
            # Otherwise, need to wait for the previous operation in this job to finish
            prev_op_end_time = 0
            if scheduled_ops:
                # Look for the last scheduled operation from this job
                for scheduled_op in reversed(scheduled_ops):
                    if scheduled_op.job_id == job_idx:
                        prev_op_end_time = scheduled_op.end_time
                        break
            
            # The operation can start as soon as both the previous operation in the job
            # and the required machine are available
            earliest_start = max(prev_op_end_time, machine_availability[op.machine_id])
            
            # Schedule this operation
            op.start_time = earliest_start
            op.end_time = earliest_start + op.processing_time
            
            # Update machine availability
            machine_availability[op.machine_id] = op.end_time
            
            # Add to scheduled operations and remove from job's pending operations
            scheduled_ops.append(op)
            job_operations[job_idx].pop(0)
            
            # After scheduling one operation, break the loop to give other jobs a chance
            # (FIFO prioritizes by job ID, but still processes operations in sequence)
            break
    
    # Calculate CPU time
    cpu_time = time.time() - start_time
    
    # Validate the schedule
    validate_schedule(copied_instance, scheduled_ops)
    
    # Calculate metrics
    metrics = calculate_metrics(copied_instance, scheduled_ops)
    metrics["algorithm"] = "FIFO"
    metrics["cpu_time"] = cpu_time
    
    return metrics, scheduled_ops 