import time
from typing import List, Dict, Tuple
from copy import deepcopy

from jssp_sim.core.instance import Instance, Operation
from jssp_sim.core.metrics import calculate_metrics, validate_schedule


def run_spt(instance: Instance) -> Tuple[Dict, List[Operation]]:
    """
    Run the Shortest Processing Time (SPT) scheduling algorithm.
    
    Operations are scheduled in order of their processing time,
    with the shortest operations scheduled first. Precedence constraints 
    within jobs are still respected.
    
    Args:
        instance: The job shop problem instance
        
    Returns:
        Tuple of (metrics, scheduled_operations)
    """
    start_time = time.time()
    
    # Create a deep copy of the operations to avoid modifying the original instance
    operations = deepcopy(instance.operations)
    
    # Keep track of when each machine and job was last used
    machine_availability = {m: 0 for m in range(instance.num_machines)}
    job_availability = {j: 0 for j in range(instance.num_jobs)}
    
    # Scheduled operations list
    scheduled_ops = []
    
    # Keep track of which operations have been scheduled
    scheduled_op_ids = set()
    
    # Group operations by job
    job_operations = {}
    for job in instance.jobs:
        job_operations[job.job_id] = [op for op in operations if op.job_id == job.job_id]
        # Sort operations within each job by their machine order
        job_operations[job.job_id].sort(key=lambda op: job.operations.index(op))
    
    # Continue until all operations are scheduled
    while len(scheduled_op_ids) < len(operations):
        # Find all operations that are available to be scheduled
        available_ops = []
        
        for job_id, ops in job_operations.items():
            if not ops:
                continue
                
            # The first operation in the list is the next one to be scheduled for this job
            next_op = ops[0]
            
            # Check if this operation's predecessors have been scheduled
            if len(scheduled_ops) == 0 or all(pred.end_time is not None for pred in scheduled_ops if pred.job_id == job_id):
                # Operation is available
                available_ops.append(next_op)
        
        if not available_ops:
            # No operations are available, which shouldn't happen if the problem is well-formed
            raise ValueError("No operations available to schedule")
        
        # Sort available operations by processing time (SPT)
        available_ops.sort(key=lambda op: op.processing_time)
        
        # Take the operation with the shortest processing time
        next_op = available_ops[0]
        
        # Calculate the earliest start time for this operation
        # It must wait for both the machine to be available and any predecessor operation in the job to finish
        earliest_start = max(
            machine_availability[next_op.machine_id],
            job_availability[next_op.job_id]
        )
        
        # Schedule the operation
        next_op.start_time = earliest_start
        next_op.end_time = earliest_start + next_op.processing_time
        
        # Update machine and job availability
        machine_availability[next_op.machine_id] = next_op.end_time
        job_availability[next_op.job_id] = next_op.end_time
        
        # Add to scheduled operations
        scheduled_ops.append(next_op)
        scheduled_op_ids.add(id(next_op))
        
        # Remove the operation from its job's list
        job_operations[next_op.job_id].pop(0)
    
    # Calculate CPU time
    cpu_time = time.time() - start_time
    
    # Validate the schedule
    validate_schedule(instance, scheduled_ops)
    
    # Calculate metrics
    metrics = calculate_metrics(instance, scheduled_ops)
    metrics["algorithm"] = "SPT"
    metrics["cpu_time"] = cpu_time
    
    return metrics, scheduled_ops 