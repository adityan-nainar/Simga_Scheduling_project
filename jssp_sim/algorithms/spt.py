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
    
    # Create a deep copy of the instance to avoid modifying the original
    copied_instance = deepcopy(instance)
    
    # Keep track of when each machine and job was last used
    machine_availability = {m: 0 for m in range(copied_instance.num_machines)}
    job_availability = {j: 0 for j in range(copied_instance.num_jobs)}
    
    # Scheduled operations list
    scheduled_ops = []
    
    # Keep track of which operations have been scheduled
    scheduled_op_ids = set()
    
    # Group operations by job and sort by machine_id to match the validation logic
    job_operations = {}
    for job in copied_instance.jobs:
        # Get operations for this job
        job_operations[job.job_id] = list(job.operations)
        # Sort operations by machine_id to match validation logic
        job_operations[job.job_id].sort(key=lambda op: op.machine_id)
    
    # Continue until all operations are scheduled
    while len(scheduled_op_ids) < len(copied_instance.operations):
        # Find all operations that are available to be scheduled
        available_ops = []
        
        for job_id, ops in job_operations.items():
            if not ops:
                continue
                
            # The first operation in the list is the next one to be scheduled for this job
            next_op = ops[0]
            
            # Check if this is the first operation for this job or
            # if all the preceding operations for this job have been scheduled
            # For machine_id-ordered operations, we need to check the operation before this one
            can_schedule = True
            if scheduled_ops:
                # Get all scheduled operations for this job
                job_scheduled_ops = [sop for sop in scheduled_ops if sop.job_id == job_id]
                if job_scheduled_ops:
                    # Sort by machine_id to match validation logic
                    job_scheduled_ops.sort(key=lambda sop: sop.machine_id)
                    # Find position of current op in the machine order
                    all_job_ops = job_scheduled_ops + ops
                    all_job_ops.sort(key=lambda op: op.machine_id)
                    machine_idx = [op.machine_id for op in all_job_ops].index(next_op.machine_id)
                    
                    # If this is not the first operation by machine_id,
                    # the previous one must be scheduled
                    if machine_idx > 0:
                        prev_machine_id = all_job_ops[machine_idx - 1].machine_id
                        # Check if the prev operation has been scheduled
                        prev_op_scheduled = any(
                            sop.job_id == job_id and sop.machine_id == prev_machine_id
                            for sop in job_scheduled_ops
                        )
                        if not prev_op_scheduled:
                            can_schedule = False
            
            if can_schedule:
                available_ops.append(next_op)
        
        if not available_ops:
            # No operations are available, which shouldn't happen if the problem is well-formed
            raise ValueError("No operations available to schedule - job precedence constraints may be circular")
        
        # Sort available operations by processing time (SPT)
        available_ops.sort(key=lambda op: op.processing_time)
        
        # Take the operation with the shortest processing time
        next_op = available_ops[0]
        
        # Calculate the earliest start time for this operation
        # It must wait for both the machine to be available and any predecessor in its job to finish
        earliest_machine_time = machine_availability[next_op.machine_id]
        
        # Find the previous operation in this job by machine_id
        prev_op_end_time = 0
        job_scheduled_ops = [sop for sop in scheduled_ops if sop.job_id == next_op.job_id]
        if job_scheduled_ops:
            # Sort by machine_id
            job_scheduled_ops.sort(key=lambda sop: sop.machine_id)
            # Find the machine index of the current operation
            all_machines = [sop.machine_id for sop in job_scheduled_ops + [next_op]]
            all_machines.sort()
            machine_idx = all_machines.index(next_op.machine_id)
            # If not the first machine, get the end time of the previous operation
            if machine_idx > 0:
                prev_machine = all_machines[machine_idx - 1]
                for sop in job_scheduled_ops:
                    if sop.machine_id == prev_machine:
                        prev_op_end_time = sop.end_time
        
        # The operation can start as soon as both the machine and any predecessor are ready
        earliest_start = max(earliest_machine_time, prev_op_end_time)
        
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
    validate_schedule(copied_instance, scheduled_ops)
    
    # Calculate metrics
    metrics = calculate_metrics(copied_instance, scheduled_ops)
    metrics["algorithm"] = "SPT"
    metrics["cpu_time"] = cpu_time
    
    return metrics, scheduled_ops 