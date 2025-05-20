import random
from typing import List, Dict, Tuple
import numpy as np

class Operation:
    """Represents a single operation in a job shop scheduling problem."""
    
    def __init__(self, job_id: int, machine_id: int, processing_time: int):
        self.job_id = job_id
        self.machine_id = machine_id
        self.processing_time = processing_time
        
        # These will be set by the scheduling algorithm
        self.start_time = None
        self.end_time = None
    
    def __repr__(self):
        status = "unscheduled" if self.start_time is None else f"scheduled: {self.start_time}-{self.end_time}"
        return f"Op(job={self.job_id}, machine={self.machine_id}, time={self.processing_time}, {status})"


class Job:
    """Represents a job consisting of a sequence of operations."""
    
    def __init__(self, job_id: int, operations: List[Operation]):
        self.job_id = job_id
        self.operations = operations
    
    def __repr__(self):
        return f"Job(id={self.job_id}, operations={len(self.operations)})"


class Instance:
    """Represents a job shop scheduling problem instance."""
    
    def __init__(self, jobs: int, machines: int, min_time: int = 1, max_time: int = 10, seed: int = None):
        """
        Create a new job shop scheduling problem instance.
        
        Args:
            jobs: Number of jobs
            machines: Number of machines
            min_time: Minimum processing time for operations
            max_time: Maximum processing time for operations
            seed: Random seed for reproducibility
        """
        if jobs < 2:
            raise ValueError("Number of jobs must be at least 2")
        if machines < 2:
            raise ValueError("Number of machines must be at least 2")
        if min_time < 1:
            raise ValueError("Minimum processing time must be at least 1")
        if max_time <= min_time:
            raise ValueError("Maximum processing time must be greater than minimum processing time")
        
        self.num_jobs = jobs
        self.num_machines = machines
        self.min_time = min_time
        self.max_time = max_time
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Generate the problem instance
        self.jobs = self._generate_jobs()
        
        # Flattened list of all operations
        self.operations = [op for job in self.jobs for op in job.operations]
    
    def _generate_jobs(self) -> List[Job]:
        """Generate random jobs with operations for each machine."""
        jobs = []
        
        for job_id in range(self.num_jobs):
            # For each job, generate a sequence of operations (one per machine)
            # Each operation is assigned to a unique machine
            machine_order = np.random.permutation(self.num_machines)
            
            operations = []
            for i, machine_id in enumerate(machine_order):
                proc_time = random.randint(self.min_time, self.max_time)
                operations.append(Operation(job_id, machine_id, proc_time))
            
            jobs.append(Job(job_id, operations))
        
        return jobs
    
    def to_dict(self) -> Dict:
        """Convert the instance to a dictionary for serialization."""
        return {
            "jobs": self.num_jobs,
            "machines": self.num_machines,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "operations": [
                {
                    "job_id": op.job_id,
                    "machine_id": op.machine_id,
                    "processing_time": op.processing_time
                }
                for op in self.operations
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Instance':
        """Create an instance from a dictionary representation."""
        instance = cls(
            jobs=data["jobs"],
            machines=data["machines"],
            min_time=data["min_time"],
            max_time=data["max_time"]
        )
        
        # Override the generated operations with the provided ones
        instance.operations = []
        instance.jobs = []
        
        job_operations = {job_id: [] for job_id in range(data["jobs"])}
        
        for op_data in data["operations"]:
            op = Operation(
                job_id=op_data["job_id"],
                machine_id=op_data["machine_id"],
                processing_time=op_data["processing_time"]
            )
            instance.operations.append(op)
            job_operations[op.job_id].append(op)
        
        for job_id, operations in job_operations.items():
            instance.jobs.append(Job(job_id, operations))
        
        return instance 