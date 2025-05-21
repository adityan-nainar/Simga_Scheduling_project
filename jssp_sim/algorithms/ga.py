import time
import random
import numpy as np
from typing import List, Dict, Tuple, Any, Set
from copy import deepcopy

from jssp_sim.core.instance import Instance, Operation
from jssp_sim.core.params import GAParams
from jssp_sim.core.metrics import calculate_metrics, validate_schedule


class Chromosome:
    """
    Represents a chromosome in the genetic algorithm.
    
    This uses a job-based encoding where each chromosome is a permutation of job indices.
    This encoding ensures that job precedence constraints are always respected.
    """
    
    def __init__(self, job_sequence: List[int], instance: Instance = None):
        """
        Create a new chromosome.
        
        Args:
            job_sequence: List of job IDs representing the order of priority for scheduling
            instance: The problem instance (needed for evaluation)
        """
        self.job_sequence = job_sequence
        self.instance = instance
        self.makespan = None
        self.schedule = None
    
    def evaluate(self, instance: Instance = None):
        """
        Evaluate the chromosome and calculate its fitness (makespan).
        
        Args:
            instance: The problem instance to evaluate against
        """
        if instance is not None:
            self.instance = instance
            
        if self.instance is None:
            raise ValueError("Cannot evaluate without an instance")
        
        operations, makespan = self._decode()
        self.makespan = makespan
        self.schedule = operations
        return self.makespan
    
    def _decode(self) -> Tuple[List[Operation], int]:
        """
        Decode the chromosome into a feasible schedule.
        
        This uses a priority-based scheduling heuristic where jobs are considered
        in the order specified by the chromosome's job sequence.
        
        Returns:
            Tuple of (operations, makespan)
        """
        # Create a deep copy of the instance
        instance = deepcopy(self.instance)
        
        # Create a list of operations for each job, sorted by machine_id
        job_operations = {}
        for job in instance.jobs:
            job_operations[job.job_id] = sorted(list(job.operations), key=lambda op: op.machine_id)
        
        # Keep track of the next operation to schedule for each job
        job_op_index = {job_id: 0 for job_id in range(instance.num_jobs)}
        
        # Keep track of when each machine and job will next be available
        machine_availability = {m: 0 for m in range(instance.num_machines)}
        job_availability = {j: 0 for j in range(instance.num_jobs)}
        
        # List of all scheduled operations
        scheduled_ops = []
        
        # Keep scheduling until all operations are scheduled
        remaining_ops = sum(len(ops) for ops in job_operations.values())
        
        while remaining_ops > 0:
            # List of jobs that have operations ready to be scheduled
            available_jobs = []
            
            # Check each job to see if it has an operation ready
            for job_id in self.job_sequence:
                # If this job has no more operations, skip it
                if job_op_index[job_id] >= len(job_operations[job_id]):
                    continue
                
                # Get the next operation for this job
                next_op = job_operations[job_id][job_op_index[job_id]]
                
                # This job has an operation ready to be scheduled
                available_jobs.append(job_id)
            
            if not available_jobs:
                # This should never happen if our logic is correct
                break
            
            # Take the highest priority job from the available ones
            # (first one in the job_sequence that is available)
            selected_job = None
            for job_id in self.job_sequence:
                if job_id in available_jobs:
                    selected_job = job_id
                    break
            
            # Get the next operation for the selected job
            op = job_operations[selected_job][job_op_index[selected_job]]
            
            # Calculate the earliest possible start time
            # It must wait for both the job's previous operation and the machine to be free
            earliest_start = max(
                job_availability[op.job_id],         # When this job will be available
                machine_availability[op.machine_id]  # When this machine will be available
            )
            
            # Schedule the operation
            op.start_time = earliest_start
            op.end_time = earliest_start + op.processing_time
            
            # Update availability times
            job_availability[op.job_id] = op.end_time
            machine_availability[op.machine_id] = op.end_time
            
            # Add to scheduled operations
            scheduled_ops.append(op)
            
            # Move to the next operation for this job
            job_op_index[selected_job] += 1
            remaining_ops -= 1
        
        # Calculate makespan (maximum end time)
        makespan = max(op.end_time for op in scheduled_ops) if scheduled_ops else 0
        
        return scheduled_ops, makespan


def run_ga(instance: Instance, params: GAParams = None) -> Tuple[Dict, List[Operation]]:
    """
    Run the Genetic Algorithm for job shop scheduling.
    
    Args:
        instance: The job shop problem instance
        params: Parameters for the genetic algorithm (optional)
        
    Returns:
        Tuple of (metrics, scheduled_operations)
    """
    start_time = time.time()
    
    # Create a deep copy of the instance to avoid modifying the original
    copied_instance = deepcopy(instance)
    
    if params is None:
        params = GAParams()
    
    # Create a population of random chromosomes
    population = _initialize_population(copied_instance, params.population_size)
    
    # Evaluate the initial population
    for chromosome in population:
        chromosome.evaluate(copied_instance)
    
    # Best solution found so far
    best_solution = min(population, key=lambda c: c.makespan)
    best_makespan = best_solution.makespan
    
    # History of best makespan values (for analysis)
    history = [best_makespan]
    
    # Main GA loop
    for generation in range(params.generations):
        # Create the next generation
        next_population = []
        
        # Elitism: keep the best solutions
        elite_count = max(1, int(params.population_size * params.elitism_rate))
        elite = sorted(population, key=lambda c: c.makespan)[:elite_count]
        next_population.extend(elite)
        
        # Fill the rest with crossover and mutation
        while len(next_population) < params.population_size:
            # Selection
            parent1 = _tournament_selection(population, params.tournament_size)
            parent2 = _tournament_selection(population, params.tournament_size)
            
            # Crossover
            if random.random() < params.crossover_rate:
                offspring = _crossover(parent1, parent2)
            else:
                # No crossover, just copy one parent
                offspring = Chromosome(deepcopy(parent1.job_sequence), copied_instance)
            
            # Mutation
            if random.random() < params.mutation_rate:
                _mutate(offspring)
            
            # Evaluate the new solution
            offspring.evaluate(copied_instance)
            
            # Add to the next generation
            next_population.append(offspring)
        
        # Replace the current population
        population = next_population
        
        # Update the best solution
        current_best = min(population, key=lambda c: c.makespan)
        if current_best.makespan < best_makespan:
            best_solution = current_best
            best_makespan = current_best.makespan
        
        # Record history
        history.append(best_makespan)
    
    # Calculate CPU time
    cpu_time = time.time() - start_time
    
    # Get the best solution
    best_chromosome = min(population, key=lambda c: c.makespan)
    best_schedule = best_chromosome.schedule
    
    # Extra validation - should always pass with this encoding
    validate_schedule(copied_instance, best_schedule)
    
    # Calculate metrics
    metrics = calculate_metrics(copied_instance, best_schedule)
    metrics["algorithm"] = "GA"
    metrics["cpu_time"] = cpu_time
    metrics["generations"] = params.generations
    metrics["population_size"] = params.population_size
    metrics["best_makespan_history"] = history
    
    return metrics, best_schedule


def _initialize_population(instance: Instance, population_size: int) -> List[Chromosome]:
    """
    Initialize a random population of chromosomes.
    
    Args:
        instance: The job shop problem instance
        population_size: Size of the population
        
    Returns:
        List of Chromosome objects
    """
    population = []
    
    # Get the number of jobs
    num_jobs = instance.num_jobs
    
    for _ in range(population_size):
        # Create a random permutation of job indices
        job_sequence = list(range(num_jobs))
        random.shuffle(job_sequence)
        
        # Create a chromosome
        chromosome = Chromosome(job_sequence, instance)
        population.append(chromosome)
    
    return population


def _tournament_selection(population: List[Chromosome], tournament_size: int) -> Chromosome:
    """
    Select a chromosome using tournament selection.
    
    Args:
        population: List of chromosomes
        tournament_size: Number of chromosomes to include in the tournament
        
    Returns:
        Selected chromosome
    """
    # Randomly select tournament_size chromosomes
    tournament = random.sample(population, tournament_size)
    
    # Return the best chromosome from the tournament
    return min(tournament, key=lambda c: c.makespan)


def _crossover(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    """
    Perform Order Crossover (OX) between two parent chromosomes.
    
    This preserves the relative order of jobs in the sequence.
    
    Args:
        parent1: First parent chromosome
        parent2: Second parent chromosome
        
    Returns:
        New offspring chromosome
    """
    p1_seq = parent1.job_sequence
    p2_seq = parent2.job_sequence
    size = len(p1_seq)
    
    # Select two random crossover points
    cx_point1 = random.randint(0, size - 2)
    cx_point2 = random.randint(cx_point1 + 1, size - 1)
    
    # Initialize the offspring with a copy of the segment from parent1
    offspring_seq = [-1] * size
    for i in range(cx_point1, cx_point2 + 1):
        offspring_seq[i] = p1_seq[i]
    
    # Fill the remaining positions with elements from parent2 in order
    p2_idx = 0
    for i in range(size):
        if offspring_seq[i] == -1:  # Position needs to be filled
            # Find the next element in parent2 that's not already in offspring
            while p2_seq[p2_idx] in offspring_seq:
                p2_idx += 1
            offspring_seq[i] = p2_seq[p2_idx]
            p2_idx += 1
    
    # Create the offspring with the new sequence
    return Chromosome(offspring_seq, parent1.instance)


def _mutate(chromosome: Chromosome) -> None:
    """
    Mutate a chromosome in place using a swap mutation.
    
    Args:
        chromosome: The chromosome to mutate
    """
    # Get the job sequence
    seq = chromosome.job_sequence
    size = len(seq)
    
    if size <= 1:
        return
    
    # Select two random positions
    pos1, pos2 = random.sample(range(size), 2)
    
    # Swap the jobs at these positions
    seq[pos1], seq[pos2] = seq[pos2], seq[pos1] 