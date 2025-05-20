import time
import random
import numpy as np
from typing import List, Dict, Tuple, Any
from copy import deepcopy

from jssp_sim.core.instance import Instance, Operation
from jssp_sim.core.params import GAParams
from jssp_sim.core.metrics import calculate_metrics, validate_schedule


class Chromosome:
    """Represents a single chromosome in the genetic algorithm."""
    
    def __init__(self, operations: List[Operation], makespan: float = None):
        self.operations = operations
        self.makespan = makespan  # Fitness value
        
    def evaluate(self, instance: Instance):
        """Evaluate the chromosome and calculate its fitness (makespan)."""
        try:
            ops_copy = self._decode(instance)
            metrics = calculate_metrics(instance, ops_copy)
            self.makespan = metrics["makespan"]
            return self.makespan
        except (ValueError, TypeError) as e:
            # Return a very high makespan for invalid schedules
            self.makespan = float('inf')
            return self.makespan
    
    def _decode(self, instance: Instance) -> List[Operation]:
        """
        Decode the chromosome into a feasible schedule.
        
        This converts the operation sequence into a schedule with
        start and end times for each operation, following job precedence constraints.
        
        Args:
            instance: The problem instance to decode against (used for validation)
            
        Returns:
            List of scheduled operations
        """
        # Make a deep copy of operations to avoid modifying the original
        operations = deepcopy(self.operations)
        
        # Extract job and machine count from operations
        job_ids = set(op.job_id for op in operations)
        machine_ids = set(op.machine_id for op in operations)
        
        # Keep track of when each machine is available
        machine_availability = {m: 0 for m in machine_ids}
        
        # Dictionary to track the last scheduled operation's end time for each job
        job_last_end_times = {j: {} for j in job_ids}  # {job_id: {machine_id: end_time}}
        
        # Get operations by job and sort by machine_id to match validation logic
        job_operations = {j: [] for j in job_ids}
        for op in operations:
            job_operations[op.job_id].append(op)
            
        # Sort operations within each job by machine_id (to match validation)
        for j in job_ids:
            job_operations[j].sort(key=lambda op: op.machine_id)
            
        # Process operations in the order given by the chromosome
        for op in operations:
            # The operation's job id and machine id
            job_id = op.job_id
            machine_id = op.machine_id
            
            # Get when the machine will be available
            machine_ready_time = machine_availability.get(machine_id, 0)
            
            # Calculate when this job's preceding operation (by machine_id) completes
            job_ready_time = 0
            
            # Get all machines for this job in sorted order
            job_machines = sorted([o.machine_id for o in job_operations[job_id]])
            
            # Find the machine that comes before this one in the job's sequence
            for prev_m in job_machines:
                if prev_m >= machine_id:
                    break
                # If we have an end time for the previous machine, use it
                if prev_m in job_last_end_times[job_id]:
                    job_ready_time = max(job_ready_time, job_last_end_times[job_id][prev_m])
            
            # The operation can start as soon as both the job and machine are ready
            start_time = max(job_ready_time, machine_ready_time)
            
            # Schedule the operation
            op.start_time = start_time
            op.end_time = start_time + op.processing_time
            
            # Update the machine availability
            machine_availability[machine_id] = op.end_time
            
            # Update the last end time for this job
            job_last_end_times[job_id][machine_id] = op.end_time
            
        # Verify operations in each job are scheduled in the correct order
        # according to validation logic (sorted by machine_id)
        for j in job_ids:
            job_ops = [op for op in operations if op.job_id == j]
            job_ops.sort(key=lambda op: op.machine_id)
            
            for i in range(1, len(job_ops)):
                prev_op = job_ops[i-1]
                curr_op = job_ops[i]
                
                # Make sure current operation doesn't start before the previous one ends
                if curr_op.start_time < prev_op.end_time:
                    curr_op.start_time = prev_op.end_time
                    curr_op.end_time = curr_op.start_time + curr_op.processing_time
                    # Update machine availability too
                    machine_availability[curr_op.machine_id] = curr_op.end_time
        
        return operations


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
        elite_count = int(params.population_size * params.elitism_rate)
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
                offspring = Chromosome(deepcopy(parent1.operations))
            
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
    best_schedule = best_chromosome._decode(copied_instance)
    
    # Validate the schedule
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
    
    for _ in range(population_size):
        # Create a random permutation of operations
        operations = deepcopy(instance.operations)
        random.shuffle(operations)
        
        # Create a chromosome from the permutation
        chromosome = Chromosome(operations)
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
    Perform crossover between two parent chromosomes.
    
    This implements the Precedence Operation Crossover (POX) which preserves the
    relative order of operations within jobs.
    
    Args:
        parent1: First parent chromosome
        parent2: Second parent chromosome
        
    Returns:
        New offspring chromosome
    """
    p1_ops = parent1.operations
    p2_ops = parent2.operations
    
    # Extract job IDs
    job_ids = set(op.job_id for op in p1_ops)
    
    # Randomly select jobs to inherit from parent1
    job_count = len(job_ids)
    jobs_from_p1 = set(random.sample(list(job_ids), job_count // 2))
    
    # Create offspring
    offspring_ops = []
    
    # First, inherit selected jobs from parent1 (maintain their relative order)
    for op in p1_ops:
        if op.job_id in jobs_from_p1:
            offspring_ops.append(op)
    
    # Then, inherit remaining jobs from parent2 (maintain their relative order)
    for op in p2_ops:
        if op.job_id not in jobs_from_p1:
            offspring_ops.append(op)
    
    return Chromosome(offspring_ops)


def _mutate(chromosome: Chromosome) -> None:
    """
    Mutate a chromosome in place.
    
    Performs a swap mutation by selecting two random positions
    and swapping the operations at those positions.
    
    Args:
        chromosome: The chromosome to mutate
    """
    if len(chromosome.operations) <= 1:
        return
    
    # Select two random positions
    pos1, pos2 = random.sample(range(len(chromosome.operations)), 2)
    
    # Swap the operations
    chromosome.operations[pos1], chromosome.operations[pos2] = chromosome.operations[pos2], chromosome.operations[pos1] 