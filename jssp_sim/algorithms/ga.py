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
        start and end times for each operation, respecting both
        job precedence and machine capacity constraints.
        
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
        
        # Keep track of when each machine will be free
        machine_availability = {m: 0 for m in machine_ids}
        
        # Track the last scheduled operation's end time for each job
        job_last_op_end_times = {j: 0 for j in job_ids}
        
        # Dictionary to map job and machine to operation
        # This helps track machine sequence within each job
        job_machine_ops = {j: {} for j in job_ids}
        for op in operations:
            if op.machine_id not in job_machine_ops[op.job_id]:
                job_machine_ops[op.job_id][op.machine_id] = op
        
        # Sort operations within each job by machine_id (to match validation)
        scheduled_ops = []
        machine_schedules = {m: [] for m in machine_ids}
        
        # Process operations in the order given by the chromosome
        for op in operations:
            # Get the job and machine IDs
            job_id = op.job_id
            machine_id = op.machine_id
            
            # Determine the earliest start time based on job precedence
            # The operation can't start until its job's previous operations are done
            job_ready_time = job_last_op_end_times.get(job_id, 0)
            
            # Determine the earliest start time based on machine availability
            machine_ready_time = machine_availability.get(machine_id, 0)
            
            # The operation can start as soon as both the job and machine are ready
            start_time = max(job_ready_time, machine_ready_time)
            
            # Set the operation's schedule
            op.start_time = start_time
            op.end_time = start_time + op.processing_time
            
            # Update when this job and machine will next be available
            job_last_op_end_times[job_id] = op.end_time
            machine_availability[machine_id] = op.end_time
            
            # Add to the scheduled operations and the machine's schedule
            scheduled_ops.append(op)
            machine_schedules[machine_id].append(op)
        
        # After initial scheduling, we need to resolve any conflicts
        # For each machine, ensure operations don't overlap
        for machine_id in machine_ids:
            # Sort operations on this machine by start time
            machine_ops = machine_schedules[machine_id]
            machine_ops.sort(key=lambda op: op.start_time)
            
            # Fix any overlaps
            for i in range(1, len(machine_ops)):
                prev_op = machine_ops[i-1]
                curr_op = machine_ops[i]
                
                # If there's an overlap, push the current operation later
                if curr_op.start_time < prev_op.end_time:
                    # Adjust the current operation's times
                    curr_op.start_time = prev_op.end_time
                    curr_op.end_time = curr_op.start_time + curr_op.processing_time
                    
                    # Update when this job will be available next
                    job_last_op_end_times[curr_op.job_id] = max(
                        job_last_op_end_times[curr_op.job_id], 
                        curr_op.end_time
                    )
        
        # Now ensure job precedence constraints are met
        # For each job, sort operations by machine_id (as validation does)
        for job_id in job_ids:
            job_ops = [op for op in scheduled_ops if op.job_id == job_id]
            job_ops.sort(key=lambda op: op.machine_id)
            
            # Check precedence between operations
            for i in range(1, len(job_ops)):
                prev_op = job_ops[i-1]
                curr_op = job_ops[i]
                
                # If current op starts before previous op ends, adjust it
                if curr_op.start_time < prev_op.end_time:
                    curr_op.start_time = prev_op.end_time
                    curr_op.end_time = curr_op.start_time + curr_op.processing_time
                    
                    # Need to check machine constraints again for this modified operation
                    machine_id = curr_op.machine_id
                    machine_ops = [op for op in scheduled_ops if op.machine_id == machine_id and op != curr_op]
                    
                    # Sort operations on this machine by start time
                    machine_ops.sort(key=lambda op: op.start_time)
                    
                    # Find where our current op fits in this sequence
                    insert_idx = 0
                    while insert_idx < len(machine_ops) and machine_ops[insert_idx].start_time < curr_op.start_time:
                        insert_idx += 1
                    
                    # Check if we need to adjust due to the operation before us
                    if insert_idx > 0 and machine_ops[insert_idx-1].end_time > curr_op.start_time:
                        curr_op.start_time = machine_ops[insert_idx-1].end_time
                        curr_op.end_time = curr_op.start_time + curr_op.processing_time
                    
                    # Check if we need to adjust operations after us
                    if insert_idx < len(machine_ops):
                        next_op = machine_ops[insert_idx]
                        if next_op.start_time < curr_op.end_time:
                            next_op.start_time = curr_op.end_time
                            next_op.end_time = next_op.start_time + next_op.processing_time
        
        return scheduled_ops


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