class GAParams:
    """Parameters for the Genetic Algorithm."""
    
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism_rate: float = 0.1,
        tournament_size: int = 3
    ):
        """
        Create a new set of parameters for the genetic algorithm.
        
        Args:
            population_size: Size of the population
            generations: Number of generations to run
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elitism_rate: Proportion of population to preserve via elitism
            tournament_size: Size of tournament for selection
        """
        if population_size < 2:
            raise ValueError("Population size must be at least 2")
        if generations < 1:
            raise ValueError("Number of generations must be at least 1")
        if not 0 <= crossover_rate <= 1:
            raise ValueError("Crossover rate must be between 0 and 1")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        if not 0 <= elitism_rate <= 1:
            raise ValueError("Elitism rate must be between 0 and 1")
        if tournament_size < 2:
            raise ValueError("Tournament size must be at least 2")
        
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
    
    def to_dict(self):
        """Convert the parameters to a dictionary for serialization."""
        return {
            "population_size": self.population_size,
            "generations": self.generations,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "elitism_rate": self.elitism_rate,
            "tournament_size": self.tournament_size
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create a parameters object from a dictionary."""
        return cls(
            population_size=data.get("population_size", 50),
            generations=data.get("generations", 100),
            crossover_rate=data.get("crossover_rate", 0.8),
            mutation_rate=data.get("mutation_rate", 0.1),
            elitism_rate=data.get("elitism_rate", 0.1),
            tournament_size=data.get("tournament_size", 3)
        ) 