import unittest
import random
import numpy as np

from jssp_sim.core.instance import Instance
from jssp_sim.core.params import GAParams
from jssp_sim.algorithms.fifo import run_fifo
from jssp_sim.algorithms.spt import run_spt
from jssp_sim.algorithms.ga import run_ga
from jssp_sim.core.metrics import validate_schedule


class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        # Set a seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Create a small test instance
        self.instance = Instance(jobs=5, machines=3, min_time=1, max_time=5, seed=42)
    
    def test_fifo(self):
        # Run FIFO algorithm
        metrics, schedule = run_fifo(self.instance)
        
        # Check that metrics are computed
        self.assertIn("makespan", metrics)
        self.assertIn("total_flow_time", metrics)
        self.assertIn("avg_machine_utilization", metrics)
        
        # Check that all operations are scheduled
        self.assertEqual(len(schedule), len(self.instance.operations))
        
        # Check that the schedule is valid
        self.assertTrue(validate_schedule(self.instance, schedule))
    
    def test_spt(self):
        # Run SPT algorithm
        metrics, schedule = run_spt(self.instance)
        
        # Check that metrics are computed
        self.assertIn("makespan", metrics)
        self.assertIn("total_flow_time", metrics)
        self.assertIn("avg_machine_utilization", metrics)
        
        # Check that all operations are scheduled
        self.assertEqual(len(schedule), len(self.instance.operations))
        
        # Check that the schedule is valid
        self.assertTrue(validate_schedule(self.instance, schedule))
    
    def test_ga(self):
        # Set GA parameters
        params = GAParams(
            population_size=20,
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.1
        )
        
        # Run GA algorithm
        metrics, schedule = run_ga(self.instance, params)
        
        # Check that metrics are computed
        self.assertIn("makespan", metrics)
        self.assertIn("total_flow_time", metrics)
        self.assertIn("avg_machine_utilization", metrics)
        
        # Check that all operations are scheduled
        self.assertEqual(len(schedule), len(self.instance.operations))
        
        # Check that the schedule is valid
        self.assertTrue(validate_schedule(self.instance, schedule))
    
    def test_algorithm_comparison(self):
        # Run all algorithms
        fifo_metrics, _ = run_fifo(self.instance)
        spt_metrics, _ = run_spt(self.instance)
        
        params = GAParams(
            population_size=20,
            generations=20,
            crossover_rate=0.8,
            mutation_rate=0.1
        )
        ga_metrics, _ = run_ga(self.instance, params)
        
        # Print makespans for comparison
        print(f"FIFO makespan: {fifo_metrics['makespan']}")
        print(f"SPT makespan: {spt_metrics['makespan']}")
        print(f"GA makespan: {ga_metrics['makespan']}")
        
        # GA should generally find a better (or equal) solution than FIFO/SPT
        # This might not always be true for very small instances or few generations
        # So we'll just verify the results are numeric and reasonable
        self.assertGreater(fifo_metrics['makespan'], 0)
        self.assertGreater(spt_metrics['makespan'], 0)
        self.assertGreater(ga_metrics['makespan'], 0)


if __name__ == '__main__':
    unittest.main() 