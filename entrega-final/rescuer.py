##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### This rescuer version implements:
### - clustering of victims by quadrants of the explored region 
### - definition of a rescue sequence for victims of a cluster using a Genetic Algorithm
### - selecting the ordering (GA vs. greedy) based on actual BFS cost (to avoid long routes)
### - assigning one cluster to one rescuer
### - calculating paths between pairs of victims using breadth-first search
###
### One of the rescuers is the master in charge of unifying the maps and the information
### about the found victims.

import os
import random
import math
import csv
import sys
from map import Map
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from bfs import BFS
from abc import ABC, abstractmethod
import regressor
import classifier

# ---------------------------------------------------------------------------
# Genetic Algorithm helper function for optimizing victim visit order
def ga_optimize_sequence(victims, population_size=50, generations=100, mutation_rate=0.1, tournament_size=5):
    """
    Optimize the visiting order (route) for victims using a Genetic Algorithm.
    
    :param victims: Dictionary of victims, where each key is a victim id and the value is ((x, y), [vital signals])
    :param population_size: Number of individuals in the GA population.
    :param generations: Number of generations to run.
    :param mutation_rate: Probability of mutation for each offspring.
    :param tournament_size: Number of individuals to consider in tournament selection.
    :return: A tuple (best_route, best_distance) where best_route is a list of victim ids in the optimized order.
    """
    # List of victim IDs (chromosome genes)
    victim_ids = list(victims.keys())
    
    def euclidean_distance(a, b):
        return math.dist(a, b)
    
    def total_distance(route):
        # Calculate total distance based on Euclidean distances:
        cost = euclidean_distance((0, 0), victims[route[0]][0])
        for i in range(len(route) - 1):
            cost += euclidean_distance(victims[route[i]][0], victims[route[i + 1]][0])
        cost += euclidean_distance(victims[route[-1]][0], (0, 0))
        return cost

    # Initialize population with random permutations.
    population = []
    for _ in range(population_size):
        perm = victim_ids.copy()
        random.shuffle(perm)
        population.append(perm)

    # Tournament selection: choose the best individual among a random sample.
    def tournament_selection(pop):
        selected = random.sample(pop, tournament_size)
        selected.sort(key=lambda route: total_distance(route))
        return selected[0]

    # Order Crossover (OX) for permutations.
    def order_crossover(parent1, parent2):
        size = len(parent1)
        child1 = [None] * size
        child2 = [None] * size
        start = random.randint(0, size - 1)
        end = random.randint(start, size - 1)
        child1[start:end + 1] = parent1[start:end + 1]
        child2[start:end + 1] = parent2[start:end + 1]

        # Fill in missing genes while preserving order.
        def fill_child(child, parent):
            current = 0
            for gene in parent:
                if gene not in child:
                    while child[current] is not None:
                        current += 1
                    child[current] = gene
            return child

        child1 = fill_child(child1, parent2)
        child2 = fill_child(child2, parent1)
        return child1, child2

    # Swap Mutation: randomly exchange two genes.
    def swap_mutation(individual):
        a = random.randint(0, len(individual) - 1)
        b = random.randint(0, len(individual) - 1)
        individual[a], individual[b] = individual[b], individual[a]
        return individual

    best_route = None
    best_distance = float('inf')

    # Main GA loop.
    for gen in range(generations):
        new_population = []
        # Create new offspring until the new population is full.
        while len(new_population) < population_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1, child2 = order_crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child1 = swap_mutation(child1)
            if random.random() < mutation_rate:
                child2 = swap_mutation(child2)
            new_population.extend([child1, child2])
        population = new_population[:population_size]  # Maintain population size.

        # Update the best found solution.
        for route in population:
            d = total_distance(route)
            if d < best_distance:
                best_distance = d
                best_route = route

    return best_route, best_distance

# ---------------------------------------------------------------------------
# Rescuer agent class with GA-based sequencing integrated
class Rescuer(AbstAgent):
    def __init__(self, env, config_file, nb_of_explorers=1, clusters=[]):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file
        @param nb_of_explorers: number of explorer agents to wait for
        @param clusters: list of clusters of victims in the charge of this agent
        """
        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.nb_of_explorers = nb_of_explorers       # number of explorer agents to wait for start
        self.received_maps = 0                       # counts the number of explorers' maps
        self.map = Map()                             # explorer will pass the map
        self.victims = {}                            # a dictionary of found victims: [vic_id]: ((x,y), [<vs>])
        self.plan = []                               # a list of planned actions (dx, dy)
        self.plan_x = 0                              # the x position during planning phase
        self.plan_y = 0                              # the y position during planning phase
        self.plan_visited = set()                    # positions already planned to be visited 
        self.plan_rtime = self.TLIM                  # remaining time during the planning phase
        self.plan_walk_time = 0.0                    # previewed walking time during rescue
        self.x = 0                                   # current x position during execution
        self.y = 0                                   # current y position during execution
        self.clusters = clusters                     # clusters of victims assigned to this agent
        self.sequences = clusters                    # rescue sequence for each cluster 
                
        # Starts in IDLE state; becomes ACTIVE once the map is received.
        self.set_state(VS.IDLE)

    def save_cluster_csv(self, cluster, cluster_id):
        os.makedirs('./clusters', exist_ok=True)
        filename = f"./clusters/cluster{cluster_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for vic_id, values in cluster.items():
                x, y = values[0]      # victim's x,y coordinates
                vs = values[1]        # list of vital signals
                writer.writerow([vic_id, x, y, vs[6], vs[7]])

    def save_sequence_csv(self, sequence, sequence_id):
        filename = f"./clusters/seq{sequence_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for id, values in sequence.items():
                x, y = values[0]      # victim's x,y coordinates
                vs = values[1]        # list of vital signals
                writer.writerow([id, x, y, vs[6], vs[7]])

    def cluster_victims(self):
        """
        Naively cluster victims into four quadrants.
        @returns: a list of clusters (dictionaries) keyed by victim id.
        """
        lower_xlim = sys.maxsize    
        lower_ylim = sys.maxsize
        upper_xlim = -sys.maxsize - 1
        upper_ylim = -sys.maxsize - 1
    
        for key, values in self.victims.items():
            x, y = values[0]
            lower_xlim = min(lower_xlim, x) 
            upper_xlim = max(upper_xlim, x)
            lower_ylim = min(lower_ylim, y)
            upper_ylim = max(upper_ylim, y)
        
        mid_x = lower_xlim + (upper_xlim - lower_xlim) / 2
        mid_y = lower_ylim + (upper_ylim - lower_ylim) / 2
    
        upper_left = {}
        upper_right = {}
        lower_left = {}
        lower_right = {}
        
        for key, values in self.victims.items():
            x, y = values[0]
            if x <= mid_x:
                if y <= mid_y:
                    upper_left[key] = values
                else:
                    lower_left[key] = values
            else:
                if y <= mid_y:
                    upper_right[key] = values
                else:
                    lower_right[key] = values
    
        return [upper_left, upper_right, lower_left, lower_right]

    def predict_severity_and_class(self):
        """
        Predict severity and class for each victim using external regressor and classifier.
        """
        dataset = []
        for _, values in self.victims.items():
            _, victim_data = values
            dataset.append(victim_data)

        severity_predicted = regressor.decision_tree_regressor("NOT_PRESSURE", dataset)
        class_predict = classifier.decision_tree_classifier("NOT_PRESSURE", dataset)

        for i, (_, values) in enumerate(self.victims.items()):
            values[1].extend([int(severity_predicted[i]), int(class_predict[i])])

    def sequencing(self):
        """
        Greedy sequencing based on Euclidean distance from base (0,0).
        (This method builds an ordering without modifying self.sequences permanently.)
        """
        ordering = {}
        current_position = (0, 0)
        remaining = list(self.clusters[0].items())
        while remaining:
            next_victim = min(
                remaining,
                key=lambda item: math.dist(current_position, item[1][0])
            )
            vic_id, data = next_victim
            ordering[vic_id] = data
            current_position = data[0]
            remaining.remove(next_victim)
        return ordering

    def ga_sequencing(self):
        """
        Compute a GA-based ordering.
        Returns a dictionary ordering victims based on the GA route.
        """
        cluster = self.clusters[0]
        best_route, best_distance = ga_optimize_sequence(
            cluster, 
            population_size=50, 
            generations=100, 
            mutation_rate=0.1, 
            tournament_size=5
        )
        print(f"GA best route (Euclidean cost): {best_route} with cost: {best_distance:.2f}")
        ordering = {}
        for vic_id in best_route:
            ordering[vic_id] = cluster[vic_id]
        return ordering

    def compute_bfs_cost_for_ordering(self, ordering):
        """
        Compute the actual BFS cost for a given victim ordering.
        It uses BFS to compute the path cost from base to the first victim,
        between victims, and back to base.
        """
        total_cost = 0
        bfs_instance = BFS(self.map, self.COST_LINE, self.COST_DIAG)
        start = (0, 0)
        # Use a high time limit so that BFS finds the path
        tlim = float('inf')
        for vic_id, data in ordering.items():
            goal = data[0]
            _, cost = bfs_instance.search(start, goal, tlim)
            total_cost += cost
            start = goal
        # Add cost to return to base
        _, cost = bfs_instance.search(start, (0, 0), tlim)
        total_cost += cost
        return total_cost

    def select_best_ordering(self):
        """
        Computes both the greedy and GA orderings, then selects the one with the lower BFS cost.
        The chosen ordering is then stored in self.sequences.
        """
        # Build greedy ordering from the current cluster
        greedy_ordering = self.sequencing()
        # Build GA ordering
        ga_ordering = self.ga_sequencing()

        cost_greedy = self.compute_bfs_cost_for_ordering(greedy_ordering)
        cost_ga = self.compute_bfs_cost_for_ordering(ga_ordering)
        print(f"Greedy BFS cost: {cost_greedy:.2f}, GA BFS cost: {cost_ga:.2f}")
        if cost_ga < cost_greedy:
            print("Selecting GA ordering.")
            self.sequences = [ga_ordering]
        else:
            print("Selecting Greedy ordering.")
            self.sequences = [greedy_ordering]

    def planner(self):
        """
        Calculate the full rescue plan by finding paths between victims using BFS.
        """
        bfs = BFS(self.map, self.COST_LINE, self.COST_DIAG)
        if not self.sequences:
            return

        sequence = self.sequences[0]
        start = (0, 0)  # always start at the base
        for vic_id in sequence:
            goal = sequence[vic_id][0]
            plan, time_taken = bfs.search(start, goal, self.plan_rtime)
            self.plan += plan
            self.plan_rtime -= time_taken
            start = goal

        # Plan return to base
        goal = (0, 0)
        plan, time_taken = bfs.search(start, goal, self.plan_rtime)
        self.plan += plan
        self.plan_rtime -= time_taken

    def sync_explorers(self, explorer_map, victims):
        """
        Called only by the master rescuer.
        Receives maps and victim data from explorer agents, updates the map,
        predicts victim severity/class, clusters victims, and assigns clusters
        to rescuers. Then, for each rescuer, it selects the best victim ordering
        (comparing GA vs. greedy) and plans the rescue trajectory.
        """
        self.received_maps += 1

        print(f"{self.NAME} Map received from the explorer")
        self.map.update(explorer_map)
        self.victims.update(victims)

        if self.received_maps == self.nb_of_explorers:
            print(f"{self.NAME} all maps received from the explorers")
            self.predict_severity_and_class()

            # Cluster the victims (using quadrants)
            clusters_of_vic = self.cluster_victims()

            for i, cluster in enumerate(clusters_of_vic):
                self.save_cluster_csv(cluster, i + 1)
  
            # Instantiate other rescuers (assume four agents)
            rescuers = [None] * 4
            rescuers[0] = self  # master rescuer

            # Assign the first cluster to the master
            self.clusters = [clusters_of_vic[0]]

            for i in range(1, 4):
                filename = f"rescuer_{i + 1:1d}_config.txt"
                config_file = os.path.join(self.config_folder, filename)
                # Each rescuer receives one cluster of victims
                rescuers[i] = Rescuer(self.get_env(), config_file, 4, [clusters_of_vic[i]])
                rescuers[i].map = self.map

            # For each rescuer, select the best ordering (GA vs. Greedy) and then plan the route.
            for i, rescuer in enumerate(rescuers):
                rescuer.select_best_ordering()
                
                for j, sequence in enumerate(rescuer.sequences):
                    if j == 0:
                        self.save_sequence_csv(sequence, i + 1)
                    else:
                        self.save_sequence_csv(sequence, (i + 1) + j * 10)

                rescuer.planner()            # plan the trajectory based on the chosen ordering
                rescuer.set_state(VS.ACTIVE)  # mark the rescuer as active for execution
         
    def deliberate(self) -> bool:
        """
        Called in each reasoning cycle.
        Chooses and executes the next action from the plan.
        @return True if there are more actions to execute; False otherwise.
        """
        if self.plan == []:
            print(f"{self.NAME} has finished the plan [ENTER]")
            return False

        dx, dy = self.plan.pop(0)
        walked = self.walk(dx, dy)

        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            if self.map.in_map((self.x, self.y)):
                vic_id = self.map.get_vic_id((self.x, self.y))
                if vic_id != VS.NO_VICTIM:
                    self.first_aid()
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.y})")
            
        return True
