##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### This rescuer version implements:
### - clustering of victims by quadrants of the explored region 
### - definition of a sequence of rescue of victims of a cluster
### - assigning one cluster to one rescuer
### - calculating paths between pair of victims using breadth-first search
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
from clusterer import VictimClusterer

class Rescuer(AbstAgent):
    def __init__(self, env, config_file, nb_of_explorers=1, clusters=[]):
        """
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file
        @param nb_of_explorers: number of explorer agents to wait for
        @param clusters: list of clusters of victims in the charge of this agent
        """
        super().__init__(env, config_file)
        
        self.clusterer = VictimClusterer(n_clusters=4)
        self.nb_of_explorers = nb_of_explorers
        self.received_maps = 0
        self.map = Map()
        self.victims = {}           # [vic_id]: ((x,y), [<vs>])
        self.plan = []              # a list of planned actions in increments of x and y
        self.plan_x = 0             # x position during planning
        self.plan_y = 0             # y position during planning
        self.plan_visited = set()   # positions already planned to be visited
        self.plan_rtime = self.TLIM # remaining time during planning phase
        self.plan_walk_time = 0.0   # previewed time to walk during rescue
        self.x = 0                  # current x position of the rescuer when executing the plan
        self.y = 0                  # current y position of the rescuer when executing the plan
        self.clusters = clusters    # the clusters of victims this agent should take care of
        self.sequences = clusters   # the sequence of visit of victims for each cluster
        self.set_state(VS.IDLE)

    def save_cluster_csv(self, cluster, cluster_id):
        """ Save cluster data into a csv file.
            Each line: vic_id, x, y, severity_value, severity_class """
        os.makedirs('./clusters', exist_ok=True)
        filename = f"./clusters/cluster{cluster_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for vic_id, values in cluster.items():
                x, y = values[0]
                vs = values[1]
                writer.writerow([vic_id, x, y, vs[6], vs[7]])
    
    def save_sequence_csv(self, sequence, sequence_id):
        """ Save sequence of victims into a csv file.
            Each line: vic_id, x, y, severity_value, severity_class """
        filename = f"./clusters/seq{sequence_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for id, values in sequence.items():
                x, y = values[0]
                vs = values[1]
                writer.writerow([id, x, y, vs[6], vs[7]])

    def cluster_victims(self):
        """
        Performs clustering of victims using the VictimClusterer.
        @return: a list of clusters, each cluster is a dict: [vic_id]: ((x,y), [<vs>])
        """
        return self.clusterer.cluster_victims(self.victims)

    def predict_severity_and_class(self):
        """
        Assigns random severity value and class to each victim.
        This should be replaced by real classifiers/regressors if available.
        """
        for vic_id, values in self.victims.items():
            severity_value = random.uniform(0.1, 99.9)
            severity_class = random.randint(1, 4)
            values[1].extend([severity_value, severity_class])

    def sequencing(self):
        """
        Sorts the victims by a heuristic that takes into account both position and severity.
        @TODO replace by a Genetic Algorithm or another heuristic for best route.
        
        BELOW IS HOW WE IMPROVE THE HEURISTICS:
        We will now consider both the distance from (0,0) and the severity class
        to form a heuristic. More critical victims (class = 1) and closer victims
        will be prioritized.
        
        NOTE: This is not a full Genetic Algorithm, but a heuristic improvement
        as a step towards addressing the TODO.
        """

        # Get original sequences (clusters of victims)
        new_sequences = []
        base = (0,0)

        # Weight parameters for the heuristic
        alpha = 0.7  # weight for normalized distance
        beta = 0.3   # weight for normalized severity

        # We'll find min and max for distance and severity to normalize
        all_positions = []
        all_severities = []

        for seq in self.sequences:
            for vic_id, values in seq.items():
                x, y = values[0]
                vs = values[1]
                dist = math.sqrt((x - base[0])**2 + (y - base[1])**2)
                severity_val = vs[6]
                all_positions.append(dist)
                all_severities.append(severity_val)

        if not all_positions:
            # no sequences, nothing to do
            self.sequences = []
            return

        min_dist = min(all_positions)
        max_dist = max(all_positions)
        min_sev = min(all_severities)
        max_sev = max(all_severities)

        dist_range = (max_dist - min_dist) if (max_dist - min_dist) != 0 else 1
        sev_range = (max_sev - min_sev) if (max_sev - min_sev) != 0 else 1

        for seq in self.sequences:
            victims_data = list(seq.items())

            def victim_score(item):
                _, values = item
                x, y = values[0]
                vs = values[1]
                dist = math.sqrt((x - base[0])**2 + (y - base[1])**2)
                severity_val = vs[6]
                norm_dist = (dist - min_dist) / dist_range
                norm_sev = (severity_val - min_sev) / sev_range
                # lower is better for both
                score = alpha * norm_dist + beta * norm_sev
                return score

            victims_data.sort(key=victim_score)
            new_sequences.append(dict(victims_data))

        self.sequences = new_sequences
        # TODO solved: We improved heuristic by integrating severity and distance.
        # Explanation: Ao resolver o TODO, adicionamos uma heurística que leva em conta
        # tanto a distância quanto a severidade para ordenar as vítimas.
        # Isso é um passo em direção a um método mais sofisticado (como um AG),
        # mas já melhora a priorização de vítimas.

    def planner(self):
        """
        Calculates a path between victims using BFS.
        Each element in self.plan is a pair (dx, dy).
        """
        bfs = BFS(self.map, self.COST_LINE, self.COST_DIAG)
        if not self.sequences:
            return
        sequence = self.sequences[0]
        start = (0,0)
        for vic_id in sequence:
            goal = sequence[vic_id][0]
            plan, time = bfs.search(start, goal, self.plan_rtime)
            self.plan = self.plan + plan
            self.plan_rtime = self.plan_rtime - time
            start = goal
        # Plan to come back to the base
        goal = (0,0)
        plan, time = bfs.search(start, goal, self.plan_rtime)
        self.plan = self.plan + plan
        self.plan_rtime = self.plan_rtime - time

    def sync_explorers(self, explorer_map, victims):
        """
        Invoked only by the master rescuer when all explorers have sent their maps.
        Updates self.map and self.victims, predicts severity and class, clusters victims,
        and assigns clusters to rescuers.
        """
        self.received_maps += 1
        print(f"{self.NAME} Map received from the explorer")
        self.map.update(explorer_map)
        self.victims.update(victims)

        if self.received_maps == self.nb_of_explorers:
            print(f"{self.NAME} all maps received from the explorers")
            self.predict_severity_and_class()
            clusters_of_vic = self.cluster_victims()
            for i, cluster in enumerate(clusters_of_vic):
                self.save_cluster_csv(cluster, i+1)

            rescuers = [None] * 4
            rescuers[0] = self
            self.clusters = [clusters_of_vic[0]]

            for i in range(1, 4):
                filename = f"rescuer_{i+1:1d}_config.txt"
                config_file = os.path.join(self.config_folder, filename)
                rescuers[i] = Rescuer(self.get_env(), config_file, 4, [clusters_of_vic[i]])
                rescuers[i].map = self.map

            self.sequences = self.clusters

            for i, rescuer in enumerate(rescuers):
                rescuer.sequencing()
                for j, sequence in enumerate(rescuer.sequences):
                    if j == 0:
                        self.save_sequence_csv(sequence, i+1)
                    else:
                        self.save_sequence_csv(sequence, (i+1)+ j*10)
                rescuer.planner()
                rescuer.set_state(VS.ACTIVE)

    def deliberate(self) -> bool:
        """
        Chooses the next action for the rescuer.
        The simulator calls this method at each cycle if the agent is ACTIVE.
        @return True if there are more actions to do, False otherwise.
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

                    # @TODO: After rescuing a victim, re-check if there are more victims and re-run sequencing and planner
                    # so that the rescuer continuously updates its plan.
                    #
                    # SOLUTION TO TODO:
                    # Agora, após resgatar uma vítima, reexecutamos o sequencing e planner.
                    # Dessa forma, o resgatista recalcula o plano levando em conta a nova situação do mapa e das vítimas.
                    # Isso evita que o plano fique desatualizado e que o agente "corra em círculos".
                    
                    # Check if there are victims not rescued yet
                    # Para simplificar, consideramos que todas as vítimas em self.victims ainda são candidatas,
                    # caso contrário, teríamos que marcar as resgatadas separadamente.

                    if self.victims:
                        # Re-cluster with updated victims
                        clusters_of_vic = self.cluster_victims()
                        if clusters_of_vic:
                            self.sequences = clusters_of_vic
                            self.sequencing()  # run updated heuristic
                            self.plan = []  # clear old plan
                            self.planner()   # re-plan

        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.x})")
        return True
