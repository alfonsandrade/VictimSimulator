# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.

import sys
import os
import random
import math
from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

class Explorer(AbstAgent):
    """ class attribute """
    MAX_DIFFICULTY = 3             # the maximum degree of difficulty to enter into a cell        
    
    def __init__(self, env, config_file, resc):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)

        self.unbackpositions = Stack()   # a stack to store the movements to return to the origin

        # {(0, 0): [2, 0, 3, 7, 4, 6, 5], (1, -1): [1, 2, 0, 3, 7, 4, 6, 5]}      -> pos: untried_Movements
        self.untried_move_by_pos = {}   # Retorna os movimentos ainda nao tentados para determinada posição

        # {(0,0), (1,1), (2,1), (3,2)}                   -> pilha de posições até a origem
        self.come_back_path = Stack()   # a stack to store the movements to return to the origin

        self.is_unbacktracking = False  # Flag para indicar se o robo esta em movimento de backtracking
        self.flag_explore = True        # Flag para indicar se esta em exploraçao ou come_back
        self.last_position = None       # Armazena a ultima posição do robo

        self.reverse_map = {movement: key for key, movement in Explorer.AC_INCR.items()}

        self.walk_time = 0              # time consumed to walk when exploring (to decide when to come back)
        self.set_state(VS.ACTIVE)       # explorer is active since the begin
        self.resc = resc                # reference to the rescuer agent
        self.x = 0                      # current x position relative to the origin 0
        self.y = 0                      # current y position relative to the origin 0
        self.map = Map()                # create a map for representing the environment
        self.victims = {}               # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                        # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())


    # 0: (0, -1),  #  u: Up
    # 1: (1, -1),  # ur: Upper right diagonal
    # 2: (1, 0),   #  r: Right
    # 3: (1, 1),   # dr: Down right diagonal
    # 4: (0, 1),   #  d: Down
    # 5: (-1, 1),  # dl: Down left left diagonal
    # 6: (-1, 0),  #  l: Left
    # 7: (-1, -1)  # ul: Up left diagonal        
    def actions(self, position):
        
        orderToCorner = {
            "EXPL_1": [0, 2, 4, 1, 3, 7, 6, 5], # Corner Up Right 
            "EXPL_2": [4, 2, 0, 3, 1, 5, 6, 7], # Corner down Right 
            "EXPL_3": [4, 6, 0, 5, 1, 3, 2, 7], # Corner Down Left 
            "EXPL_4": [0, 6, 4, 1, 5, 7, 2, 3], # Corner Up Left 
        }

        # Obtem a ordem baseada no nome do robô
        order = orderToCorner.get(self.NAME, list(self.AC_INCR.keys()))
        
        # Retorna todas as ações, já ordenadas pela sequência do robô
        return order


    def manhattan_distance(self, position):
        return abs(position[0])+abs(position[1])*1.5*2


    # Função para realizar a exploração do mapa
    def online_dfs(self, position):

        # Verifica se é uma nova posição e ordena seus movimentos
        if position not in self.untried_move_by_pos:
            self.untried_move_by_pos[position] = self.actions(position)

        # Checa se já realizou todas as ações possiveis para o estado atual
        if len(self.untried_move_by_pos[position]) == 0:

            # ERRO
            if self.unbackpositions.is_empty():
                print('Error: UnpackPositions is empty')
                return (0,0)
            
            # Realizando o backtrack
            else:
                self.is_unbacktracking = True
                
                # Pegando a posicao para unbacktrack
                unback_pos = self.unbackpositions.pop()
                while unback_pos == position:
                    unback_pos = self.unbackpositions.pop()
           
                # Obtendo o movimento para voltar a essa posicao
                movement = self.reverse_map.get( (unback_pos[0] - position[0], unback_pos[1] - position[1]) )

        # Caso ainda tenha ações possiveis para o estado atual
        else:
            self.is_unbacktracking = False
            movement = self.untried_move_by_pos[position][0]
            del self.untried_move_by_pos[position][0]

        # Checa se moveu-se para um novo estado, se sim e não esta em backtracking, adiciona o last_state para backtracking
        if self.last_position != None and self.last_position != position and not self.is_unbacktracking:
            self.unbackpositions.push(position)
                
        self.last_position = position
        return movement

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        return total_path   


    def return_neighbors(self, position):
        neighbors = []
        for i in range(-1,2):
            for j in range(-1,2):
                neighbor = position[0]+i, position[1]+j
                if self.map.in_map(neighbor) and neighbor != position:
                    neighbors.append(neighbor)
        return neighbors
    

    def a_star(self, cur_Position, goal):
        open_set = {cur_Position}
        came_from = {}
        g_score = {}
        f_score = {}
        
        for e in self.map.data:
            g_score[e] = math.inf
            f_score[e] = math.inf
        
        g_score[cur_Position] = 0
        f_score[cur_Position] = self.manhattan_distance(cur_Position)

        while len(open_set) != 0:
            value = math.inf
            for e in open_set:
                if f_score[e] < value:
                    current = e
                    value = f_score[e]

            if current == goal:
                return self.reconstruct_path(came_from, current)
            #print(f'open set: {open_set}')
            open_set.remove(current)

            for neighbor in self.return_neighbors(current):
                
                tentative_gscore = g_score[current] + self.map.get_difficulty(neighbor)
                if tentative_gscore < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_gscore
                    f_score[neighbor] = tentative_gscore + self.manhattan_distance(neighbor)
                    if neighbor not in open_set:
                        open_set.add(neighbor)
        print('Erro!')
        return EOFError 
    

    def get_next_position(self):
        # Check the neighborhood walls and grid limits
        obstacles = self.check_walls_and_lim()

        # Loop until a CLEAR position is found
        while True:
                    
            movement = self.online_dfs((self.x, self.y))
            dx, dy = Explorer.AC_INCR[movement]
            position = (self.x + dx, self.y + dy)
            # Check if the corresponding position in walls_and_lim is CLEAR
            if obstacles[movement] == VS.CLEAR:
                if not self.map.in_map(position) or self.is_unbacktracking:
                    return Explorer.AC_INCR[movement]
        
        
    def explore(self):
        # get an random increment for x and y       
        dx, dy = self.get_next_position()

        # Moves the body to another position  
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()


        # Test the result of the walk action
        # Should never bump, but for safe functionning let's test
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            #print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

        if result == VS.EXECUTED:
            # check for victim returns -1 if there is no victim or the sequential
            # the sequential number of a found victim

            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            
            # update the walk time
            self.walk_time = self.walk_time + (rtime_bef - rtime_aft)
            #print(f"{self.NAME} walk time: {self.walk_time}")

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                #print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                #print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")
            # Calculates the difficulty of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG
            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
            #print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

        return

    def come_back(self):

        # Find the best path until the origem position (0,0)
        if self.flag_explore == True:
            self.come_back_path = self.a_star((self.x, self.y), (0,0))
            self.flag_explore = False

        # Obtaining dx and dy movements to next position
        if len(self.come_back_path) > 1:
             dx, dy = self.come_back_path[-2][0] - self.come_back_path[-1][0], self.come_back_path[-2][1] - self.come_back_path[-1][1]
             self.come_back_path.pop()
        else:
            dx, dy = self.come_back_path.pop()

        # Showing Results
        result = self.walk(dx, dy)
        if result == VS.BUMPED:
            print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
        
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
        
    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        # Time to go, read and comeback
        time_tolerance = 2* self.COST_DIAG * Explorer.MAX_DIFFICULTY + self.COST_READ

        # keeps exploring while there is enough time
        if ( self.get_rtime() > (self.manhattan_distance((self.x, self.y)) + time_tolerance) ) and self.flag_explore:
            self.explore()
            return True

        # no more come back walk actions to execute or already at base
        if (self.x == 0 and self.y == 0):
            # time to pass the map and found victims to the master rescuer
            self.resc.sync_explorers(self.map, self.victims)
            # finishes the execution of this agent
            return False
        
        # proceed to the base
        self.come_back()
        return True

