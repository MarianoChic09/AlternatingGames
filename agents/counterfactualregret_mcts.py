import numpy as np
from numpy import ndarray
from base.game import AlternatingGame, AgentID, ObsType
from base.agent import Agent
import logging
import sys
from base.game import AlternatingGame, AgentID, ActionType
from math import log, sqrt
from typing import Callable

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
#, stream=sys.stdout
class MCTSNode:
    def __init__(self, parent: 'MCTSNode', game: AlternatingGame, action: ActionType):
        self.parent = parent
        self.game = game
        self.action = action
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.value = 0
        self.cum_rewards = np.zeros(len(game.agents))
        self.agent = self.game.agent_selection

def ucb(node, C=sqrt(2)) -> float:
    agent_idx = node.game.agent_name_mapping[node.agent]  # Indice del agente del nodo
    
    logging.debug(f'node.cum_rewards: {node.cum_rewards}')
    logging.debug(f'node.visits: {node.visits}')
    logging.debug(f'node.parent.visits: {node.parent.visits}')
    logging.debug(f'agent_idx: {agent_idx}')
    
    return node.cum_rewards[agent_idx] / node.visits + C * sqrt(log(node.parent.visits)/node.visits) # Calculo del Upper Confidence Bound. 
                # Cuando un nodo hijo es poco explorado, empieza a llamar la atención. 

def uct(node: MCTSNode, agent: AgentID) -> MCTSNode: # Criterio de Seleccion. Tree search Policy. Estariamos dentro de la zona conocida del arbol. 
    # Pido el hijo que maximice el UCB. Como elijo los hijos para saber quien jugar.  
    child = max(node.children, key=ucb)
    return child

class Node():

    def __init__(self, game: AlternatingGame, agent: AgentID, obs: ObsType) -> None:
        self.game = game
        self.agent = agent #self.game.agent_selection
        self.obs = obs
        self.num_actions = game.num_actions(agent)
        self.cumulative_regrets = np.zeros(self.num_actions)
        self.strategy_sum = np.zeros(self.num_actions)
        self.learned_policy = np.full(self.num_actions, 1.0 / self.num_actions)  # Uniform random policy
        self.curr_policy = np.full(self.num_actions, 1.0 / self.num_actions)  # Initialize curr_policy as well
        self.sum_policy = np.zeros(self.num_actions)  # Initialize sum_policy as well
        
    def update(self, utility, node_utility, probability) -> None:
        logging.debug(f"Agent index: {self.agent}")
        logging.debug(f"Probability array: {probability}")
        logging.debug(f"utility: {utility}")  
        logging.debug(f"node_utility: {node_utility}")  
            
        agent_index = self.game.agent_name_mapping[self.agent]
        probability_new = probability.copy()
        
        probability_new[agent_index] = 1

        probaility_contrafactual = np.prod(probability_new)
        action_regrets = (utility - node_utility) * probaility_contrafactual#* probability_new
        self.cumulative_regrets = np.maximum(0, self.cumulative_regrets + action_regrets)
        
        self.sum_policy += self.curr_policy * probability[agent_index] 
        self.learned_policy = self.sum_policy / np.sum(self.sum_policy)
        # self.regret_matching()

        # self.cumulative_regrets += action_regrets
    def policy(self):
        return self.learned_policy
    
    def regret_matching(self):
        positive_regrets = np.maximum(0, self.cumulative_regrets)
        sum_positive_regrets = np.sum(positive_regrets)
        
        if sum_positive_regrets > 0:
            self.curr_policy = positive_regrets / sum_positive_regrets
        else:
            self.curr_policy = np.ones(self.num_actions) / self.num_actions
        # self.curr_policy = #self.learned_policy
        logging.debug(f"Updated Policy: {self.learned_policy}")  # Log the updated policy
        logging.debug(f"Updated Strategy: {self.curr_policy}")  # Log the updated strategy
    
    def cumulative_regret(self):
        return np.sum(self.cumulative_regrets)
class CounterFactualRegretMCTS(Agent):

    def __init__(self, game: AlternatingGame, agent: AgentID, max_depth: int=2, rollouts: int=10,selection: Callable[[MCTSNode, AgentID], MCTSNode]=uct) -> None: #
        super().__init__(game, agent)
        self.node_dict: dict[ObsType, Node] = {}
        self.utilities = []
        self.cumulative_regrets_display = []
        self.strategy_evolution = []  # To store the strategy profile after each iteration
        self.max_depth = max_depth
        self.rollouts = rollouts
        self.selection = selection

    def action(self):
        try:
            
            obs = self.game.observe(self.agent)
            # obs_tuple = tuple(obs.flat)
            logging.debug(type(obs))  # Print out the type of the observation
            # logging.debug(obs)        # Print out the observation itself
            logging.debug(obs)
            if obs not in self.node_dict:
                # Create a new Node if it doesn't exist in node_dict
                self.node_dict[obs] = Node(self.game, self.agent, self.game.observe(self.agent))
            node = self.node_dict[obs]
            
            logging.debug(f"Policy: {node.policy()}")

            legal_moves = self.game.available_actions()  # Get legal moves
            legal_move_probabilities = node.policy()[legal_moves]
            legal_move_probabilities /= np.sum(legal_move_probabilities)  # Normalize the probabilities
            logging.debug(f"Legal moves: {legal_moves}")
            logging.debug(f"Legal move probabilities: {legal_move_probabilities}")
            a = legal_moves[np.argmax(np.random.multinomial(1, legal_move_probabilities, size=1))]
            logging.debug(f"Chosen action: {a}")
            return a
        except:
            raise ValueError('Train agent before calling action()')
    
    def train(self, niter=10):
        logging.debug("Training agent {}".format(self.agent))  

        # logging.debug(f"Training agent {self.agent}")
        for i in range(niter):
            self.cfr()

    def calculate_utility(self):
        # Implement logic to calculate utility
        return self.game.reward(self.agent)
    def calculate_strategy(self):
        # Implement logic to extract current strategy
        return {obs: node.strategy_sum / sum(node.strategy_sum) for obs, node in self.node_dict.items() if sum(node.strategy_sum) > 0}
    def cfr(self):
        game = self.game.clone()
        for agent in game.agents:#self.game.agents:
            game.reset() # verificar esto esta raro
            probability = np.ones(game.num_agents)
            logging .debug(f"Entering CFR recursive call, game state: {game.observe(agent)}")
            logging.debug(f"Agent: {agent}")
            # self.utilities[agent] = 
            self.cfr_rec(game=game, agent=agent, probability=probability)

    def cfr_rec(self, game: AlternatingGame, agent: AgentID, probability: ndarray,depth=2):
    
        node_agent = game.agent_selection
        logging.debug(f"Node agent: {node_agent}")
        

        # base cases
        if game.done(): # if h belong to Z then return up(h)
            reward = game.reward(agent)
            return reward
        
        if depth >= self.max_depth:
            # Use MCTS for utility estimation
            mcts_root = MCTSNode(parent=None, game=game, action=None)
            _, estimated_utility = self.mcts(game, simulations=100)

            # Get the current observation and corresponding CFR node
            obs = game.observe(node_agent)
            node = self.node_dict.get(obs, Node(game=game, agent=node_agent, obs=obs))
            self.node_dict[obs] = node  # Ensure node is stored in node_dict
            assumed_probabilities = np.ones(game.num_agents) / game.num_agents
            logging.debug(f"Assumed probabilities: {assumed_probabilities}")
            logging.debug(f"Estimated utility: {estimated_utility}")

            node.update(utility=np.zeros(game.num_actions(node_agent)), 
                        node_utility=estimated_utility, 
                        probability=probability)

            return estimated_utility  # Or further processing depending on your game's nature

        # if self.depth == 0:
        #     return function_evaluacion(self.tipo_funcion, game, agent) # no quiero hacer regret entonces uso una estimacion. La mas rapida es rollout. 
        # # implementar aca montecarlo tree search, rollout(monte carlo), heuristica. 

        # get observation node
        obs = game.observe(node_agent)
        # obs_tuple = tuple(obs.flat)

        try:
            node = self.node_dict[obs]
        except:
            node = Node(game=game, agent=node_agent, obs=obs)
            self.node_dict[obs] = node

        # assert(node_agent == node.agent)

        # compute expected (node) utility
        utility = np.zeros(game.num_actions(node_agent))
        for a in game.action_iter(node_agent):
            # if game.done():
            #     logging.debug(f"Game terminated in CFR Recursive step, returning utility: {game.utility(agent)}")
            #     return game.reward(agent)
            game_clone = game.clone()
            legal_moves = game_clone.available_actions()  # Get legal moves

            logging.debug(f"Legal moves: {legal_moves}")
            if a not in legal_moves:  # Skip illegal moves
                continue
             # update probability vector
            logging.debug(f"Attempting action {a} in game state {obs}")
            
            
            # prob_a = node.policy()[a] # Reveer! Sea P0 = P salvo que P0 [j] = πI [a] ∗ P[j]
            probability_new = probability.copy()
            agent_index = game_clone.agent_name_mapping[node_agent]
            probability_new[agent_index] = probability[agent_index] * node.curr_policy[a] # Pp + Pi
            
            logging.debug(f"Probability array: {probability_new}")
            # play the game
            # game.step(node_agent, a)

            game_clone.step(a)
            # call cfr recursively on updated game with new probability and update node utility

            utility[a] = self.cfr_rec(game_clone, agent, probability_new,depth=depth+1)
        
        node_utility = np.sum(utility * node.curr_policy) # Valor esperado de la probabilidad de la
        # vI = vI + πI [a] ∗ vI→a[a]
        logging.info("Node utility: {}".format(node_utility))

        # update node cumulative regrets using regret matching
        if node_agent == agent:
            logging.debug(f"Node utility: {node_utility}")
            logging.debug(f"Utility: {utility}")
            logging.debug(f"Probability: {probability}")
            
            node.update(utility=utility, node_utility=node_utility, probability=probability)
            node.regret_matching()  

        return node_utility
    
    def get_node(self, agent: AgentID, game: AlternatingGame = None):
        obs = game.observe(agent)
        # obs_tuple = tuple(obs.flat)

        if obs not in self.node_dict:
            self.node_dict[obs] = Node(self.game, agent, obs)
        return self.node_dict[obs],obs
    

    # MCST

    def mcts(self,game,simulations=100) -> (ActionType, float):
        logging.debug('MCTS')
        root = MCTSNode(parent=None, game=game, action=None) # Nodo raiz del arbol de busqueda. primer nodo del arbol.

        for i in range(simulations):

            node = root
            node.game = game.clone()

            logging.debug(f'Simulation {i}')
            node.game.render()

            # selection
            logging.debug('selection')
            node = self.select_node(node=node) # La primera vez  deberias quedar en el nodo raiz. No deberias hacer nada. 
            logging.debug(f'Selected node: {node.action}')
            # expansion
            #print('expansion')
            self.expand_node(node) # si el juego no esta terminado jugar una available action del nodo y crear un nuevo nodo hijo.
            logging.debug(f'Expanded node: {node.action}')
            # Aca podes decir que el nodo sea el nodo nuevo : node = self.expand_node(node)
            # rollout
            #print('rollout')
            rewards = self.rollout(node) # Jugar aleatoriamente y guardar el promedio de las recompensas.
            logging.debug(f'Rollout rewards: {rewards}')
            #update values / Backprop
            #print('backprop')
            self.backprop(node, rewards)

        #print('root childs')
        #for child in root.children:
        #    print(child.action, child.cum_rewards / child.visits)

        action, value = self.action_selection(root)
        logging.debug(f'Action: {action}')
        logging.debug(f'Value: {value}')
        

        return action, value

    def backprop(self, node, rewards): #aca actualizo el self.value de cada nodo. value = node.cum_rewards[agent_idx] / node.visits
        # TODO
        # update node values and visits from node to root navigating backwards through parent
        curr_node = node
        while curr_node is not None: #.parent:
            agent_idx = curr_node.game.agent_name_mapping[curr_node.agent]
            curr_node.cum_rewards[agent_idx] += rewards[agent_idx]
            curr_node.visits += 1
            curr_node.value = curr_node.cum_rewards[agent_idx] / curr_node.visits
            curr_node = curr_node.parent

        # cumulate rewards and visits from node to root navigating backwards through parent
        # pass

    def rollout(self, node):
        rewards = np.zeros(len(self.game.agents))
        # TODO
        # implement rollout policy
        for i in range(self.rollouts): 
            game = node.game.clone()
            while not game.is_terminal():
                available_action = game.available_actions()
                action = np.random.choice(available_action)
                game.step(action)
            
            rewards += [game.reward(agent) for agent in self.game.agents]
        #     play random game and record average rewards
        return rewards

    def select_node(self, node: MCTSNode) -> MCTSNode:
        curr_node = node
        logging.debug(f'curr_node: {curr_node.action}')
        logging.debug(f'curr_node.children: {curr_node.children}')
        while curr_node.children:
            visited_child = [child for child in curr_node.children if child.visits > 0]
            unvisited_child = [child for child in curr_node.children if child not in visited_child]
            
            logging.debug(f'visited_child: {visited_child}')
            logging.debug(f'unvisited_child: {unvisited_child}')
            # if curr_node.explored_children < len(curr_node.children):
            if unvisited_child:
                # TODO
                # set curr_node to an unvisited child
                
                curr_node = np.random.choice(unvisited_child) # mejor de izquierda a derecha.
                logging.debug(f'curr_node: {curr_node}') 
                logging.debug(f'curr_node action: {curr_node.action}')
                
                curr_node.explored_children += 1
                logging.debug(f'curr_node.explored_children: {curr_node.explored_children}')
                
                # pass
            else:
                # TODO
                # set curr_node to a child using the selection function
                curr_node = self.selection(curr_node, self.agent) # utilizo la funcion de seleccion (UCB/UCT) para elegir el siguiente nodo.
                # pass
        return curr_node

    def expand_node(self, node) -> None: # Crear un nuevo nodo en mi arbol simplemente. 
        # TODO
        # if the game is not terminated: 
        if not node.game.terminated():
            available_action = node.game.available_actions()
            action = np.random.choice(available_action)
            node.game.step(action) # decia self.game
            child = MCTSNode(parent=node, game=node.game, action=action)
            node.children.append(child)
            # node = child
            return child
            
        #    play an available action in node
        #    create a new child node and add it to node children
        # pass

    def action_selection(self, node: MCTSNode) -> (ActionType, float): 
        action: ActionType = None
        value: float = 0
        # hint: return action of child with max value 
        # other alternatives could be considered
        
        # pass
        child = max(node.children, key = lambda x: x.value) #value = node.cum_rewards[agent_idx] / node.visits
        action = child.action
        value = child.value
        
        return action, value    

