from base.game import AlternatingGame, AgentID, ActionType
from base.agent import Agent
from math import log, sqrt
import numpy as np
from typing import Callable

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
    return node.cum_rewards[agent_idx] / node.visits + C * sqrt(log(node.parent.visits)/node.visits) # Calculo del Upper Confidence Bound. 
                # Cuando un nodo hijo es poco explorado, empieza a llamar la atención. 

def uct(node: MCTSNode, agent: AgentID) -> MCTSNode: # Criterio de Seleccion. Tree search Policy. Estariamos dentro de la zona conocida del arbol. 
    # Pido el hijo que maximice el UCB. Como elijo los hijos para saber quien jugar.  
    child = max(node.children, key=ucb)
    return child

class MonteCarloTreeSearch(Agent):
    def __init__(self, game: AlternatingGame, agent: AgentID, simulations: int=100, rollouts: int=10, selection: Callable[[MCTSNode, AgentID], MCTSNode]=uct) -> None:
        """
        Parameters:
            game: alternating game associated with the agent
            agent: agent id of the agent in the game
            simulations: number of MCTS simulations (default: 100)
            rollouts: number of MC rollouts (default: 10)
            selection: tree search policy (default: uct)
        """
        super().__init__(game=game, agent=agent)
        self.simulations = simulations
        self.rollouts = rollouts
        self.selection = selection
        
    def action(self) -> ActionType:
        a, _ = self.mcts()
        return a # Devuelve una acción luego de haber hecho el MCTS

    def mcts(self) -> (ActionType, float):

        root = MCTSNode(parent=None, game=self.game, action=None) # Nodo raiz del arbol de busqueda. primer nodo del arbol.

        for i in range(self.simulations):

            node = root
            node.game = self.game.clone()

            #print(i)
            #node.game.render()

            # selection
            #print('selection')
            node = self.select_node(node=node) # La primera vez  deberias quedar en el nodo raiz. No deberias hacer nada. 

            # expansion
            #print('expansion')
            self.expand_node(node) # si el juego no esta terminado jugar una available action del nodo y crear un nuevo nodo hijo.
            
            # Aca podes decir que el nodo sea el nodo nuevo : node = self.expand_node(node)
            # rollout
            #print('rollout')
            rewards = self.rollout(node) # Jugar aleatoriamente y guardar el promedio de las recompensas.

            #update values / Backprop
            #print('backprop')
            self.backprop(node, rewards)

        #print('root childs')
        #for child in root.children:
        #    print(child.action, child.cum_rewards / child.visits)

        action, value = self.action_selection(root)

        return action, value

    def backprop(self, node, rewards): #aca actualizo el self.value de cada nodo. value = node.cum_rewards[agent_idx] / node.visits
        # TODO
        # update node values and visits from node to root navigating backwards through parent
        curr_node = node
        while curr_node.parent:
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
            while not game.done():
                available_action = game.available_actions()
                action = np.random.choice(available_action)
                game.step(action)
            rewards += game.rewards()
        #     play random game and record average rewards
        return rewards

    def select_node(self, node: MCTSNode) -> MCTSNode:
        curr_node = node
        while curr_node.children:
            if curr_node.explored_children < len(curr_node.children):
                # TODO
                # set curr_node to an unvisited child
                visited_child = [child for child in curr_node.children if child.visits > 0]
                unvisited_child = [child for child in curr_node.children if child not in visited_child]
                
                curr_node = np.random.choice(unvisited_child) # mejor de izquierda a derecha. 
                curr_node.explored_children += 1
                pass
            else:
                # TODO
                # set curr_node to a child using the selection function
                curr_node = self.selection(curr_node, self.agent) # utilizo la funcion de seleccion (UCB/UCT) para elegir el siguiente nodo.
                pass
        return curr_node

    def expand_node(self, node) -> None: # Crear un nuevo nodo en mi arbol simplemente. 
        # TODO
        # if the game is not terminated: 
        if not node.game.done():
            available_action = self.game.available_actions()
            action = np.random.choice(available_action)
            self.game.step(action)
            child = MCTSNode(parent=node, game=self.game, action=action)
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