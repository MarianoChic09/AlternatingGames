import numpy as np
from numpy import ndarray
from base.game import AlternatingGame, AgentID, ObsType
from base.agent import Agent
import logging
import sys
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')

class Node():

    def __init__(self, game: AlternatingGame, agent: AgentID, obs: ObsType) -> None:
        self.game = game
        self.agent = agent
        self.obs = obs
        self.num_actions = game.num_actions(agent)
        self.cumulative_regrets = np.zeros(self.num_actions)
        self.strategy_sum = np.zeros(self.num_actions)
        self.learned_policy = np.full(self.num_actions, 1.0 / self.num_actions)  # Uniform random policy
        self.curr_policy = np.full(self.num_actions, 1.0 / self.num_actions)  # Initialize curr_policy as well


    def update(self, utility, node_utility, probability) -> None:
        logging.debug(f"Agent index: {self.agent}")
        logging.debug(f"Probability array: {probability}")
        logging.debug(f"utility: {utility}")  
        logging.debug(f"node_utility: {node_utility}")  
            
        agent_index = self.game.agent_name_mapping[self.agent]

        action_regrets = (utility - node_utility) / probability[agent_index]
        self.cumulative_regrets = np.maximum(0, self.cumulative_regrets + action_regrets)

    def policy(self):
        return self.learned_policy
    
    def update_strategy(self):
        positive_regrets = np.maximum(0, self.cumulative_regrets)
        sum_positive_regrets = np.sum(positive_regrets)
        if sum_positive_regrets > 0:
            self.learned_policy = positive_regrets / sum_positive_regrets
        else:
            self.learned_policy = np.ones(self.num_actions) / self.num_actions
        self.curr_policy = self.learned_policy
        logging.debug(f"Updated Policy: {self.learned_policy}")  # Log the updated policy
        logging.debug(f"Updated Strategy: {self.curr_policy}")  # Log the updated strategy

class CounterFactualRegret(Agent):

    def __init__(self, game: AlternatingGame, agent: AgentID) -> None:
        super().__init__(game, agent)
        self.node_dict: dict[ObsType, Node] = {}

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
    
    def train(self, niter=1000):
        logging.debug("Training agent {}".format(self.agent))  

        # logging.debug(f"Training agent {self.agent}")
        for _ in range(niter):
            self.cfr()

    def cfr(self):
        game = self.game.clone()
        for agent in self.game.agents:
            game.reset()
            probability = np.ones(game.num_agents)
            self.cfr_rec(game=game, agent=agent, probability=probability)

    def cfr_rec(self, game: AlternatingGame, agent: AgentID, probability: ndarray):
    
        node_agent = game.agent_selection
        logging.debug(f"Node agent: {node_agent}")
        

        # base cases
        if game.is_terminal():
            return game.utility(agent)
        # if node_agent != agent:
        #     return self.eval_node(game, node_agent, probability)
        # recursive call

        # get observation node
        obs = game.observe(node_agent)
        # obs_tuple = tuple(obs.flat)

        try:
            node = self.node_dict[obs]
        except:
            node = Node(game=game, agent=node_agent, obs=obs)
            self.node_dict[obs] = node

        assert(node_agent == node.agent)

        # compute expected (node) utility
        utility = np.zeros(game.num_actions(node_agent))
        for a in game.action_iter(node_agent):
            if game.done():
                logging.debug(f"Game terminated in CFR Recursive step, returning utility: {game.utility(agent)}")
                return game.reward(agent)
            game_clone = game.clone()
            legal_moves = game_clone.available_actions()  # Get legal moves

            logging.debug(f"Legal moves: {legal_moves}")
            if a not in legal_moves:  # Skip illegal moves
                continue
             # update probability vector
            logging.debug(f"Attempting action {a} in game state {obs}")
            prob_a = node.policy()[a]
            
            
            probability_new = probability.copy()
            agent_index = self.game.agent_name_mapping[node_agent]
            probability_new[agent_index] = prob_a

            # play the game
            # game.step(node_agent, a)

            logging.debug(f"Legal moves: {game_clone.available_actions()}")
            game_clone.step(a)
            # call cfr recursively on updated game with new probability and update node utility
             
            node, obs = self.get_node(node_agent,game_clone)
            logging.debug(f"Before recursive call, game state: {game_clone}, attempting action: {a}")


            utility[a] = self.cfr_rec(game_clone, agent, probability_new)

            
            node, obs = self.get_node(node_agent,game_clone)        
            logging.debug(f"After recursive call, game state: {obs}")

        node_utility = np.sum(utility * node.curr_policy)

        # update node cumulative regrets using regret matching
        if node_agent == agent:
            logging.debug(f"Node utility: {node_utility}")
            logging.debug(f"Utility: {utility}")
            logging.debug(f"Probability: {probability}")
            
            node.update(utility=utility, node_utility=node_utility, probability=probability)
            node.update_strategy()  

        return node_utility
        
    # def eval_node(self, game: AlternatingGame, node_agent: AgentID, probability: ndarray) -> float:
    #     game = game.clone()

    #     node,obs = self.get_node(node_agent,game)
    #     logging.debug(f"Entering eval_node, game state: {obs}")


    #     expected_utility = 0.0
    #     for a in game.action_iter(node_agent):
    #         legal_moves = game.available_actions()  # Get legal moves
    #         logging.debug(f"Legal moves: {legal_moves}")
    #         if a not in legal_moves:  # Skip illegal moves
    #             continue
    #         logging.debug(f"Attempting action {a} in game state {obs}")


    #         prob_a = node.policy()[a]
    #         probability_new = probability.copy()
    #         agent_index = game.agent_name_mapping[node_agent]
    #         probability_new[agent_index] = prob_a

    #         game.step(a)
    #         expected_utility += prob_a * self.cfr_rec(game, node_agent, probability_new)
    #         node, obs = self.get_node(node_agent,game)
    #         logging.debug(f"Before undo, game state: {obs}")

    #         # game.undo()
            
    #         node, obs = self.get_node(node_agent,game)
    #         logging.debug(f"After undo, game state: {obs}")
        
    #     node, obs = self.get_node(node_agent,game)
    #     logging.debug(f"Exiting eval_node, game state: {game}")

    #     return expected_utility
    
    def get_node(self, agent: AgentID, game: AlternatingGame = None):
        obs = game.observe(agent)
        # obs_tuple = tuple(obs.flat)

        if obs not in self.node_dict:
            self.node_dict[obs] = Node(self.game, agent, obs)
        return self.node_dict[obs],obs
