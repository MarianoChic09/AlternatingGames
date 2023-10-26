import numpy as np
from numpy import ndarray
from base.game import AlternatingGame, AgentID, ObsType
from base.agent import Agent

class Node():

    def __init__(self, game: AlternatingGame, agent: AgentID, obs: ObsType) -> None:
        self.game = game
        self.agent = agent
        self.obs = obs

    def update(self, utility, node_utility, probability) -> None:
        # TODO
        pass 

    def policy(self):
        return self.learned_policy

class CounterFactualRegret(Agent):

    def __init__(self, game: AlternatingGame, agent: AgentID) -> None:
        super().__init__(game, agent)
        self.node_dict: dict[ObsType, Node] = {}

    def action(self):
        try:
            node = self.node_dict[self.game.observe(self.agent)]
            a = np.argmax(np.random.multinomial(1, node.policy(), size=1))
            return a
        except:
            raise ValueError('Train agent before calling action()')
    
    def train(self, niter=1000):
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

        # base cases
        # TODO

        # recursive call

        # get observation node
        obs = game.observe(node_agent)
        try:
            node = self.node_dict[obs]
        except:
            node = Node(game=game, agent=node_agent, obs=obs)
            self.node_dict[obs] = node

        assert(node_agent == node.agent)

        # compute expected (node) utility
        utility = np.zeros(game.num_actions(node_agent))
        for a in game.action_iter(node_agent):
            # update probability vector
            # play the game
            # call cfr recursively on updated game with new probability and update node utility
            # TODO
            pass

        node_utility = np.sum(utility * node.curr_policy)

        # update node cumulative regrets using regret matching
        if node_agent == agent:
            node.update(utility=utility, node_utility=node_utility, probability=probability)

        return node_utility
        
