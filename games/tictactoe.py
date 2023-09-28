from base.game import AgentID, ObsType
from numpy import ndarray
from gymnasium.spaces import Discrete, Text, Dict, Tuple
from pettingzoo.utils import agent_selector
from pettingzoo.classic import tictactoe_v3 as tictactoe
from base.game import AlternatingGame, AgentID
import numpy as np

import warnings
warnings.filterwarnings("ignore")

class TicTacToe(AlternatingGame):

    def __init__(self, render_mode=''):
        super().__init__()
        self.env = tictactoe.raw_env(render_mode=render_mode)
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces
        self.action_space = self.env.action_space
        self.agents = self.env.agents
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))

    def _update(self):
        self.rewards = self.env.rewards
        self.terminations = self.env.terminations
        self.truncations = self.env.truncations
        self.infos = self.env.infos
        self.agent_selection = self.env.agent_selection

    def reset(self):
        self.env.reset()
        self._update()

    def observe(self, agent: AgentID) -> ObsType:
        # A grid is list of lists, where each list represents a row
        # blank space = 0
        # agent = 1
        # opponent = 2
        # Ex:
        # [[0,0,2]
        #  [1,2,1]
        #  [2,1,0]]
        observation = self.env.observe(agent=agent)['observation']
        grid = np.sum(observation*[1,2], axis=2)
        return grid

    def step(self, action):
        self.env.step(action)
        self._update()

    def available_actions(self):
        return self.env._legal_moves()

    def eval(self, agent: AgentID) -> float:
        grid = self.observe(agent=agent)
        evalp = np.zeros(2)
        for p in range(2):
            op = (p + 1) % 2 + 1
            ev = 3 - np.sum(list(map(lambda j: np.any(grid[j,:] == op), range(3))))
            ev += 3 - np.sum(list(map(lambda j: np.any(grid[:,j] == op), range(3))))
            ev += 1 - np.sum(list(map(lambda j: np.any(grid[j,j] == op), range(3))))
            ev += 1 - np.sum(list(map(lambda j: np.any(grid[j,2-j] == op), range(3))))
            evalp[p] = ev
        return (evalp[0] - evalp[1]) / 8.



    
    
