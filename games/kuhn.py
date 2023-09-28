import numpy as np
from numpy import ndarray
from gymnasium.spaces import Discrete, Text, Dict, Tuple
from pettingzoo.utils import agent_selector
from base.game import AlternatingGame, AgentID, ActionType

class KuhnPoker(AlternatingGame):

    def __init__(self, render_mode='human'):
        self.render_mode = render_mode

        # agents
        self.agents = ["agent_" + str(r) for r in range(2)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))

        # actions
        self._moves = ['p', 'b']
        self._num_actions = 2
        self.action_spaces = {
            agent: Discrete(self._num_actions) for agent in self.agents
        }

        # states
        self._max_moves = 3
        self._start = ''
        self._terminalset = set(['pp', 'pbp', 'pbb', 'bp', 'bb'])
        self._hist_space = Text(min_length=0, max_length=self._max_moves, charset=frozenset(self._moves))
        self._hist = None
        self._card_names = ['J', 'Q', 'K']
        self._num_cards = len(self._card_names)
        self._cards = list(range(self._num_cards))
        self._card_space = Discrete(self._num_cards)
        self._hand = None

        # observations
        self.observation_spaces = {
            agent: Dict({ 'card': self._card_space, 'hist': self._hist_space}) for agent in self.agents
        }
    
    def step(self, action: ActionType) -> None:
        agent = self.agent_selection
        # check for termination
        if (self.terminations[agent] or self.truncations[agent]):
            try:
                self._was_dead_step(action)
            except ValueError:
                print('Game has already finished - Call reset if you want to play again')
                return

        # perform step
        self._hist += self._moves[action]
        self.agent_selection = self._agent_selector.next()

        if self._hist in self._terminalset:
            # game over - compute rewards
            if self._hist == 'pp':                  
                # pass pass
                _rewards = list(map(lambda p: 1 if p == np.argmax(self._hand) else -1, range(self.num_agents))) 
            elif self._hist == 'pbp':               
                # pass bet pass
                _rewards = list(map(lambda p: 1 if p == 1 else -1, range(self.num_agents)))
            elif self._hist == 'bp':                
                # bet pass
                _rewards = list(map(lambda p: 1 if p == 0 else -1, range(self.num_agents))) 
            else:                                   
                # pass bet bet OR bet bet
                _rewards = list(map(lambda p: 2 if p == np.argmax(self._hand) else -2, range(self.num_agents)))              
        
            self.rewards = dict(map(lambda p: (p, _rewards[self.agent_name_mapping[p]]), self.agents))
            self.terminations = dict(map(lambda p: (p, True), self.agents))

    def _set_initial(self, seed=None):
        # set initial history
        self._hist = self._start

        # deal a card to each player
        np.random.seed(seed)
        self._hand = np.random.choice(self._cards, size=self.num_agents, replace=False)      

        # reset agent selection
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        
    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self._set_initial(seed=seed)

        self.rewards = dict(map(lambda agent: (agent, None), self.agents))
        self.terminations = dict(map(lambda agent: (agent, False), self.agents))
        self.truncations = dict(map(lambda agent: (agent, False), self.agents))
        self.infos = dict(map(lambda agent: (agent, {}), self.agents))

    def render(self) -> ndarray | str | list | None:
        for agent in self.agents:
            print(agent, self._card_names[self._hand[self.agent_name_mapping[agent]]], self._hist)

    def observe(self, agent: AgentID) -> str:
        observation = str(self._hand[self.agent_name_mapping[agent]]) + self._hist
        return observation
    
    def available_actions(self):
        return list(range(self._num_actions))

