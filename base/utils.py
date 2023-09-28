from base.game import AlternatingGame
from base.agent import Agent, AgentID
import numpy as np

def play(game: AlternatingGame, agents: dict[AgentID, Agent]):
    game.reset()
    game.render()
    while not game.terminated():
        action = agents[game.agent_selection].action()
        game.step(action)
    game.render()
    print(game.rewards)

def run(game: AlternatingGame, agents: dict[AgentID, Agent], N=100):
    values = []
    for i in range(N):    
        game.reset()
        while not game.terminated():
            action = agents[game.agent_selection].action()
            game.step(action)
        values.append(game.reward(game.agents[0]))
    v, c = np.unique(values, return_counts=True)
    return dict(zip(v, c)), np.mean(values)
