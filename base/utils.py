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

def play_game(game: AlternatingGame, agents: dict[AgentID, Agent]):
    game.reset()
    turn_count = 0

    while not game.terminated():
        current_agent = game.agent_selection
        action = agents[current_agent].action()
        if action not in game.available_actions():
            break
        game.step(action)
        turn_count += 1

    # Gather final game statistics
    winner = max(game.rewards, key=game.rewards.get)  # Agent with the highest reward
    final_rewards = game.rewards
    num_turns = turn_count

    return winner, final_rewards, num_turns


