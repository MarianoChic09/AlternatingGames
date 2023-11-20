import numpy as np
from numpy import ndarray
from base.game import AlternatingGame, AgentID, ObsType
from base.agent import Agent
from base.ann import RegretNetwork,StrategyNetwork
import torch
from torch import nn
import logging
import sys
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

import random
import numpy as np
import torch.optim as optim

class DeepCFR:
    def __init__(self, strategy_net, advantage_nets, num_actions):
        self.strategy_net = strategy_net
        self.advantage_nets = advantage_nets
        self.strategy_memory = []  # Memory for storing strategy-related data
        self.advantage_memory = {p: [] for p in advantage_nets}  # Memory for storing advantage-related data
        self.num_actions = num_actions

    def train(self, env, num_iterations, batch_size, learning_rate):
        # Training loop
        for _ in range(num_iterations):
            self._run_cfr_iteration(env)
            self._train_networks(batch_size, learning_rate)

    def _run_cfr_iteration(self, env):
        # Run external sampling CFR for each player
        for player in range(env.num_players):
            self._external_sampling(env, player)        # ...

    def _train_networks(self, batch_size, learning_rate):
        # Implement training logic for neural networks
        # ...
        strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=learning_rate)
        for player, advantage_net in self.advantage_nets.items():
            advantage_optimizer = optim.Adam(advantage_net.parameters(), lr=learning_rate)
            
    def _train_networks(self, batch_size, learning_rate):
    # Train strategy network
    strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=learning_rate)
    for _ in range(self.n_training_steps_Π):
        # Sample batch from strategy memory
        batch = random.sample(self.strategy_memory, min(len(self.strategy_memory), batch_size))
        states, strategies = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        strategies = torch.tensor(strategies, dtype=torch.float32)

        # Forward pass
        strategy_optimizer.zero_grad()
        predicted_strategies = self.strategy_net(states)
        loss = F.mse_loss(predicted_strategies, strategies)
        loss.backward()
        strategy_optimizer.step()

    # Train advantage networks
    for player, advantage_net in self.advantage_nets.items():
        advantage_optimizer = optim.Adam(advantage_net.parameters(), lr=learning_rate)
        for _ in range(self.n_training_steps_V):
            # Sample batch from advantage memory
            batch = random.sample(self.advantage_memory[player], min(len(self.advantage_memory[player]), batch_size))
            states, advantages = zip(*batch)

            states = torch.tensor(states, dtype=torch.float32)
            advantages = torch.tensor(advantages, dtype=torch.float32)

            # Forward pass
            advantage_optimizer.zero_grad()
            predicted_advantages = advantage_net(states)
            loss = F.mse_loss(predicted_advantages, advantages)
            loss.backward()
            advantage_optimizer.step()
    def _external_sampling(self, env, player):
        if env.is_terminal():
            return env.get_reward(player)
        
        current_player = env.get_current_player()
        state = env.get_state_representation()

        if current_player == player:
            # Player's turn: Use the strategy network to determine action
            strategy = self._get_strategy(state)
            action = self._sample_action(strategy)
            next_state, reward, done = env.step(action)
            next_value = 0 if done else self._external_sampling(env, player)
            advantage = next_value - reward

            # Store the experience in advantage memory
            self.advantage_memory[player].append((state, advantage))

            return next_value

        else:
            # Opponent's turn: Use the advantage network to determine action
            advantage_values = self.advantage_nets[current_player](torch.tensor(state, dtype=torch.float32))
            action = self._sample_action(advantage_values)
            env.step(action)
            return self._external_sampling(env, player)

    def _get_strategy(self, state):
        # Get strategy from strategy network
        strategy = self.strategy_net(torch.tensor(state, dtype=torch.float32))
        return F.softmax(strategy, dim=0).detach().numpy()

    def _sample_action(self, strategy):
        # Sample an action based on the given strategy
        return np.random.choice(range(len(strategy)), p=strategy)
