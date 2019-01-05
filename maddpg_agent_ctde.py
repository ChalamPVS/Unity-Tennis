from collections import namedtuple, deque
import numpy as np 
import random
import torch
import copy
import torch.nn.functional as F 
from model import Actor, Critic
import torch.optim as optim

BUFFER_SIZE = int(1e6)
SHARED_REPLAY_BUFFER = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
DISCOUNT = 0.99
UPDATE_EVERY = 4
TAU = 1e-3

ACTOR_HIDDEN_UNITS = (512, 256)
ACTOR_LR = 1e-4

CRITIC_HIDDEN_UNITS = (512, 256)
CRITIC_LR = 3e-4



class MultiAgent:
	def __init__(self, action_size, state_size, num_agents, seed=2):
		self.action_size = action_size
		self.state_size = state_size

		if SHARED_REPLAY_BUFFER:
			self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, DEVICE)

		self.ddpg_agents = [DDPGAgent(action_size, state_size, seed, self.memory if SHARED_REPLAY_BUFFER else None)
		for  _ in range(num_agents)]

		self.t_step = 0

	def reset(self):
		for agent in self.ddpg_agents:
			agent.reset()

	def act(self, all_states):
		"""get actions from all the agents in the MADDPG object"""
		actions = [agent.act(np.expand_dims(all_states[self.state_size*en:self.state_size*(en+1)], axis=0)) for en, agent in enumerate(self.ddpg_agents)]
		return actions

	def step(self, states, actions, rewards, next_states, dones, agent_index=None):
		# Save the experience in the replay memory
		
		if SHARED_REPLAY_BUFFER:
			self.memory.add(states, actions, rewards, next_states, dones)
		else:
			self.ddpg_agents[agent_index].memory.add(states, actions, rewards, next_states, dones)

		# Learn every UPDATE_EVERY time steps.
		self.t_step = (self.t_step + 1) % UPDATE_EVERY
		if self.t_step == 0:
			# If enough samples are available in the memory, get random subset and learn
			
			if SHARED_REPLAY_BUFFER:
				sample_size = len(self.memory)
			else:
				sample_size = len(self.ddpg_agents[agent_index].memory)

			if (sample_size) > BATCH_SIZE:
				for en, agent in enumerate(self.ddpg_agents):
					if SHARED_REPLAY_BUFFER:
						experiences = self.memory.sample()
					else:
						experiences = agent.memory.sample()

					agent.learn(experiences, DISCOUNT, en)



class DDPGAgent:
	def __init__(self, action_size, state_size, seed=2, commmon_buffer=None):
		self.action_size = action_size
		self.state_size = state_size
		self.seed = seed

		# Actor Network (w/ Target Network)
		self.actor_local = Actor(action_size, state_size, ACTOR_HIDDEN_UNITS, seed).to(DEVICE)
		self.actor_target = Actor(action_size, state_size, ACTOR_HIDDEN_UNITS, seed).to(DEVICE)
		self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=ACTOR_LR)

		#Critic Network (w/ Target Network)
		self.critic_local = Critic(action_size*2, state_size*2, CRITIC_HIDDEN_UNITS, seed).to(DEVICE)
		self.critic_target = Critic(action_size*2, state_size*2, CRITIC_HIDDEN_UNITS, seed).to(DEVICE)
		self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=CRITIC_LR)

		# ---------------- initialize target networks ---------------------- #
		self.soft_update(self.critic_local, self.critic_target, 1)
		self.soft_update(self.actor_local, self.actor_target, 1)

		self.noise = OUNoise(action_size, seed)

		if not SHARED_REPLAY_BUFFER:
			self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, DEVICE)
		else:
			self.memory = commmon_buffer


	def reset(self):
		self.noise.reset()


	def act(self, states):
		"""Returns actions for a given state as per the current policy."""
		states = torch.from_numpy(states).float().to(DEVICE)
		self.actor_local.eval()
		with torch.no_grad():
			actions = self.actor_local(states).cpu().data.numpy()
		self.actor_local.train()
		actions += self.noise.sample()
		return np.clip(actions, -1, 1)


	def learn(self, experiences, gamma, agent_index):
		"""Update policy and value parameters using the givne batch of experience tuples.
		Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
		where:
			actor_target(state) -> action
			critic_target(state, action) -> Q-value
		Params
		=======
			experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
			gamma (float):discount factor
		"""

		states, actions, rewards, next_states, dones = experiences

		# ----------------------------- update critic ----------------------#
		# Get the predicted next-state actions and Q values from target models
		ns = next_states[:, self.state_size*agent_index: self.state_size*(agent_index+1)]
		s = states[:, self.state_size*agent_index: self.state_size*(agent_index+1)]
		a = actions[:, self.action_size*agent_index: self.action_size*(agent_index+1)]
		r = rewards[:, agent_index]
		d = dones[:, agent_index]

		actions_next = self.actor_target(ns)
		if agent_index == 0:
			actions_next = torch.cat((actions_next, actions[:, self.action_size:]), dim=1)
		else:
			actions_next = torch.cat((actions[:, :self.action_size], actions_next), dim=1)

		Q_targets_next = self.critic_target(next_states, actions_next)
		# Compute Q targets fro the current state (y_i)
		Q_targets = r + (gamma * Q_targets_next * (1 - d))
		# Compute critic loss
		Q_expected = self.critic_local(states, actions)
		critic_loss = F.mse_loss(Q_expected, Q_targets)
		# Minimize the loss
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# -------------------------------- update actor ---------------------- #
		# Compute actor loss
		actions_pred = self.actor_local(s)
		if agent_index == 0:
			actions_pred = torch.cat((actions_pred, actions[:, self.action_size:]), dim=1)
		else:
			actions_pred = torch.cat((actions[:, :self.action_size], actions_pred), dim=1)
		actor_loss = -self.critic_local(states, actions_pred).mean()
		# Minimize the loss
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# ----------------------------------- update target networks --------------- #
		self.soft_update(self.critic_local, self.critic_target, TAU)
		self.soft_update(self.actor_local, self.actor_target, TAU)

	def soft_update(self, local_model, target_model, tau):
		"""Soft update the model parameters.
		θ_target = τ*θ_local + (1 - τ)*θ_target
		Params
		=======
			local_model: PyTorch model (weights will be copied from)
			target_model: PyTorch model (weights will be copied to)
			tau (float): interpolation parameter
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
	"""Ornstein-Uhlenbeck process."""

	def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
		"""Initialize parameters and noise process."""
		self.mu = mu * np.ones(size)
		self.theta = theta
		self.sigma = sigma
		self.seed = random.seed(seed)
		self.reset()

	def reset(self):
		"""Reset the internal state (= noise) to mean (mu)."""
		self.state = copy.copy(self.mu)

	def sample(self):
		"""Update internal state and return it as a noise sample."""
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
		self.state = x + dx
		return self.state


class ReplayBuffer:
	"""Fixed-size buffer to store experience tuples."""

	def __init__(self, action_size, buffer_size, batch_size, seed, device):
		"""Initialize a ReplayBuffer object.

		Params
		=======
			action_size (int): dimension of each action
			buffer_size (int): maximum size of buffer
			batch_size (int): size of each training batch
			seed (int): random seed
		"""
		self.action_size = action_size
		self.memory = deque(maxlen=buffer_size)
		self.batch_size = batch_size
		self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
		self.seed = random.seed(seed)
		self.device = device

	def add(self, state, action, reward, next_state, done):
		"""Add a new experience to memory."""
		e = self.experience(state, action, reward, next_state, done)
		self.memory.append(e)

	def sample(self):
		experiences = random.sample(self.memory, k=self.batch_size)

		states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
		actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
		rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
		next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
		dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

		return (states, actions, rewards, next_states, dones)

	def __len__(self):
		"""Return the current size of the internal memory."""
		return len(self.memory)