import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from net_helpers import *

#partly based on https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py
class Learner:
	def __init__(self, model_name, game, net_template, net_params, lr_start=.00005, lr_end=.00005, batch_size=128, steps_per_round=128, num_epochs=4, clip_eps = 0.1, gamma=0.99, multi_input=False):
		self.model_filename = "models/"+model_name+".h5"
		self.model = net_template(net_params)
		self.actor_model, self.critic_model = get_actor_critic_models(self.model)
		self.game = game
		self.batch_size = batch_size
		self.steps_per_round = steps_per_round
		self.num_epochs = num_epochs
		self.clip_eps = clip_eps
		self.gamma = gamma
		self.multi_input = multi_input
		self.lr = lr_start
		self.lr_end = lr_end
		self.optimizer = Adam(learning_rate=self.lr)
		self.mse_loss = MeanSquaredError()
		self.mem = []
		self.wins = 0
		self.episodes = 0
		self.model.summary()
		self.reset()

	def reset(self):
		self.curr_state = self.game.reset()
		self.reward_sum = 0.0
		self.ep_mem = []

	def choose_action(self, is_training):
		[action_probs, state_value] = self.model(batch_single(self.curr_state, self.multi_input))
		action_probs = np.squeeze(action_probs.numpy())
		if is_training:
			print(action_probs, state_value.numpy())
		if not is_training:
			action = np.argmax(action_probs)
		else:
			action = np.random.choice(self.game.num_actions, p = action_probs)
		logprob = np.log(np.maximum(1e-8, action_probs[action]))
		return action, logprob

	def step(self, is_training):
		action, logprob = self.choose_action(is_training)
		next_state, reward, done, success = self.game.step(action)
		self.reward_sum += reward
		self.ep_mem.append((self.curr_state, action, reward, logprob))
		self.curr_state = next_state
		if done:
			print(self.reward_sum, success, "=====================================")
			if success:
				self.wins += 1
			self.episodes += 1
			ret = 0.0
			for (state, action, reward, logprob) in reversed(self.ep_mem):
				ret = reward + self.gamma * ret
				self.mem.append((state, action, ret, logprob))
			self.reset()

	def train(self):
		all_returns = np.array([val[2] for val in self.mem])
		ret_mean = np.mean(all_returns)
		ret_std = np.std(all_returns)

		for i in range(self.num_epochs):
			self.loss_sum = 0.0
			random.shuffle(self.mem)
			start = 0
			while start < len(self.mem):
				#print(start)
				stop = min(start + self.batch_size, len(self.mem))
				batch = self.mem[start: stop]
				states = batch_multi([val[0] for val in batch], self.multi_input)
				actions = np.array([val[1] for val in batch])
				returns = np.array([val[2] for val in batch])
				returns = (returns - ret_mean) / (ret_std + 1e-7)
				logprobs = np.array([val[3] for val in batch])

				self.train_on_batch(states, actions, returns, logprobs)

				start += self.batch_size

			print("epoch", i, self.loss_sum)

		self.mem = []
		self.reset()

	#remember to normalize the returns
	def train_on_batch(self, states, actions, returns, old_logprobs):
		with tf.GradientTape() as tape:
			state_values = self.critic_model(states)
			fixed_state_values = state_values.numpy()
			critic_loss = tf.cast(self.mse_loss(state_values, returns), tf.float32)
			grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
			self.optimizer.apply_gradients(zip(grads, self.critic_model.trainable_variables))

		with tf.GradientTape() as tape:
			probs = self.actor_model(states)

			#actions starts as 1d array
			logprobs = tf.math.log(tf.math.maximum(probs, 1e-8))
			new_logprobs = tf.squeeze(tf.gather(logprobs, tf.expand_dims(actions, -1), axis=-1, batch_dims=-1))

			ratios = tf.math.exp(new_logprobs - old_logprobs)

			advantages = returns - fixed_state_values

			#can add entropy to the loss function if you uncomment the next two comments
			#entropy = tf.math.reduce_mean(-logprobs * probs)

			surr1 = ratios * advantages
			surr2 = tf.clip_by_value(ratios, 1.0-self.clip_eps, 1.0+self.clip_eps) * advantages

			actor_loss = -tf.math.reduce_mean(tf.math.minimum(surr1, surr2)) #- .05 * entropy
			self.loss_sum += actor_loss.numpy()

			grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)

			self.optimizer.apply_gradients(zip(grads, self.actor_model.trainable_variables))

	def decay_lr(self):
		if self.lr > self.lr_end:
			self.lr -= self.lr_end
			if self.lr < self.lr_end:
				self.lr = self.lr_end
			self.optimizer = Adam(learning_rate=self.lr)

	def run(self, is_training, num_rounds):
		if not is_training:
			self.load()
		for i in range(num_rounds):
			for _ in range(self.steps_per_round):
				self.step(is_training)
			print("Success rate:", self.wins, "/", self.episodes, "===========================================") 
			self.wins = 0
			self.episodes = 0
			if is_training:
				print("round", i, "=============================================")
				self.train()
				self.save()
				self.decay_lr()

	def save(self):
		self.model.save_weights(self.model_filename)

	def load(self):
		self.model.load_weights(self.model_filename)
		self.actor_model, self.critic_model = get_actor_critic_models(self.model)
