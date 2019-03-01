#!/usr/bin/env python3
import os
import sys
import time
import retro
import random
import scipy.misc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple

################################################

class envA(object):
	def __init__(self):
		super(envA, self).__init__()
		self.env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
		self._obs = None
		self._rew = None
		self.done = False
		self._info = {'enemy_matches_won': 0, 'score': 0, 'matches_won': 0, 'continuetimer': 0, 'enemy_health': 176,'health': 176}
		self.lastHundred = []
		self.win_nb = 0
		self.lose_nb = 0
		self.timeout_nb = 0

	def reset(self):
		self.done = False
		self._info = {'enemy_matches_won': 0, 'score': 0, 'matches_won': 0, 'continuetimer': 0, 'enemy_health': 176,'health': 176}
		return self.env.reset()

	def show(self):
		self.env.render()

	def step(self, action):
		self._obs, _, self.done, _info = self.env.step(set_actions()[action])
		self._rew = (self._info['enemy_health'] - _info['enemy_health']) - (self._info['health'] - _info['health']) / 50
		self._info = _info
		return self._obs, self._rew, self.done, self._info

	def is_finished(self):
		if self._info['enemy_health'] < 0:
			self.lastHundred.append(1)
		elif self._info['health'] < 0:
			self.lastHundred.append(-1)
		if (len(self.lastHundred) > 100):
			self.lastHundred.pop(0)
		self.win_nb += 1 if self._info['enemy_health'] < 0 else 0
		self.lose_nb += 1 if self._info['health'] < 0 else 0
		self.timeout_nb += 1 if self.done else 0
		if self._info['enemy_health'] < 0 or self._info['health'] < 0 or self.done:
			return True
		return False

################################################

class stateProcessor(object):
	def __init__(self): #Create an sess of image transformation
		with tf.variable_scope("process"): # Image size 200 256
			self.input_state = tf.placeholder(shape=[200, 256, 3], dtype=tf.uint8, name="input_process")
			self.output = tf.image.rgb_to_grayscale(self.input_state)
			self.output = tf.image.crop_to_bounding_box(self.output, 28, 0, 158, 256)
			self.output = tf.image.resize_images(self.output, [128, 128], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			self.output = tf.squeeze(self.output)

	def process(self, sess, state)	:
		return sess.run(self.output, {self.input_state:state})

################################################

class Network():
	def __init__(self, scope):
		self.width = 128
		self.height = 128
		self.channel = 4
		self.scope = scope
		with tf.variable_scope(self.scope):
			self._build_model()

	def _build_model(self):

		print('\n', '---------------------------','\n')
		# 4 last frames of the game
		self.tf_st = tf.placeholder(dtype=tf.float32, shape=[None, self.height, self.width, self.channel], name='tf_image')
		self.tf_target = tf.placeholder(dtype=tf.float32, shape=[None], name='tf_target')
		self.tf_action = tf.placeholder(dtype=tf.int32, shape=[None], name='tf_action')

		# 32 filter with a size of 8 by 8 pixels, that move  4 pixels
		conv1 = tf.layers.conv2d(self.tf_st, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
		conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
		conv3 = tf.layers.conv2d(conv2, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
		# 64 image  with a size of 12 by 12 pixels

		print(conv1, '\n',conv2, '\n',conv3)

		# Create a vector with thes images
		flattened = tf.contrib.layers.flatten(conv3)
		# 9216 vector

		print(flattened)

		# 9216 input to 512 neurones
		fc1 = tf.layers.dense(flattened, 512, activation=tf.nn.relu)

		print(fc1)

		# 512 neurones to 64 output
		self.output = tf.layers.dense(fc1, 19, activation=None) # <- 64 doit devenir le nombre d'action possible

		print(self.output)

		self.action = tf.math.argmax(self.output, axis=1)

		print(self.action)

		self.value = tf.math.reduce_max(self.output, axis=1)

		print(self.value)

		self.action_value = tf.gather(tf.reshape(self.output, [-1]), self.tf_action)

		self.error = tf.reduce_mean(tf.squared_difference(self.action_value, self.tf_target))

		self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)

		self.train_op = self.optimizer.minimize(self.error)

		print('\n', '---------------------------','\n')

	def predict(self, sess, s):
		return sess.run(self.output, {self.tf_st:s})

	def update(self, sess, s, a, y):
		feed_dict = {self.tf_st:s, self.tf_target:y, self.tf_action:a}
		ops = [self.train_op, self.error]
		_, loss = sess.run(ops, feed_dict)
		return loss

################################################

def set_actions():
	#Movements
	neutral = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	right = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] 
	left = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
	crouch = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
	crouch_right = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
	crouch_left = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
	jump_neutral = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
	jump_right = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
	jump_left = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]

	#Normals
	standing_low_punch = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
	standing_medium_punch = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
	standing_high_punch = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
	standing_low_kick = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	standing_medium_kick = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	crouching_low_punch = [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
	crouching_medium_punch = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
	crouching_high_punch = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
	crouching_low_kick = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
	crouching_medium_kick = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]


	return [neutral, right, left, crouch, crouch_right, crouch_left, jump_neutral,jump_right, jump_left, standing_low_punch, standing_medium_punch, standing_high_punch, standing_low_kick, standing_medium_kick, crouching_low_punch, crouching_medium_punch, crouching_high_punch, crouching_low_kick, crouching_medium_kick]

################################################

def make_eps_greedy_policy(estimator, nA):
	def policy_fn(sess, observation, eps):
		A = np.ones(nA, dtype=float) * eps / nA
		q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
		best_act = np.argmax(q_values)
		A[best_act] += (1.0 - eps)
		return A
	return policy_fn

################################################

def copy_model_parameters(sess, estimator1, estimator2):
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)

################################################

if __name__ == '__main__':

	# Create env
	env = envA()

	# Create network
	net = Network(scope="net")
	# Create network target
	t_net = Network(scope="t_net")

	state_processor = stateProcessor()

	num_episode = 10000

	rp_memory_size = 5000
	rp_memory_init_size = 3000
	rp_memory = []

	update_target_estimator_every = 10000

	eps_start = 1
	eps_end = 0.1
	eps_decay_steps = 500000
	epss = np.linspace(eps_start, eps_end, eps_decay_steps)


	discount_fact = 0.99
	size_batch = 32

	opti_step = -1
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

		chkpnt_dir = os.path.join("./", "checkpoints")
		chkpnt_path = os.path.join(chkpnt_dir, "model")
		if not os.path.exists(chkpnt_dir):
			os.makedirs(chkpnt_dir)

		saver = tf.train.Saver()

		ladt_chckpnt = tf.train.latest_checkpoint(chkpnt_dir)

		valid_actions = set_actions()
		policy = make_eps_greedy_policy(net, len(valid_actions))

		epi_r = []
		best_epi_r = 0

		for i in range(num_episode):
			st = env.reset()
			st = np.stack([state_processor.process(sess, st) / 255] * 4, axis=2)
			loss = None
			r_sum = 0
			mean_epi_r = np.mean(epi_r)
			if best_epi_r < mean_epi_r:
				best_epi_r = mean_epi_r
				saver.save(tf.get_default_session(), chkpnt_path)

			act = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

			len_rp_memory = len(rp_memory)

			while not env.is_finished():
				eps = epss[min(opti_step+1, eps_decay_steps-1)]

				if opti_step % update_target_estimator_every == 0:
					copy_model_parameters(sess, net, t_net)

				print("\r Epsilon ({}) ReplayMemorySize : ({}) rSum: ({}) best_epi_reward: ({}) OptiStep ({}) @ Episode {}/{}, loss: {}".format(eps, len_rp_memory, mean_epi_r, best_epi_r, opti_step, i + 1, num_episode, loss), end="")
				sys.stdout.flush()

				action_probs = policy(sess, st, eps)
				act = np.random.choice(np.arange(len(valid_actions)), p=action_probs)


				# Step in the env with this action
				stp1, r, done, _ = env.step(act)
				r_sum += r
				# Process image
				stp1 = state_processor.process(sess, stp1) / 255

				# Add the image to the array
				stp1 = np.append(st[:,:,1:], np.expand_dims(stp1, 2), axis=2)

				if len(rp_memory) == rp_memory_size:
					rp_memory.pop(0)

				rp_memory.append(Transition(st, act, r, stp1, env.done))
				
				st = stp1[:]
				if done:
					break
				
				env.show()

			if len_rp_memory > rp_memory_init_size:
				for i in range(100):
					print(i)
					# Sample a minibatch from the replay memory
					samples = random.sample(rp_memory, size_batch)                
					states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

					# We compute the next q value with                
					q_values_next_target = t_net.predict(sess, next_states_batch)
					t_best_actions = np.argmax(q_values_next_target, axis=1)
					targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_fact * q_values_next_target[np.arange(size_batch), t_best_actions]
					
					# Perform gradient descent update
					states_batch = np.array(states_batch)
					loss = net.update(sess, states_batch, action_batch, targets_batch)

					opti_step +=1


			epi_r.append(r_sum)
			if len(epi_r) > 100:
				epi_r = epi_r[1:]
			print("Win :", env.win_nb, "Lose :", env.lose_nb, "Timeout :", env.timeout_nb)
			print("Diff Win - Lose last 100:", sum(env.lastHundred))
