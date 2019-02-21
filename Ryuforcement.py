#!/usr/bin/env python3
import sys
import cv2
import time
import retro
import random
import scipy.misc
import collections
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

################################################

class envA(object):
	def __init__(self):
		super(envA, self).__init__()
		self.env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
		self._obs = None
		self._rew = None
		self.done = False
		self._info = {'enemy_matches_won': 0, 'score': 0, 'matches_won': 0, 'continuetimer': 0, 'enemy_health': 176,'health': 176}
	
	def reset(self):
		self.done = False
		self._info = {'enemy_matches_won': 0, 'score': 0, 'matches_won': 0, 'continuetimer': 0, 'enemy_health': 176,'health': 176}
		return self.env.reset()

	def show(self):
		self.env.render()

	def step(self, action):
		self._obs, _, self.done, _info = self.env.step(action)
		self._rew = self._info['enemy_health'] - _info['enemy_health'] - self._info['health'] - _info['health']
		"""if (_info['enemy_health'] < self._info['enemy_health']):
			self._rew = 1
		elif (_info['health'] < self._info['health']):
			self._rew = -1
		else:
			self._rew = 0"""
		self._info = _info
		return self._obs, self._rew, self.done, self._info

	def randomPlay(self):
		return self.env.action_space.sample()

	def is_finished(self, p):
		p.win_nb += 1 if self._info['enemy_health'] < 0 else 0
		p.lose_nb += 1 if self._info['health'] < 0 else 0
		p.timeout_nb += 1 if self.done else 0
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

class player(object):
	def __init__(self):
		super(player, self).__init__()
		self.win_nb = 0.
		self.timeout_nb = 0.
		self.lose_nb = 0.

	def greedy_step(self, env):
		print("greedy_step")
		return 0

	def train(self, st, output, action, value, tf_st, stp1, tf_target, tf_action, r):
		print(tf_st, '\n\n\n')	
		o, at, vt = sess.run([output, action, value], feed_dict={tf_st:st})
		_, _, vtp1 = sess.run([output, action, value], feed_dict={tf_st: stp1})
		target = [r + 0.99*vtp1]
		_, err = sess.run([train_op, error], feed_dict={tf_st: st, tf_target: target, tf_action: at})
		print("train")


	def play(self, env, eps):
		if random.uniform(0, 1) < eps:
			action = env.randomPlay()
		else:
			action = self.greedy_step(env)
			action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
		return action

################################################

def create_model():
	print('\n','\n','\n','\n')
	width = 128
	height = 128
	channel = 4

	tf_st = tf.placeholder(dtype=tf.float32, shape=[None, height, width, channel], name='tf_image')
	tf_target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='tf_target')
	tf_action = tf.placeholder(dtype=tf.int32, shape=[None], name='tf_action')

	# 32 filtre de 8 par 8 pixels qui ce déplace de 4 pixels
	conv1 = tf.layers.conv2d(tf_st, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
	conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
	conv3 = tf.layers.conv2d(conv2, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
	# 64 image de 12 par 12 pixels

	print(conv1, '\n',conv2, '\n',conv3)

	# Créé un vecteur a partir des images
	flattened = tf.contrib.layers.flatten(conv3)
	# Vecteur de 9216

	print(flattened)

	# 9216 input to 512 neurones
	fc1 = tf.layers.dense(flattened, 512, activation=tf.nn.relu)

	print(fc1)

	# 512 neurones to 64 output
	output = tf.layers.dense(fc1, 64, activation=None)

	print(output)

	action = tf.math.argmax(output, axis=1)

	print(action)

	value = tf.math.reduce_max(output, axis=1)

	print(value)

	action_value = tf.gather(tf.reshape(output, [-1]), tf_action)

	error = tf.reduce_mean(tf.squared_difference(action_value, tf_target))

	optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)

	train_op = optimizer.minimize(error)
	
	print('\n', '\n','\n','\n')

	return output, action, value, train_op, error, tf_st, tf_target, tf_action

################################################

def play(env, p, state_processor, sess, output, action, value, train_op, error, tf_st, tf_target, tf_action, rp_memory, Transition):
	st = env.reset()
	st = np.stack([state_processor.process(sess, st)] * 4, axis=2)
	print(st.shape)
	env.show()

	act = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

	while not env.is_finished(p):

		act = p.play(env, eps)

		# Step in the env with this action
		stp1, r, _, _ = env.step(act)

		# Process image
		stp1 = state_processor.process(sess, stp1)

		# Add the image to the array
		stp1 = np.append(st[:,:,1:], np.expand_dims(stp1, 2), axis=2)

		if len(rp_memory) == 250000:
			rp_memory.pop(0)

		rp_memory.append(Transition(st, act, r, stp1, env.done))

		p.train(st, output, action, value, tf_st, stp1, tf_target, tf_action, r)


		st = stp1[:]
		env.show()

################################################

if __name__ == '__main__':

	# Create env
	env = envA()
	p = player()
	state_processor = stateProcessor()
	eps = 1
	rp_memory = []

	Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
	
	output, action, value, train_op, error, tf_st, tf_target, tf_action = create_model()
	
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	
	for i in range(1000):
		if i % 10 == 0:
			print(i)

		play(env, p, state_processor, sess, output, action, value, train_op, error, tf_st, tf_target, tf_action, rp_memory, Transition)
		print("Win :", p.win_nb, "Lose :", p.lose_nb, "Timeout :", p.timeout_nb)

		eps = max(eps * 0.999, 0.05)