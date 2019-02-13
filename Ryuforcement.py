#!/usr/bin/env python3
import retro
import sys
import time
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.misc

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
		self.env.reset()

	def show(self):
		self.env.render()

	def step(self, action):
		self._obs, self._rew, self.done, self._info = self.env.step(action)
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

class imgA(object):
	def __init__(self, array):
		super(imgA, self).__init__()
		self.array = array
		self.gray = [[None]]
		self.opti = [[None]]
		self.resiz = [[None]]
		self.actSize = [len(array), len(array[0])]

	def toGray(self):
		self.gray = np.dot(self.array[...,:3], [0.299, 0.587, 0.114])
		return self.gray
	
	def toOpit(self):
		self.opti = self.gray[28:182]
		self.actSize = [len(self.opti), len(self.opti[0])]
		return self.opti
	
	def toResiz(self, x, y):
		self.resiz = cv2.resize(self.opti, dsize=(x, y), interpolation=cv2.INTER_CUBIC)
		self.actSize = [x, y]
		return self.resiz
	
	def show(self):
		plt.imshow(self.array)
		plt.imshow(self.gray) if self.gray[0][0] != None else 0
		plt.imshow(self.opti) if self.opti[0][0] != None else 0
		plt.imshow(self.resiz) if self.resiz[0][0] != None else 0
		plt.show()

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

	def train(self, st, output, action, value):
		o, at, vt = sess.run([output, action, value], feed_dict={tf_st:st})
		print("train")


	def play(self, env, eps):
		if random.uniform(0, 1) < eps:
			action = env.randomPlay()
		else:
			action = self.greedy_step(env)
			action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		return action

################################################

def create_model():
	print('\n','\n','\n','\n')
	width = 128
	height = 128
	channel = 4

	tf_st = tf.placeholder(dtype=tf.float32, shape=[None, height, width, channel], name='tf_image')
	tf_target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='tf_target')
	tf_action = tf.placeholder(dtype=tf.int32, shape=[None], name='action')

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

	return tf.Session(), output, action, value, train_op, error

################################################

def play(env, p, sess, output, action, value, train_op, error):
	env.show()
	first = True
	st = [None, None, None, None]
	while not env.is_finished(p):

		action = p.play(env, eps)

		_obs, _rew, done, _info = env.step(action)

		#FIT IMAGE
		img = imgA(_obs)
		#img.show()
		img.toGray()
		#img.show()
		img.toOpit()
		#img.show()
		img.toResiz(128, 128)
		#img.show()
		
		if first:
			first = False
			st = [img.resiz, img.resiz, img.resiz, img.resiz]
		else:
			st.append(img.resiz)
			del st[0]
		


		stp1 = st[:]
		env.show()

################################################

if __name__ == '__main__':

	# Create env
	env = envA()
	p = player()
	eps = 1
	
	sess, output, action, value, train_op, error = create_model()
	sess.run(tf.global_variables_initializer())
	for i in range(1000):
		env.reset()

		if i % 10 == 0:
			print(i)

		play(env, p, sess, output, action, value, train_op, error)
		print("Win :", p.win_nb, "Lose :", p.lose_nb, "Timeout :", p.timeout_nb)

		eps = max(eps * 0.999, 0.05)