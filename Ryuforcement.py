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

	def is_finished(self):
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

def pick_action(env, eps):
	if random.uniform(0, 1) < eps:
		action = env.randomPlay()
	else:
		action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	return action

################################################

"""
def createNetwork():
	s = tf.placeholder("float", [None, 128, 77])
"""

################################################

"""
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")
"""

def play(env, sess):
	env.show()
	while not env.is_finished():
		action = pick_action(env, eps)

		_obs, _rew, done, _info = env.step(action)

		#FIT IMAGE
		img = imgA(_obs)
		#img.show()
		img.toGray()
		#img.show()
		img.toOpit()
		#img.show()
		img.toResiz(80, 80)
		#img.show()
		
		env.show()

################################################

if __name__ == '__main__':

	# Create env
	env = envA()
	eps = 1
	
	sess = tf.InteractiveSession()
	for i in range(1000):
		env.reset()

		if i % 10 == 0:
			print(i)

		play(env, sess)

		eps = max(eps * 0.999, 0.05)