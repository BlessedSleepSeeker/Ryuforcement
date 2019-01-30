#!/usr/bin/env python3
import retro
import sys
import time
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

################################################
# ScreenPlay Handle

def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

################################################
# Q-Learning

def random_play(env):
	return env.action_space.sample()

################################################

def pick_action(env, eps):
	if random.uniform(0, 1) < eps:
		action = random_play(env)
	else:
		action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	return action

################################################

if __name__ == '__main__':

	# Create env
	env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')

	# Create Model
	eps = 1

	for i in range(1000):
		if i % 10 == 0:
			print(i)
		env.reset()
		while True:
			action = pick_action(env, eps)

			_obs, _rew, done, _info = env.step(action)

			# Fit picture
			_grayObs = rgb2gray(_obs)
			_screenObs = _grayObs[28:182]
			_resScreenObs = cv2.resize(_screenObs, dsize=(128, 77), interpolation=cv2.INTER_CUBIC)


			#plt.imshow(_screenObs)
			#plt.show()
			#plt.imshow(_resScreenObs)
			#plt.show()

			if _info['enemy_health'] < 0 or _info['health'] < 0 or done:
				break
		eps = eps * 0.999