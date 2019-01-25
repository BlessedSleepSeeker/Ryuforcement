#!/usr/bin/env python3
import retro
import sys
import time
import random
import matplotlib.pyplot as plt
import numpy as np

################################################
#ScreenPlay Handle

def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

################################################

################################################
#Q-Learning

def eps_modify(eps, eps_mult):
	return (eps * eps_mult);

def random_play(env):
	return env.action_space.sample()

def pick_action(env, eps):
	if random.uniform(0, 1) < eps:
		action = random_play(env);
	else:
		action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	return action;

################################################
#Main

if __name__ == '__main__':
	env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
	env.reset()
	env.render()

	eps = 1;
	while True:
		action = pick_action(env, eps)

		_obs, _rew, done, _info = env.step(action)

		eps = eps_modify(eps, 0.999);
		#_grayObs = rgb2gray(_obs)
		#plt.imshow(_grayObs[28:182], cmap = plt.get_cmap('gray'))
		#plt.show()

		env.render()
		if done:
			break
