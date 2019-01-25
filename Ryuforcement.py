#!/usr/bin/env python3
import retro
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

################################################

def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

################################################

if __name__ == '__main__':
	check_action = 0
	yeet = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	
	env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
	env.reset()
	env.render()
	
	while True:

		if check_action == 0:
			yeet = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
			check_action = 1
		elif check_action == 1:
			yeet = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
			check_action = 0
		
		_obs, _rew, done, _info = env.step(yeet)

		_grayObs = rgb2gray(_obs)
		plt.imshow(_grayObs[28:182], cmap = plt.get_cmap('gray'))
		plt.show()

		env.render()

		time.sleep(0.01)

		if done:
				break
