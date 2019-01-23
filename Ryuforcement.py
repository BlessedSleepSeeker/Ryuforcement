#!/usr/bin/env python3
import retro
import sys
import time
import matplotlib.pyplot as plt

env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
env.reset()
env.render()
check_action = 0;
yeet = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
while True:
	#yeet = env.action_space.sample()
	if check_action == 0:
		yeet = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
		check_action = 1
	elif check_action == 1:
		yeet = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		check_action = 2
	elif check_action == 2:
		yeet = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		check_action = 0
	_obs, _rew, done, _info = env.step(yeet)
	print(_obs)
	plt.imshow(_obs)
	plt.show()
	print(yeet)
	env.render()
	time.sleep(0.01)
	if done:
			break
