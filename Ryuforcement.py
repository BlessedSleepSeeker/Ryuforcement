#!/usr/bin/env python3
import retro
import sys
import time
import matplotlib.pyplot as plt

def getAction(env, _obs, _info):
	act = env.action_space.sample()
	return act

def main():
	env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
	env.reset()
	_obs = 0;
	_info = 0;
	while True:
		act = getAction(env, _obs, _info)
		_obs, _rew, done, _info = env.step(act)
		print(act)
		env.render()
		if done:
			break
		time.sleep(0.01)

if __name__ == '__main__':
	main()
