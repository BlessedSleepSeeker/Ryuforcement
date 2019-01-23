#!/usr/bin/env python3
import retro

env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
env.reset()
env.render()
while True:
	_obs, _rew, done, _info = env.step(env.action_space.sample())
	env.render()
	if done:
			break
