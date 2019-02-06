#!/usr/bin/env python3
import pandas as pd
import numpy as np
from random import randint
import random

class envA(object):
	def __init__(self):
		super(envA, self).__init__()
		self.St = 2
		self.r = [-1, -0.1, -0.1, -0.1, 1]
		self.action = [-1, 1]

	def reset(self):
		self.St = 2
		return self.St
	
	def step(self, choice):
		self.St += choice
		return self.r[self.St], self.St

	def show(self):
		for i in range(5):
			if i == self.St:
				print("+", end="")
			else:
				print("_", end="")
		print()
	
	def is_finished(self):
		if self.St == 0 or self.St == 4:
			return True
		return False

class player(object):
	def __init__(self):
		self.Q = np.zeros((5, 2))

	def greedy_step(self, St):
		return np.argmax(self.Q[St])

	def train(self, Qlist, at, env):
		St1 = env.St + at
		at1 = env.action[self.greedy_step(St1)]
		if (at == -1):
			at = 0
		self.Q[env.St][at] += (0.1 * (env.r[St1] + 0.99 * (self.Q[St1][at1] - self.Q[env.St][at])))

def play(env, p, eps):
	choices = env.action
	if random.uniform(0, 1) < eps:
		random.shuffle(choices)
		choice = choices[0]
	else:
		choice = env.action[p.greedy_step(env.St)]

	p.train(p.Q, choice, env)
	env.step(choice)

if __name__ == '__main__':
	env = envA()
	env.show()
	p = player()
	eps = 1

	for i in range(10000):
		st = env.reset()
		while not env.is_finished():
			play(env, p, eps)
			#input()
		print("------------------", eps)
		print(p.Q,'\n')
		print("------------------")
		eps = max(eps * 0.999, 0.05)
	print("END")