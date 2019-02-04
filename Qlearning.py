#!/usr/bin/env python3
import numpy as np
from random import randint
import random
import pandas as pd


class envA(object):
	def _init_(self):
		


reward = [-1, -0.1, -0.1, -0.1, 1]
action = [-1, 1]

def step(state, choice):
	state += choice
	return reward[state], state

def greedy_step(state, Qlist):
	#print(state)
	if (Qlist[0][state] >= Qlist[1][state]):
		return -1
	else:
		return 1


def greedy_step2(state, Qlist):
	#print(state)
	if (Qlist[0][state] >= Qlist[1][state]):
		return 0
	else:
def train(state, Qlist, choice):
	stachoi = state + choice
	if (choice == -1):
		choice = 0
	Qlist[choice][state] += (0.1 * (reward[stachoi] + 0.99 * (Qlist[greedy_step2(stachoi, Qlist)][stachoi] - Qlist[choice][state])))
	return Qlist

def play(state, Qlist, eps):
	choice = None
	if random.uniform(0, 1) < eps:
		choice = random.randint(0,1)
		if (choice == 0):
			choice = -1
		print(choice, state+choice)
		print(pd.DataFrame(Qlist),'\n')
		#input()
	else:
		choice = greedy_step(state, Qlist)
	#print(choice)
	#input()

	Qlist = train(state, Qlist, choice)
	reward, state = step(state, choice)
	if (reward != -0.1):
		return True, state, Qlist
	return False, state, Qlist

if __name__ == '__main__':
	Qlist = [[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]]
	eps = 1
	#print(Qlist[1])
	for i in range(10000):
		"""if (i % 100 == 0):
			print(i)"""
		state = 2
		endGame = False
		while (endGame == False):
			endGame, state, Qlist = play(state, Qlist, eps)
			#input()
		print("------------------", eps)
		print(pd.DataFrame(Qlist),'\n')
		print("------------------")
		eps = max(eps * 0.999, 0.05)
	print("END")