import numpy as np
import random
from env import *

#generates a random level with rules of the form "a is you, b is win, c is lose" along the edges, and objects A, B, C inside

class Generator:

	def __init__(self, is_training):
		self.is_training = is_training
		self.num_rules = 3

	def generate(self, game):
		if self.is_training:
			available_names = {
				YOU: [BABA_NAME, ROCK_NAME, FLAG_NAME, SKULL_NAME],# WALL_NAME],
				WIN: [BABA_NAME, ROCK_NAME, FLAG_NAME, WALL_NAME],# SKULL_NAME],
				DEFEAT: [ROCK_NAME, FLAG_NAME, WALL_NAME, SKULL_NAME]#, BABA_NAME]
			}
		else:
			#this particular combination is set aside for testing
			available_names = {
				YOU: [WALL_NAME],
				WIN: [SKULL_NAME],
				DEFEAT: [BABA_NAME]
			}

		roles = [YOU, WIN, DEFEAT]
		positions = [0, 1, 2, 3]

		random.shuffle(roles)
		random.shuffle(positions)

		for i in range(3):
			name = random.choice(available_names[roles[i]])
			for k in available_names:
				if name in available_names[k]:
					available_names[k].remove(name)

			if positions[i] == 0 or positions[i] == 1:
				items = np.array([[name, IS, roles[i]]], dtype=int)
			else:
				items = np.array([[name], [IS], [roles[i]]], dtype=int)
			if positions[i] == 0: #top
				game.addItemsToArea(items, 0, 0, 1, game.cols)
			if positions[i] == 1: #bottom
				game.addItemsToArea(items, game.rows-1, 0, 1, game.cols)
			if positions[i] == 2: #left
				game.addItemsToArea(items, 0, 0, game.rows, 1)
			if positions[i] == 3: #right
				game.addItemsToArea(items, 0, game.cols-1, game.rows, 1)
	
			obj = np.array([[object_for_name(name)]], dtype=int)
			game.addItemsToArea(obj, 1, 1, game.rows-2, game.cols-2)
