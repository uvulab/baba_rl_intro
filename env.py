import numpy as np
import random

BABA_OBJ = 0
FLAG_OBJ = 1
ROCK_OBJ = 2
WALL_OBJ = 3
SKULL_OBJ = 4
BABA_NAME = 5
FLAG_NAME = 6
ROCK_NAME = 7
WALL_NAME = 8
SKULL_NAME = 9
IS = 10
YOU = 11
WIN = 12
DEFEAT = 13
PUSH = 14
STOP = 15

#we aren't currently using PUSH or STOP, although they are implemented here. If you want to use them, increase NUM_SYMBOLS and NUM_ROLES by 2
NUM_SYMBOLS = 14#16
NUM_OBJECTS = 5
NUM_ROLES = 3#5

separate_onehots = {
	BABA_OBJ: 0,
	FLAG_OBJ: 1,
	ROCK_OBJ: 2,
	WALL_OBJ: 3,
	SKULL_OBJ: 4,
	BABA_NAME: 0,
	FLAG_NAME: 1,
	ROCK_NAME: 2,
	WALL_NAME: 3,
	SKULL_NAME: 4,
	IS: 0,
	YOU: 0,
	WIN: 1,
	DEFEAT: 2,
	PUSH: 3,
	STOP: 4
}

icons = {
	BABA_OBJ: 'B',
	FLAG_OBJ: 'F',
	ROCK_OBJ: 'R',
	WALL_OBJ: 'W',
	SKULL_OBJ: 'S',
	BABA_NAME: 'b',
	FLAG_NAME: 'f',
	ROCK_NAME: 'r',
	WALL_NAME: 'w',
	SKULL_NAME: 's',
	IS: 'i',
	YOU: 'y',
	WIN: 'v',
	DEFEAT: 'd',
	PUSH: 'p',
	STOP: 'x'
}

def is_object(s):
	return s >= 0 and s <= 4

def is_name(s):
	return s >= 5 and s <= 9

def is_role(s):
	return s >= 11 and s <= 15

def is_text(s):
	return s >= 5

def object_for_name(s):
	if s == BABA_NAME:
		return BABA_OBJ
	if s == FLAG_NAME:
		return FLAG_OBJ
	if s == ROCK_NAME:
		return ROCK_OBJ
	if s == WALL_NAME:
		return WALL_OBJ
	if s == SKULL_NAME:
		return SKULL_OBJ
	print("error: unknown object")
	return None

def name_for_object(s):
	if s == BABA_OBJ:
		return BABA_NAME
	if s == FLAG_OBJ:
		return FLAG_NAME
	if s == ROCK_OBJ:
		return ROCK_NAME
	if s == WALL_OBJ:
		return WALL_NAME
	if s == SKULL_OBJ:
		return SKULL_NAME
	print("error: unknown name")
	return None

class Game:
	def __init__(self, rows, cols, generator, separate_rules=False, max_t=5):
		self.rows = rows
		self.cols = cols
		self.generator = generator
		self.separate_rules = separate_rules
		self.max_t = max_t
		self.num_actions = 4
		self.reset()

	def reset(self):
		self.grid = []
		for r in range(self.rows):
			self.grid.append([])
			for c in range(self.cols):
				self.grid[r].append([])
		self.generator.generate(self)
		self.roles = self.read_roles()
		self.already_moved = []
		self.t = 0
		return self.get_state()

	#items is a 2d numpy array of symbol IDs, or -1 for any empty space
	#symbols may only be placed on empty spaces
	#start_row, start_col, num_rows, num_cols denote a bounding box in which the items could possibly be placed
	def addItemsToArea(self, items, s_row, s_col, n_rows, n_cols):
		options = []
		for r in range(s_row, s_row + n_rows - items.shape[0] + 1):
			for c in range(s_col, s_col + n_cols - items.shape[1] + 1):
				if self.can_fit(items, r, c):
					options.append((r,c))
		if len(options) == 0:
			print("invalid placement")
			return False
		(row, col) = random.choice(options)
		for r in range(items.shape[0]):
			for c in range(items.shape[1]):
				if items[r][c] != -1:
					self.grid[row+r][col+c].append(items[r][c])
		self.roles = self.read_roles()
		return True

	def can_fit(self, items, row, col):
		for r in range(items.shape[0]):
			for c in range(items.shape[1]):
				if not self.in_bounds(row + r, col + c):
					return False
				if items[r][c] != -1 and len(self.grid[row+r][col+c]) > 0:
					return False
		return True

	def step(self, action):
		if action == 0:
			dr, dc = -1, 0
		elif action == 1:
			dr, dc = 1, 0
		elif action == 2:
			dr, dc = 0, -1
		elif action == 3:
			dr, dc = 0, 1
		else:
			print("invalid action")
			exit()

		did_move = False
		for r in range(self.rows):
			for c in range(self.cols):
				for obj in self.grid[r][c]:
					if obj in self.roles[YOU] and self.can_move(obj, r, c, dr, dc):
						self.move(obj, r, c, dr, dc)
						did_move = True

		self.roles = self.read_roles()
		self.already_moved = []

		self.t += 1
		reward = 0.0#-1.0/self.max_t# * (1.0 if did_move else 2.0)
		done = False

		if self.t >= self.max_t:
			done = True

		if self.defeat():
			reward = -1.0
			done = True

		if self.victory():
			reward = 1.0
			done = True

		return self.get_state(), reward, done, reward > 0

	def get_state(self):
		if self.separate_rules:
			#the edges are out of bounds, rules must be placed there
			g = np.zeros((self.rows-2, self.cols-2, NUM_OBJECTS),dtype=float)
			k = np.zeros((self.generator.num_rules, NUM_OBJECTS),dtype=float)
			v = np.zeros((self.generator.num_rules, NUM_ROLES),dtype=float)
			for r in range(self.rows-2):
				for c in range(self.cols-2):
					for s in self.grid[r+1][c+1]:
						if is_object(s):
							g[r,c,separate_onehots[s]] = 1.0
			rule_i = 0
			for role in self.roles:
				for obj in self.roles[role]:
					k[rule_i,separate_onehots[name_for_object(obj)]] = 1.0
					v[rule_i,separate_onehots[role]] = 1.0
					rule_i += 1
			return [g, k, v]
		else:
			result = np.zeros((self.rows, self.cols, NUM_SYMBOLS),dtype=float)
			for r in range(self.rows):
				for c in range(self.cols):
					for s in self.grid[r][c]:
						result[r,c,s] = 1.0
			q = result[:,:,BABA_OBJ:SKULL_OBJ+1]
			k = result[:,:,BABA_NAME:SKULL_NAME+1]
			v = result[:,:,IS:DEFEAT+1]
			return [q, k, v]
			#return result

	def read_roles(self):
		roles = dict()
		roles[YOU] = []
		roles[WIN] = []
		roles[DEFEAT] = []
		roles[PUSH] = []
		roles[STOP] = []

		for r in range(self.rows):
			for c in range(self.cols):
				for s in self.grid[r][c]:
					if is_name(s):
						obj = object_for_name(s)
						if self.contains_is(r, c+1) and not self.get_role(r, c+2) == None:
							role = self.get_role(r, c+2)
							if not obj in roles[role]:
								roles[role].append(obj)
						if self.contains_is(r+1, c) and not self.get_role(r+2, c) == None:
							role = self.get_role(r+2, c)
							if not obj in roles[role]:
								roles[role].append(obj)
		return roles

	def can_move(self, obj, r, c, dr, dc):
		if (obj, r, c) in self.already_moved:
			return False
		new_r = r + dr
		new_c = c + dc
		if not self.in_bounds(new_r, new_c):
			return False
		#with separate rules, the edge contains the rules and is out of bounds
		if self.separate_rules and (new_r == 0 or new_r == self.rows-1 or new_c == 0 or new_c == self.cols-1):
			return False
		for other in self.grid[new_r][new_c]:
			if self.stopped_by(obj, other):
				return False
			if self.can_push(obj, other) and not self.can_move(other, new_r, new_c, dr, dc):
				return False
		return True

	#requires can_move to be called first!
	def move(self, obj, r, c, dr, dc):
		new_r = r + dr
		new_c = c + dc
		#there shouldn't be more than one pushable object, but just in case
		to_push = []
		for other in self.grid[new_r][new_c]:
			if self.can_push(obj, other):
				to_push.append(other)
		for other in to_push:
			self.move(other, new_r, new_c, dr, dc)
		self.grid[new_r][new_c].append(obj)
		self.grid[r][c].remove(obj)
		if obj in self.roles[YOU]:
			self.already_moved.append((obj, new_r, new_c))

	def defeat(self):
		if len(self.roles[YOU]) == 0:
			return True
		for r in range(self.rows):
			for c in range(self.cols):
				you = False
				defeat = False
				for s in self.grid[r][c]:
					if s in self.roles[YOU]:
						you = True
					if s in self.roles[DEFEAT]:
						defeat = True
					if you and defeat:
						return True
		return False

	def victory(self):
		for r in range(self.rows):
			for c in range(self.cols):
				you = False
				win = False
				for s in self.grid[r][c]:
					if s in self.roles[YOU]:
						you = True
					if s in self.roles[WIN]:
						win = True
					if you and win:
						return True
		return False

	def stopped_by(self, obj, other):
		return other in self.roles[STOP]

	def can_push(self, obj, other):
		return is_text(other) or other in self.roles[PUSH] or obj == other

	def contains_is(self, r, c):
		if not self.in_bounds(r, c):
			return False
		for s in self.grid[r][c]:
			if s == IS:
				return True
		return False

	def get_role(self, r, c):
		if not self.in_bounds(r, c):
			return None
		for s in self.grid[r][c]:
			if is_role(s):
				return s
		return None

	def in_bounds(self, r, c):
		return r >= 0 and r < self.rows and c >= 0 and c < self.cols

	def show(self):
		for r in range(self.rows):
			line = ""
			for c in range(self.cols):
				if len(self.grid[r][c]) == 0:
					line += "_"
				else:
					for item in self.grid[r][c]:
						line += icons[item]
				line += " "
			print(line)
