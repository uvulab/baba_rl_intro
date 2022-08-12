from env import Game
from level_gen_1 import Generator

print("w,a,s,d to move")

gen = Generator(True)
game = Game(6, 6, gen, separate_rules=True, max_t=10)
game.show()

done = False

while not done:
	c = input(">")
	if c == 'w':
		action = 0
	elif c == 's':
		action = 1
	elif c == 'a':
		action = 2
	elif c == 'd':
		action = 3
	else:
		print("valid actions: w, a, s, d")
		continue
	
	_, reward, done, _ = game.step(action)
	game.show()

	if done:
		print("reward:", reward)
