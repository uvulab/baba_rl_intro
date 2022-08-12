from ppo import Learner
from models import ExternalRuleModel, InternalRuleModel
from env import *
from level_gen_1 import Generator
import tensorflow as tf
import numpy as np
import random

seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

#EDIT THESE PARAMETERS========================================================

is_loading = False #if true load a saved model with model_name before training. testing will load a saved model either way.
is_training = True #run in train mode or test mode
num_rounds = 100 #total iterations of proximal policy optimization

external = True #whether to use the external rule task or the internal rule task
model_name = "10x10e"
rows = 10 #grid size: note that with external rules the playable grid size will be 2 less since the edges are out of bounds
cols = 10 #must be at least 5.
max_t = 75 #time limit per episode. 75 is ideal with a 10x10 grid; smaller grids can have less.

lr_start = .00005 #if lr_start > lr_end, the learning rate will decay to lr_end.
lr_end = .00005 #.00005 is recommended

steps_per_round=40000 #environment steps per iteration of PPO. So far, the larger the number, the better the model generalizes
num_epochs=32 #training epochs per iteration. Reducing this would increase speed.
clip_eps = 0.1 #a parameter for PPO.

#=============================================================================

gen = Generator(is_training)
game = Game(rows, cols, gen, separate_rules=external, max_t=max_t)

grid_channels = NUM_OBJECTS if external else NUM_SYMBOLS
num_rules = gen.num_rules
key_dim = NUM_OBJECTS
val_dim = NUM_ROLES

if external:
	net_params = [rows, cols, grid_channels, num_rules, key_dim, val_dim, lr_end]
else:
	net_params = [rows, cols, grid_channels, key_dim, val_dim, lr_end]

template = ExternalRuleModel if external else InternalRuleModel
multi_input = True

learner = Learner(model_name, game, template, net_params, multi_input=multi_input, lr_start=lr_start, lr_end=lr_end, steps_per_round=steps_per_round, num_epochs=num_epochs, clip_eps=clip_eps, gamma=.95)

#Increasing the scale of the attention layer is essential, otherwise all of the (query dot key) weights will be close to zero after softmaxing. Especially for the internal rule task.
if is_training:
	learner.model.get_layer("attention").set_weights([np.array(7.0)])

if is_loading and is_training:
	learner.load()

learner.run(is_training, num_rounds)
