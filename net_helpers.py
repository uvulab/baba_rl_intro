import numpy as np
from tensorflow.keras.models import Model

def predict(model, inputs):
	return model(inputs).numpy()

def train(model, inputs, outputs):
	model.fit(inputs, outputs, epochs=1, verbose=0)

def batch_multi(states, multi_input):
	#list of states
	if multi_input:
		#each state is a list of inputs, return list of batches
		result = []
		for i in range(len(states[0])):
			result.append(np.array([s[i] for s in states]))
		return result
	else:
		return np.array(states)

def batch_single(state, multi_input):
	if multi_input:
		result = []
		for x in state:
			result.append(np.expand_dims(x,0))
		return result
	else:
		return np.expand_dims(state,0)

def copy_layer(dest_model, src_model, layer_name):
	dest_layer = dest_model.get_layer(layer_name)
	src_layer = src_model.get_layer(layer_name)
	dest_layer.set_weights(src_layer.get_weights())

def copy_weights(source_model, target_model, fraction):
	new_weights = []
	for ws, wt in zip(source_model.get_weights(), target_model.get_weights()):
		new_weights.append(fraction * ws + (1.0 - fraction) * wt)
	target_model.set_weights(new_weights)

def get_actor_critic_models(model):
	actor_model = Model(inputs=model.input, outputs=model.get_layer("actor").output)
	critic_model = Model(inputs=model.input, outputs=model.get_layer("critic").output)
	return actor_model, critic_model
