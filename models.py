from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Attention, Flatten, Dense, Reshape, Softmax, MaxPooling2D, Concatenate, LeakyReLU
from tensorflow.keras.constraints import max_norm

def ExternalRuleModel(params):
	grid_rows = params[0]-2
	grid_cols = params[1]-2
	grid_channels = params[2]
	num_rules = params[3]
	key_dim = params[4]
	val_dim = params[5]
	lr = params[6]
	num_actions = 4

	grid = Input(shape=(grid_rows, grid_cols, grid_channels))
	keys = Input(shape=(num_rules, key_dim))
	vals = Input(shape=(num_rules, val_dim))
	#softmax activation for the queries because the goal is to match one-hot encoded keys
	query = Conv2D(key_dim, 1, data_format='channels_last', padding='same', activation='softmax', kernel_initializer='zeros')(grid)
	query = Reshape((grid_rows * grid_cols, key_dim))(query)
	att = Attention(use_scale=True)([query, vals, keys]) #remember to increase the scale after initializing the weights
	att = Reshape((grid_rows, grid_cols, val_dim))(att)
	
	conv = Conv2D(32, 3, data_format='channels_last', padding='same', activation='linear')(att)
	pool = MaxPooling2D(pool_size=2, padding='same', data_format='channels_last')(conv)
	conv2 = Conv2D(32, 3, data_format='channels_last', padding='same', activation='linear')(pool)
	flat = Flatten()(conv2)
	hidden = Dense(128, activation='linear')(flat)
	
	hidden = LeakyReLU(alpha=.01)(hidden)

	actor = Dense(num_actions, activation='softmax', name="actor")(hidden)
	critic = Dense(1, activation='linear', name="critic")(hidden)
	model = Model(inputs=[grid, keys, vals], outputs=[actor,critic])
	return model

def InternalRuleModel(params):
	grid_rows = params[0]
	grid_cols = params[1]
	grid_channels = params[2]
	key_dim = params[3]
	val_dim = params[4]
	lr = params[5]
	num_actions = 4

	objects = Input(shape=(grid_rows, grid_cols, key_dim))
	names = Input(shape=(grid_rows, grid_cols, key_dim))
	roles = Input(shape=(grid_rows, grid_cols, val_dim+1)) #includes val_dim roles + "is"

	query = Conv2D(key_dim, 1, data_format='channels_last', padding='same', activation='softmax', bias_constraint=max_norm(0.0), name="query")(objects)
	query = Reshape((grid_rows * grid_cols, key_dim))(query)

	keys = Reshape((grid_rows * grid_cols, key_dim), name="key")(names)

	#we multiply the value embedding size by 2 to allow some redundancy and possibly make training easier. We only NEED val_dim channels.
	vals = Conv2D(val_dim*2, 5, data_format='channels_last', padding='same', activation='linear', name="value")(roles)
	vals = Reshape((grid_rows * grid_cols, val_dim*2))(vals)

	att = Attention(use_scale=True)([query, vals, keys]) #remember to increase the scale after initializing the weights
	att = Reshape((grid_rows, grid_cols, val_dim*2))(att)

	conv = Conv2D(32, 3, data_format='channels_last', padding='same', activation='linear')(att)
	pool = MaxPooling2D(pool_size=2, padding='same', data_format='channels_last')(conv)
	conv2 = Conv2D(32, 3, data_format='channels_last', padding='same', activation='linear')(pool)
	flat = Flatten()(conv2)
	hidden = Dense(128, activation='linear')(flat)

	hidden = LeakyReLU(alpha=.01)(hidden)

	actor = Dense(num_actions, activation='softmax', name="actor")(hidden)
	critic = Dense(1, activation='linear', name="critic")(hidden)
	model = Model(inputs=[objects, names, roles], outputs=[actor,critic])
	return model
