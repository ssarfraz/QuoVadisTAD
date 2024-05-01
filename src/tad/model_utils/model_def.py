import os
import re
from pathlib import Path
import numpy as np
import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tad import DEFAULT_RESOURCE_DIR
from tad.dataset.reader import GeneralDataset, dataset_loader_map
from tad.utils.data_utils import preprocess_data, concatenate_windows_feat
from tad.model_utils.tf_data_loader import create_tf_dataset
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tad.model_utils.mc_model_utils import load_model_mc
from tad.model_utils.gnn import get_graph_info, LSTMGC, LSTMGC_recons


def get_label_loader(labels, config):
	labels_gen = timeseries_dataset_from_array(
		labels[:-1],
		None,
		sequence_length=config["input_sequence_length"],
		shuffle=False,
		batch_size=config["batch_size"],
	)
	return labels_gen


def get_dataloader(
	data_array,
	input_sequence_length=12,
	batch_size=1,
	reconstruction_dataset=False,
	shuffle=False,
):
	# TODO paqss this also from configs. for now it is fixed
	forecast_horizon = 1
	return create_tf_dataset(
		data_array,
		input_sequence_length,
		forecast_horizon,
		batch_size=batch_size,
		reconstruction_dataset=reconstruction_dataset,
		shuffle=shuffle,
	)


def norm_1_1(arr):
	return 2 * arr - 1


def read_data(
	dataset: GeneralDataset, dataset_trace=None, preprocess="0-1", config=None
):
	# prepare dataset
	trainset, testset, labels = dataset_loader_map[dataset]()

	if dataset in [GeneralDataset.WADI, GeneralDataset.WADI_GDN]:
		# WADI dataset is very large, we use the first 60.000 samples
		trainset = trainset[:60_000]

	if type(trainset) is list:
		if dataset_trace is not None:
			trainset, testset, labels = (
				trainset[dataset_trace],
				testset[dataset_trace],
				labels[dataset_trace],
			)
		else:
			trainset, testset, labels = trainset[0], testset[0], labels[0]
			print(" using first trace of this dataset. Specify dataset_trace otherwise")

	if trainset.shape[1] == 1:  # check if univariate timeseries
		trainset = concatenate_windows_feat(trainset, window_size=5)
		testset = concatenate_windows_feat(testset, window_size=5)

	if len(labels.shape) > 1:
		labels = labels.max(1)

	# Preprocess datasets
	trainset, valset, testset = preprocess_data(
		trainset, testset, 0.9, 0.1, normalization=preprocess
	)
	# if config["model"] == "seq_sensor_model" :  #and config["preprocess"] == "0-1"
	#    return norm_1_1(trainset), norm_1_1(valset), norm_1_1(testset), labels
	return trainset, valset, testset, labels


class MLPMixerLayer(layers.Layer):
	def __init__(self, num_patches, hidden_units, dropout_rate, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.mlp1 = keras.Sequential(
			[
				layers.Dense(units=num_patches),
				tfa.layers.GELU(),
				layers.Dense(units=num_patches),
				layers.Dropout(rate=dropout_rate),
			]
		)
		self.mlp2 = keras.Sequential(
			[
				layers.Dense(units=num_patches),
				tfa.layers.GELU(),
				layers.Dense(units=hidden_units),
				layers.Dropout(rate=dropout_rate),
			]
		)
		self.normalize = layers.LayerNormalization(epsilon=1e-6)

	def call(self, inputs):
		# Apply layer normalization.
		x = self.normalize(inputs)
		# Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
		x_channels = tf.linalg.matrix_transpose(x)
		# Apply mlp1 on each channel independently.
		mlp1_outputs = self.mlp1(x_channels)
		# Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
		mlp1_outputs = tf.linalg.matrix_transpose(mlp1_outputs)
		# Add skip connection.
		x = mlp1_outputs + inputs
		# Apply layer normalization.
		x_patches = self.normalize(x)
		# Apply mlp2 on each patch independtenly.
		mlp2_outputs = self.mlp2(x_patches)
		# Add skip connection.
		x = x + mlp2_outputs
		return x


class FNetLayer(layers.Layer):
	def __init__(self, num_patches, embedding_dim, dropout_rate, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.ffn = keras.Sequential(
			[
				layers.Dense(units=embedding_dim),
				tfa.layers.GELU(),
				layers.Dropout(rate=dropout_rate),
				layers.Dense(units=embedding_dim),
			]
		)

		self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
		self.normalize2 = layers.LayerNormalization(epsilon=1e-6)

	def call(self, inputs):
		# Apply fourier transformations.
		x = tf.cast(
			tf.signal.fft2d(tf.cast(inputs, dtype=tf.dtypes.complex64)),
			dtype=tf.dtypes.float32,
		)
		# Add skip connection.
		x = x + inputs
		# Apply layer normalization.
		x = self.normalize1(x)
		# Apply Feedfowrad network.
		x_ffn = self.ffn(x)
		# Add skip connection.
		x = x + x_ffn
		# Apply layer normalization.
		return self.normalize2(x)


class TransformerBlock(layers.Layer):
	def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
		super().__init__()
		self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
		self.ffn = layers.Dense(ff_dim, activation="relu")
		# self.ffn = keras.Sequential(
		#    [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
		# )
		self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
		self.dropout1 = layers.Dropout(rate)
		self.dropout2 = layers.Dropout(rate)

	def call(self, inputs, training):
		attn_output = self.att(inputs, inputs)
		attn_output = self.dropout1(attn_output, training=training)
		out1 = self.layernorm1(inputs + attn_output)
		ffn_output = self.ffn(out1)
		ffn_output = self.dropout2(ffn_output, training=training)
		return self.layernorm2(out1 + ffn_output)


def conv_blk(inputs, filters, kernel_size, strides=(1, 1), activation=None, BN=True):
	conv_out = layers.Conv2D(
		filters, kernel_size, strides=strides, padding="same", activation=activation
	)(inputs)
	if BN:
		conv_out = layers.BatchNormalization()(conv_out)
	return conv_out


def build_conv_model(input_shape, training=True):
	inputs = layers.Input(shape=input_shape)
	inputs = layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
	conv_out = conv_blk(inputs, 16, 3, strides=(1, 1), activation="relu", BN=False)
	conv_out = conv_blk(conv_out, 32, 3, strides=(1, 1), activation="relu", BN=False)
	conv_out = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(
		conv_out
	)
	conv_out = conv_blk(conv_out, 64, 3, strides=(1, 1), activation="relu", BN=False)
	conv_out = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(
		conv_out
	)

	conv_out = conv_blk(
		conv_out, input_shape[1], 3, strides=(1, 1), activation="relu", BN=False
	)

	# conv_out = layers.Dropout(rate=0.1, name='target_dropout')(conv_out)

	regression_targets = layers.GlobalAveragePooling2D(keepdims=True)(conv_out)
	regression_targets = layers.Reshape((1, input_shape[1]))(regression_targets)
	if not training:
		model = keras.Model(inputs=inputs, outputs=[regression_targets, conv_out])
	else:
		model = keras.Model(inputs=inputs, outputs=regression_targets)
	return model


def sequense_sensor_error_model(input_shape, config=None, training=True):
	lstm_units = config["lstm_units"]
	embedd_1 = config["embedding_dim"][0]
	embedd_2 = config["embedding_dim"][1]
	num_seq, num_nodes = input_shape

	inputs_nodes = layers.Input(shape=input_shape)  # (batch, seq, nodes)
	inputs_seq = layers.Permute((2, 1))(inputs_nodes)  # (batch, nodes, seq)
	# seq_trend_leaner
	inputs_seq = layers.Dense(embedd_1, activation="relu")(
		inputs_seq
	)  # (batch, nodes, embedding_dim)
	regression_seq = layers.LayerNormalization(epsilon=1e-6)(inputs_seq)

	# regression_seq = MLPMixerLayer(num_nodes, embedd_1, config['dropout_rate'])(inputs_seq)
	# regression_seq = TransformerBlock(embedd_1, config['MHA_blocks'], embedd_1, config['dropout_rate'])(inputs_seq)

	regression_seq = layers.LSTM(lstm_units)(regression_seq)  # (batch, embedding_dim)

	# sensors_trend_leaner
	regression_node = layers.Dense(embedd_2, activation="relu")(
		inputs_nodes
	)  # (batch, seq, embedding_dim)
	regression_node = layers.LayerNormalization(epsilon=1e-6)(regression_node)

	# regression_node = MLPMixerLayer(num_seq, embedd_2, config['dropout_rate'])(regression_node)
	# regression_node = TransformerBlock(embedd_2, config['MHA_blocks'], embedd_2, config['dropout_rate'])(regression_node)

	regression_node = layers.LSTM(lstm_units)(regression_node)  # (batch, embedding_dim)

	regression_embeddings = layers.Concatenate()(
		[regression_seq, regression_node]
	)  # Add()
	regression_embeddings = layers.Reshape((1, 2 * lstm_units))(
		regression_embeddings
	)  # (batch, 1, embedding_dim)
	regression_targets = layers.Dense(num_nodes, activation=None)(regression_embeddings)

	if not training:
		model = keras.Model(
			inputs=inputs_nodes, outputs=[regression_targets, regression_embeddings]
		)
	else:
		model = keras.Model(inputs=inputs_nodes, outputs=regression_targets)
	return model


def gcn_sequense_sensor_error_model(input_shape, config=None, training=True):
	lstm_units = config["lstm_units"]
	embedd_1 = config["embedding_dim"][0]
	embedd_2 = config["embedding_dim"][1]
	output_dim_1 = config["output_dim"][0]
	output_dim_2 = config["output_dim"][1]
	num_seq, num_nodes = input_shape
	# modeling sensor nodes
	gcn_node = LSTMGC(
		1,
		embedd_1,
		config["lstm_units"],
		num_seq,
		output_dim_1,
		config["graph_info_nodes"],
	)

	# modeling timestamp
	gcn_seq = LSTMGC(
		1,
		embedd_2,
		config["lstm_units"],
		num_nodes,
		output_dim_2,
		config["graph_info_seq"],
	)

	inputs = layers.Input(shape=input_shape)  # (batch, seq, nodes)

	inputs_nodes = layers.Reshape((num_seq, num_nodes, 1))(
		inputs
	)  # (batch, seq, nodes, 1)
	inputs_seq = layers.Reshape((num_nodes, num_seq, 1))(
		inputs
	)  # (batch, nodes, seq, 1)

	outputs_nodes = gcn_node(
		inputs_nodes
	)  # output shape (batch_size, output_dim_1 x num_nodes]
	outputs_seq = gcn_seq(
		inputs_seq
	)  # output shape (batch_size, output_dim_2 x num_seq]

	regression_embeddings = layers.Concatenate()([outputs_nodes, outputs_seq])  # Add()
	regression_embeddings = layers.Reshape((1, -1))(
		regression_embeddings
	)  # (batch, 1, embedding_dim)
	regression_targets = layers.Dense(num_nodes)(regression_embeddings)

	if not training:
		model = keras.Model(
			inputs=inputs, outputs=[regression_targets, regression_embeddings]
		)
	else:
		model = keras.Model(inputs=inputs, outputs=regression_targets)
	return model


def gcn_sequense_sensor_error_model_v1(input_shape, config=None, training=True):
	lstm_units = config["lstm_units"]
	embedd_1 = config["embedding_dim"][0]
	# embedd_2 = config['embedding_dim'][1]
	output_dim_1 = config["output_dim"][0]
	# output_dim_2 = config['output_dim'][1]
	num_seq, num_nodes = input_shape
	# modeling sensor nodes
	graph_conv_params = {
		"aggregation_type": "max",  # mean
		"combination_type": "concat",
		"activation": "relu",
	}

	gcn_node = LSTMGC(
		1,
		embedd_1,
		config["lstm_units"],
		num_seq,
		output_dim_1,
		config["graph_info_nodes"],
		graph_conv_params,
		return_normal=True,
	)

	inputs = layers.Input(shape=input_shape)  # (batch, seq, nodes)

	inputs_nodes = layers.Reshape((num_seq, num_nodes, 1))(
		inputs
	)  # (batch, seq, nodes, 1)

	regression_targets = gcn_node(
		inputs_nodes
	)  # output shape (batch_size, output_dim_1, num_nodes]

	# regression_embeddings = layers.Concatenate()([outputs_nodes, outputs_seq])  #Add()
	# regression_embeddings = layers.Reshape((1, -1))(regression_embeddings) #(batch, 1, embedding_dim)
	regression_targets = layers.Dense(num_nodes)(regression_targets)

	if not training:
		model = keras.Model(
			inputs=inputs, outputs=[regression_targets, regression_targets]
		)
	else:
		model = keras.Model(inputs=inputs, outputs=regression_targets)
	return model


### GCN-LSTM reconstruction model using seq as graph nodes
def gscn_lstm_recons_model(input_shape, config=None, training=True):
	lstm_units = config["lstm_units"]  # should be out_nodes if not using dense
	embedd_1 = config["embedding_dim"][
		0
	]  # gcn out_dim for each sensor output some embedding of this size

	num_seq, num_nodes = input_shape
	# modeling sensor nodes
	gcn_seq = LSTMGC_recons(
		1, embedd_1, config["lstm_units"], num_seq, config["graph_info_seq"]
	)

	inputs = layers.Input(shape=input_shape)  # (batch, seq, nodes)

	inputs_nodes = layers.Reshape((num_seq, num_nodes, 1))(
		inputs
	)  # (batch, seq, nodes, 1)

	regression_targets = gcn_seq(
		inputs_nodes
	)  # output shape (batch_size, seq, lstm_units]

	regression_targets = layers.Dense(num_nodes)(
		regression_targets
	)  # dot his if lstm_units are not num_nodes

	if not training:
		model = keras.Model(
			inputs=inputs, outputs=[regression_targets, regression_targets]
		)
	else:
		model = keras.Model(inputs=inputs, outputs=regression_targets)
	return model


def build_sequence_embedder(input_shape=(None, None), config=None, training=False):
	# input of shape [batch_size, input_seq_len, sensors)
	num_seq = input_shape[0]

	model = config["model"]
	num_blocks = config["num_blocks"]
	embedding_dim = config["embedding_dim"]
	dropout_rate = config["dropout_rate"]
	positional_encoding = config["positional_encoding"]

	if model == "Transformer":
		blocks = keras.Sequential(
			[
				TransformerBlock(
					embedding_dim, config["MHA_blocks"], embedding_dim, dropout_rate
				)
				for _ in range(num_blocks)
			],
			name="representation",
		)

	elif model == "FNet":
		blocks = keras.Sequential(
			[
				FNetLayer(num_seq, embedding_dim, dropout_rate)
				for _ in range(num_blocks)
			],
			name="representation",
		)
	elif model == "MLPMixer":
		blocks = keras.Sequential(
			[
				MLPMixerLayer(num_seq, embedding_dim, dropout_rate)
				for _ in range(num_blocks)
			],
			name="representation",
		)
	elif model == "MLP_small":
		blocks = keras.Sequential(
			[
				# layers.Dense(units=embedding_dim),
				# tfa.layers.GELU(),
				# layers.ReLU(),
				layers.Dropout(rate=dropout_rate, name="target_dropout")
			],
			name="representation",
		)
	elif model == "conv_model":
		return build_conv_model(input_shape, training=training)

	elif model == "seq_sensor_model":
		return sequense_sensor_error_model(
			input_shape, config=config, training=training
		)

	elif model == "gcn_seq_sensor_model":
		return gcn_sequense_sensor_error_model(
			input_shape, config=config, training=training
		)

	elif model == "gcn_v1":  # _v1
		return gcn_sequense_sensor_error_model_v1(
			input_shape, config=config, training=training
		)

	elif model == "gcn_recons":  ## a different variant of gcn_v1
		return gscn_lstm_recons_model(input_shape, config=config, training=training)
	else:
		raise ValueError(
			"model type not Implemented. use one of {Transformer, MLPMixer, FNet, MLP_simple}"
		)

	# Specify Model architecture
	inputs = layers.Input(shape=input_shape)

	# x = layers.Permute((2, 1))(inputs)  # convert input to (batch_size, num_nodes, input_seq_len)

	x = layers.Dense(units=embedding_dim)(
		inputs
	)  # (batch_size, input_seq_len, embedding_dim) project input embedding dim length
	if positional_encoding:
		positions = tf.range(start=0, limit=num_nodes, delta=1)
		position_embedding = layers.Embedding(
			input_dim=num_nodes, output_dim=embedding_dim
		)(positions)
		x = x + position_embedding

	if model == "MLP_small":
		embedding = x
	else:  # Process x using the model blocks.
		embedding = blocks(x)

	if not config["reconstruction"] and num_seq > 1:
		# Apply global average pooling to generate a [batch_size, 1, embedding_dim] tensor.
		embedding = layers.GlobalAveragePooling1D(name="embedding")(embedding)
	# regression_targets = layers.Dropout(rate=dropout_rate, name='target_dropout')(embedding)
	regression_targets = layers.Dense(input_shape[1])(embedding)
	# Create the Keras model.
	if not training:
		return keras.Model(inputs=inputs, outputs=[regression_targets, embedding])
	else:
		return keras.Model(inputs=inputs, outputs=regression_targets)


def get_model(
	input_shape,
	config_data,
	model_dir=None,
	load_weights=False,
	compile_model=True,
	training=True,
	monte_carlo_model=False,
):
	model = build_sequence_embedder(
		input_shape=input_shape, config=config_data, training=training
	)

	if monte_carlo_model and not training:
		model = load_model_mc(model, input_shape=input_shape, config=config_data)

	if config_data["optimizer"] == "adam":
		optim = keras.optimizers.legacy.Adam(
			learning_rate=config_data["learning_rate"]
		)  # switched to legacy

	elif config_data["optimizer"] == "adamw":
		# Create Adam optimizer with weight decay.
		optim = tfa.optimizers.AdamW(
			learning_rate=config_data["learning_rate"],
			weight_decay=config_data["weight_decay"],
		)
	elif config_data["optimizer"] == "rmsprop":
		optim = keras.optimizers.legacy.RMSprop(
			learning_rate=config_data["learning_rate"]
		)  # switched to legacy
	else:
		optim = keras.optimizers.legacy.SGD(
			learning_rate=config_data["learning_rate"], momentum=0.9
		)  # switched to legacy

	if load_weights:
		latest_checkpoint = tf.train.latest_checkpoint(model_dir)
		model.load_weights(latest_checkpoint).expect_partial()
		print("Loaded pretrained_checkpoint")
	if compile_model:
		model.compile(
			optimizer=optim,
			loss=keras.losses.MeanSquaredError(),  #
			# metrics=[keras.metrics.RootMeanSquaredError()],
		)
	return model


def train_embedder(
	dataset: GeneralDataset,
	dataset_trace=None,
	config_path="./conifgs/..",
	load_weights=False,
	training=True,
):
	config_name = os.path.basename(config_path)
	# Read config
	with open(config_path) as f:
		config_data = yaml.safe_load(f)

	# handle missing key in old configs, set if doesnt exist
	if "reconstruction" not in config_data:
		config_data["reconstruction"] = False
		print("reconstruction dataset mode not found in config... setting it to False")

	if dataset_trace is None:
		checkpoint_folder = Path(
			DEFAULT_RESOURCE_DIR,
			"model_checkpoints",
			dataset[0],
			config_name.split(".")[0],
		)
	else:
		checkpoint_folder = Path(
			DEFAULT_RESOURCE_DIR,
			"model_checkpoints",
			dataset[0],
			str(dataset_trace),
			config_name.split(".")[0],
		)

	checkpoint_folder.mkdir(parents=True, exist_ok=True)

	# read dataset
	trainset, valset, testset, _ = read_data(
		dataset,
		dataset_trace=dataset_trace,
		preprocess=config_data["preprocess"],
		config=config_data,
	)

	# graph Info for gcn
	match = re.search("gcn", config_data["model"])
	if match:
		graph_info_nodes, graph_info_seq = get_graph_info(
			trainset, config_data, verbose=True
		)
		config_data["graph_info_nodes"] = graph_info_nodes
		config_data["graph_info_seq"] = graph_info_seq

	# get data loaders
	train_dataset = get_dataloader(
		trainset,
		batch_size=config_data["batch_size"],
		input_sequence_length=config_data["input_sequence_length"],
		reconstruction_dataset=config_data["reconstruction"],
		shuffle=True,
	)
	val_dataset = get_dataloader(
		valset,
		batch_size=config_data["batch_size"],
		input_sequence_length=config_data["input_sequence_length"],
		reconstruction_dataset=config_data["reconstruction"],
	)

	# model inout_shape (dataloader output [batch_size, input_sequence_length, nodes])
	input_shape = (config_data["input_sequence_length"], trainset.shape[1])

	model = get_model(
		input_shape,
		config_data,
		model_dir=checkpoint_folder,
		load_weights=load_weights,
		compile_model=True,
		training=training,
	)

	# checkpointer
	checkpointer = keras.callbacks.ModelCheckpoint(
		filepath=Path(checkpoint_folder, "weights_best"),
		monitor="val_loss",
		mode="min",
		save_best_only=True,
		save_weights_only=True,
		verbose=1,
	)
	# Create a learning rate scheduler callback.
	reduce_lr = keras.callbacks.ReduceLROnPlateau(
		monitor="val_loss", factor=0.9, patience=15
	)
	# Create an early stopping callback.
	early_stopping = keras.callbacks.EarlyStopping(
		monitor="val_loss", patience=10, restore_best_weights=True
	)
	# Fit the model.
	history = model.fit(
		train_dataset,
		validation_data=val_dataset,
		epochs=config_data["epochs"],
		callbacks=[checkpointer, reduce_lr, early_stopping],  #
	)

	return history


######## Model Evaluation ########


def test_embedder(
	dataset: GeneralDataset,
	dataset_trace=None,
	config_path="./MLPmixer_blocks_2.yaml",
	load_weights=True,
	training=False,
	subset="test",
	do_MC=False,
	output_model=False,
):
	config_name = os.path.basename(config_path)
	dataset_name = dataset.value
	# Read config
	with open(config_path) as f:
		config_data = yaml.safe_load(f)

	# handle missing key in old configs, set if doesnt exist
	if "reconstruction" not in config_data:
		config_data["reconstruction"] = False

	# prep checkpoint folder
	if dataset_trace is None:
		checkpoint_folder = Path(
			DEFAULT_RESOURCE_DIR,
			"model_checkpoints",
			dataset_name,
			config_name.split(".")[0],
		)
	else:
		checkpoint_folder = Path(
			DEFAULT_RESOURCE_DIR,
			"model_checkpoints",
			dataset_name,
			str(dataset_trace),
			config_name.split(".")[0],
		)

	# checkpoint_folder.mkdir(parents=True, exist_ok=True)
	print(f"check_path: {checkpoint_folder}")
	# read dataset
	trainset, _, testset, labels = read_data(
		dataset,
		dataset_trace=dataset_trace,
		preprocess=config_data["preprocess"],
		config=config_data,
	)

	# graph Info for gcn
	match = re.search("gcn", config_data["model"])
	if match:
		graph_info_nodes, graph_info_seq = get_graph_info(
			trainset, config_data, verbose=True
		)
		config_data["graph_info_nodes"] = graph_info_nodes
		config_data["graph_info_seq"] = graph_info_seq

	if not config_data["reconstruction"]:
		target_offset = config_data[
			"input_sequence_length"
		]  # input_sequence_length + forecast_horizon - 1
		gt_labels = labels[target_offset:]
	else:
		labels_gen = get_label_loader(labels, config_data)
		gt_labels = []
		for gt in labels_gen.as_numpy_iterator():
			gt_labels.append(gt)
		gt_labels = np.vstack(gt_labels)

	# get data loaders
	if subset == "train":
		test_dataset = get_dataloader(
			trainset,
			batch_size=config_data["batch_size"],
			input_sequence_length=config_data["input_sequence_length"],
			reconstruction_dataset=config_data["reconstruction"],
		)
	else:
		test_dataset = get_dataloader(
			testset,
			batch_size=config_data["batch_size"],
			input_sequence_length=config_data["input_sequence_length"],
			reconstruction_dataset=config_data["reconstruction"],
		)

	# model inout_shape (dataloader output [batch_size, input_sequence_length, nodes])
	input_shape = (config_data["input_sequence_length"], testset.shape[1])

	model = get_model(
		input_shape,
		config_data,
		model_dir=checkpoint_folder,
		load_weights=load_weights,
		compile_model=True,
		training=training,
		monte_carlo_model=do_MC,
	)

	inputs = []
	predictions = []
	orig_target = []
	embeddings = []
	for input_batch, target in test_dataset:
		pred, embedding = model.predict_on_batch(input_batch)
		predictions.append(pred)
		embeddings.append(embedding)
		inputs.append(input_batch.numpy())
		if config_data["reconstruction"]:
			orig_target.append(input_batch.numpy())
		else:
			orig_target.append(target.numpy())

	if output_model:
		inputs = np.vstack(inputs)
		predictions = np.vstack(predictions)
		embeddings = np.vstack(embeddings)
		orig_target = np.vstack(orig_target)
		return (
			inputs,
			predictions.squeeze(),
			orig_target.squeeze(),
			gt_labels,
			embeddings.squeeze(),
			model,
		)
	else:
		predictions = np.vstack(predictions).squeeze()
		embeddings = np.vstack(embeddings).squeeze()
		orig_target = np.vstack(orig_target).squeeze()
		return predictions, orig_target, gt_labels, embeddings
