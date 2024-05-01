"""
modelling utilities for graph neural networks

"""
from pathlib import Path
import random
from sklearn import metrics
import scipy.sparse as sp
import numpy as np
import typing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# data_dir = str(Path.cwd())
module_path = str(Path.cwd().parents[0])


class GraphInfo:
	def __init__(self, edges: typing.Tuple[list, list], num_nodes: int):
		self.edges = edges
		self.num_nodes = num_nodes


def compute_finch_adjacency_matrix(train_array, symmetric=True):
	"""
	train_array : N x d (timestamps x sensors)
	"""
	train_array = train_array.T
	s = train_array.shape[0]
	orig_dist = metrics.pairwise.pairwise_distances(
		train_array, train_array, metric="cosine"
	)
	np.fill_diagonal(orig_dist, 1e12)
	initial_rank = np.argmin(orig_dist, axis=1)
	A = sp.csr_matrix(
		(np.ones_like(initial_rank, dtype=np.float32), (np.arange(0, s), initial_rank)),
		shape=(s, s),
	)
	A = A + sp.eye(s, dtype=np.float32, format="csr")
	if symmetric:
		A = A @ A.T
		A = A.sign()
	A = A.tolil()
	A.setdiag(0)
	return A


def adj_gdn(train_array, topk=30):
	train_array = train_array.T
	orig_dist = metrics.pairwise.pairwise_distances(
		train_array, train_array, metric="euclidean"
	)
	adj = np.zeros_like(orig_dist)
	top_idx = np.argsort(orig_dist, axis=-1)
	top_idx = top_idx[:, :topk]
	for i in range(orig_dist.shape[0]):
		adj[i, top_idx[i]] = 1
	return adj


def compute_adj_on_slices(arr, seq_len=16, slices=500):
	random.seed(9001)
	num_samples, dim = arr.shape
	ind = random.sample(range(num_samples - 2 * seq_len), slices)
	adj = sp.csr_matrix((seq_len, seq_len), dtype=np.float32)
	adj = adj.tolil()
	connsensus_threshold = 0.15 * slices
	for r in ind:
		window = arr[r : r + seq_len, :]
		adj += compute_finch_adjacency_matrix(window.T)
	adj = (adj.todense() >= connsensus_threshold).astype(np.float32)
	return adj


def graph_info(adj, verbose=True):
	node_indices, neighbor_indices = np.where(adj == 1.0)
	graph = GraphInfo(
		edges=(node_indices.tolist(), neighbor_indices.tolist()),
		num_nodes=adj.shape[0],
	)
	print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")
	return graph


def get_graph_info(array, config, verbose=True, gdn_adj=False, gdn_topk=30):
	if not gdn_adj:
		adj_n = compute_finch_adjacency_matrix(array)
		adj_n = adj_n.todense()
	else:
		adj_n = adj_gdn(array, topk=gdn_topk)
	graph_info_nodes = graph_info(adj_n, verbose=True)
	adj_seq = compute_adj_on_slices(
		array, seq_len=config["input_sequence_length"], slices=500
	)
	graph_info_seq = graph_info(adj_seq, verbose=True)
	return graph_info_nodes, graph_info_seq


class GraphConv(layers.Layer):
	"""
	Graph convolution layer implementation for Keras


	"""

	def __init__(
		self,
		in_feat,
		out_feat,
		graph_info: GraphInfo,
		aggregation_type="mean",
		combination_type="concat",
		activation: typing.Optional[str] = None,
		**kwargs,
	):
		super().__init__(**kwargs)
		self.in_feat = in_feat
		self.out_feat = out_feat
		self.graph_info = graph_info
		self.aggregation_type = aggregation_type
		self.combination_type = combination_type
		self.weight = tf.Variable(
			initial_value=keras.initializers.glorot_uniform()(
				shape=(in_feat, out_feat), dtype="float32"
			),
			trainable=True,
		)
		self.activation = layers.Activation(activation)

	def aggregate(self, neighbour_representations: tf.Tensor):
		aggregation_func = {
			"sum": tf.math.unsorted_segment_sum,
			"mean": tf.math.unsorted_segment_mean,
			"max": tf.math.unsorted_segment_max,
		}.get(self.aggregation_type)

		if aggregation_func:
			return aggregation_func(
				neighbour_representations,
				self.graph_info.edges[0],
				num_segments=self.graph_info.num_nodes,
			)

		raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")

	def compute_nodes_representation(self, features: tf.Tensor):
		"""Computes each node's representation.

		The nodes' representations are obtained by multiplying the features tensor with
		`self.weight`. Note that
		`self.weight` has shape `(in_feat, out_feat)`.

		Args:
		    features: Tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`

		Returns:
		    A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`
		"""
		return tf.matmul(features, self.weight)

	def compute_aggregated_messages(self, features: tf.Tensor):
		neighbour_representations = tf.gather(features, self.graph_info.edges[1])
		aggregated_messages = self.aggregate(neighbour_representations)
		return tf.matmul(aggregated_messages, self.weight)

	def update(self, nodes_representation: tf.Tensor, aggregated_messages: tf.Tensor):
		if self.combination_type == "concat":
			h = tf.concat([nodes_representation, aggregated_messages], axis=-1)
		elif self.combination_type == "add":
			h = nodes_representation + aggregated_messages
		else:
			raise ValueError(f"Invalid combination type: {self.combination_type}.")

		return self.activation(h)

	def call(self, features: tf.Tensor):
		"""Forward pass.

		Args:
		    features: tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`

		Returns:
		    A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`
		"""
		nodes_representation = self.compute_nodes_representation(features)
		aggregated_messages = self.compute_aggregated_messages(features)
		return self.update(nodes_representation, aggregated_messages)


class LSTMGC(layers.Layer):
	"""Layer comprising a convolution layer followed by LSTM and dense layers."""

	def __init__(
		self,
		in_feat,
		out_feat,
		lstm_units: int,
		input_seq_len: int,
		output_seq_len: int,
		graph_info: GraphInfo,
		graph_conv_params: typing.Optional[dict] = None,
		return_normal: bool = False,
		**kwargs,
	):
		super().__init__(**kwargs)

		# graph conv layer
		if graph_conv_params is None:
			graph_conv_params = {
				"aggregation_type": "mean",
				"combination_type": "concat",
				"activation": None,
			}
		self.return_normal = return_normal
		self.graph_conv = GraphConv(in_feat, out_feat, graph_info, **graph_conv_params)

		self.lstm = layers.LSTM(lstm_units, activation="tanh")

		self.dense = layers.Dense(output_seq_len, activation="relu")

		self.input_seq_len, self.output_seq_len = input_seq_len, output_seq_len

	def call(self, inputs):
		"""Forward pass.
		    (batch, seq, nodes, 1)
		Args:
		    inputs: tf.Tensor of shape `(batch_size, input_seq_len, num_nodes, in_feat)`

		Returns:
		    A tensor of shape `(batch_size, output_seq_len, num_nodes)`.
		"""

		# convert shape to  (num_nodes, batch_size, input_seq_len, in_feat)
		inputs = tf.transpose(inputs, [2, 0, 1, 3])

		gcn_out = self.graph_conv(
			inputs
		)  # gcn_out has shape: (num_nodes, batch_size, input_seq_len, out_feat)
		shape = tf.shape(gcn_out)
		num_nodes, batch_size, input_seq_len, out_feat = (
			shape[0],
			shape[1],
			shape[2],
			shape[3],
		)

		# LSTM takes only 3D tensors as input
		gcn_out = tf.reshape(gcn_out, (batch_size * num_nodes, input_seq_len, out_feat))
		lstm_out = self.lstm(
			gcn_out
		)  # lstm_out has shape: (batch_size * num_nodes, lstm_units)

		dense_output = self.dense(
			lstm_out
		)  # dense_output has shape: (batch_size * num_nodes, output_seq_len)
		output = tf.reshape(dense_output, (num_nodes, batch_size, self.output_seq_len))

		if self.return_normal:
			output = tf.transpose(
				output, [1, 2, 0]
			)  # returns Tensor of shape (batch_size, output_seq_len, num_nodes)
		else:
			output = tf.reshape(
				output, (batch_size, self.output_seq_len * num_nodes)
			)  # returns 2D Tensor of shape (batch_size, output_seq_len x num_nodes]

		return output


class LSTMGC_recons(layers.Layer):
	"""reconstruction layer comprising a convolution layer followed by LSTM and dense layers"""

	def __init__(
		self,
		in_feat,
		out_feat,
		lstm_units: int,
		input_seq_len: int,
		graph_info: GraphInfo,
		graph_conv_params: typing.Optional[dict] = None,
		**kwargs,
	):
		super().__init__(**kwargs)

		# graph conv layer
		if graph_conv_params is None:
			graph_conv_params = {
				"aggregation_type": "mean",
				"combination_type": "concat",
				"activation": None,
			}

		self.graph_conv = GraphConv(in_feat, out_feat, graph_info, **graph_conv_params)

		self.lstm = layers.LSTM(lstm_units, activation="tanh")
		self.lstm_units = lstm_units

		self.input_seq_len = input_seq_len

	def call(self, inputs):
		"""Forward pass.
		    (batch, seq, nodes, 1)
		Args:
		    inputs: tf.Tensor of shape `(batch_size, input_seq_len, num_nodes, in_feat)`

		Returns:
		    A tensor of shape `(batch_size, output_seq_len, num_nodes)`.
		"""

		# convert shape to  (input_seq_len, batch_size, num_nodes, in_feat) (num_nodes, batch_size, input_seq_len, in_feat)
		inputs = tf.transpose(inputs, [1, 0, 2, 3])

		gcn_out = self.graph_conv(
			inputs
		)  # gcn_out has shape: (input_seq_len, batch_size, num_nodes, out_feat)
		shape = tf.shape(gcn_out)
		input_seq_len, batch_size, num_nodes, out_feat = (
			shape[0],
			shape[1],
			shape[2],
			shape[3],
		)

		# LSTM takes only 3D tensors as input
		gcn_out = tf.reshape(gcn_out, (batch_size * input_seq_len, num_nodes, out_feat))
		lstm_out = self.lstm(
			gcn_out
		)  # lstm_out has shape: (batch_size * input_seq_len, lstm_units)  ## Note if not usingdense then lstm_units should be num_nodes

		# lstm_out = layers.Dense(num_nodes, activation=None)(lstm_out)

		# dense_output has shape: (batch_size * input_seq_len, num_nodes)
		output = tf.reshape(lstm_out, (input_seq_len, batch_size, self.lstm_units))

		output = tf.transpose(
			output, [1, 0, 2]
		)  # returns Tensor of shape (batch_size, output_seq_len, lstm_units)
		return output
