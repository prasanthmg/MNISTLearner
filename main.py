import numpy
import random

LAYERS = [784, 60, 10]
MINI_BATCH_SIZE = 10
EPOCHS = 15
LEARNING_RATE = 0.5
REGULARIZATION_PARAMETER = 10

def load_data():
	training_inputs  = load_image_file("train-images.idx3-ubyte")	#ndarray, shape: 784, 60k
	training_outputs = load_label_file("train-labels.idx1-ubyte")	#ndarray, shape: 10, 60k
	test_inputs = load_image_file("t10k-images.idx3-ubyte")			#ndarray, shape: 784, 10k
	test_outputs = load_label_file("t10k-labels.idx1-ubyte")		#ndarray, shape: 10, 10k
	return (training_inputs, training_outputs), (test_inputs, test_outputs)

def load_image_file(path):
	with open(path,"rb") as f:
		return  (1/255)*(numpy.array(numpy.frombuffer(f.read(), dtype=numpy.uint8, offset=16))).reshape(784, -1, order='F')

def load_label_file(path):
	with open(path,"rb") as f:
		f.read(8)
		labels = f.read()
		labels_matrix = numpy.zeros((10, len(labels)), dtype=numpy.uint8)
		for i,label in enumerate(labels):
			labels_matrix[label][i] = 1
		return labels_matrix

class Network:
	def __init__(self):
		self.weights = []
		self.biases = []
		self.initialise_weights()
		self.initialise_biases()

	def initialise_weights(self):
		for i in range(1,len(LAYERS)):
			weight_matrix = numpy.random.randn(LAYERS[i], LAYERS[i-1])	#normal distribution with mean=0 and sd=1
			self.weights.append(weight_matrix)

	def initialise_biases(self):
		for i in range(1, len(LAYERS)):
			bias_matrix = numpy.random.randn(LAYERS[i], 1)
			self.biases.append(bias_matrix) 

	def train(self, training_data):
		mini_batches = self.get_mini_batches(training_data)
		for mini_batch in mini_batches:
			#forward pass
			activation_matrices = [mini_batch[0]]
			for weight_matrix, bias_matrix in zip(self.weights, self.biases):
				activation_matrices.append(self.sigmoid(numpy.matmul(weight_matrix, activation_matrices[-1]) + bias_matrix))
			#backpropogation
			# delta_matrices = [(activation_matrices[-1] - mini_batch[1])*(activation_matrices[-1] * (1 - activation_matrices[-1]))] #for quadratic cost
			delta_matrices = [activation_matrices[-1] - mini_batch[1]]	#for cross-entropy cost
			i=-2
			while i>(-len(LAYERS)):
				delta_matrices.insert(0, numpy.matmul(self.weights[i+1].T, delta_matrices[0]) * (activation_matrices[i] * (1 - activation_matrices[i])))
				i-=1
			#weight & bias updation
			for i in range(len(self.biases)):
				self.biases[i] -= (LEARNING_RATE/MINI_BATCH_SIZE) * numpy.sum(delta_matrices[i], axis=1, keepdims=True)
				self.weights[i] -= (LEARNING_RATE)*(((1/MINI_BATCH_SIZE)*numpy.matmul(delta_matrices[i], activation_matrices[i].T)) + ((REGULARIZATION_PARAMETER/(MINI_BATCH_SIZE * len(mini_batches)))*self.weights[i])) #with regularization
				# self.weights[i] -= (LEARNING_RATE)*(((1/MINI_BATCH_SIZE)*numpy.matmul(delta_matrices[i], activation_matrices[i].T))) #without regularization

	def test(self, test_data):
		activation_matrix = test_data[0]
		for weight_matrix, bias_matrix in zip(self.weights, self.biases):
			activation_matrix = self.sigmoid(numpy.matmul(weight_matrix, activation_matrix) + bias_matrix)
		activation_matrix = (activation_matrix >= numpy.sort(activation_matrix, axis=0)[[-1], :]).astype(numpy.uint8)
		success_count = test_data[1].shape[1] - numpy.count_nonzero(numpy.sum(abs(activation_matrix - test_data[1]), axis=0))
		return (str(success_count) +' / '+ str(test_data[1].shape[1]))


	def get_mini_batches(self, training_data):
		l = list(range(training_data[0].shape[1]))
		random.shuffle(l)
		mini_batches = []
		i=0
		while i<len(l):
			mini_batch_input_matrix = training_data[0][:, l[i:i+MINI_BATCH_SIZE]]
			mini_batch_output_matrix = training_data[1][:, l[i:i+MINI_BATCH_SIZE]]
			mini_batches.append((mini_batch_input_matrix, mini_batch_output_matrix))
			i+=MINI_BATCH_SIZE
		return mini_batches

	def sigmoid(self, z):
		return 1.0/(1.0 +numpy.exp(-z))

training_data, test_data = load_data()
network = Network()
print("Initial succeess_rate = " + network.test(test_data))
for epoch in range(EPOCHS):
	network.train(training_data)
	print("After epoch "+ str(epoch+1) +", succeess_rate = " + network.test(test_data))